/**
 * The Orient & Parting Line tool's state and actions.
 *
 * Every button computes its result here, in the browser, with the port of
 * `core/orientation.py` in `lib/orientation.ts`: the part moves on the frame
 * the button is pressed, and the placement is then written through the
 * ordinary part PUT. Only the widest-section suggestion and the cope/drag
 * volume split need the server, because they slice the mesh; those are
 * fetched in the background and shown when they arrive.
 *
 * The web client had no undo, so this tool keeps its own: every action
 * records the placements and parting line it is about to change, and
 * "Undo" puts back the last of them.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react';

import { api, type PartingLine, type PartingLineInfo, type ProjectPart } from '../lib/api';
import {
  boundsCenter,
  dropToFloorDz,
  eulerToMatrix,
  faceToDirection,
  partWorldBounds,
  placementOf,
  rotateAboutWorldAxis,
  snapToPartingDz,
  transpose,
  applyDelta,
  type PartingSide,
  type Vec3,
  type WorldAxis,
  type WorldBounds
} from '../lib/orientation';
import type { OrientPick, OrientPickMode, OrientViewState } from '../lib/viewState';

export const ROTATION_STEPS = [90, 45, 15, 5] as const;
export const HEIGHT_STEPS = [0.1, 0.25, 0.5, 1] as const;

/** Parts that sit on the cope side of the parting line by default. */
const SEAT_ABOVE: readonly string[] = ['riser', 'riser_post', 'sprue', 'pour_cup'];
/** Parts that hang below it, in the drag, by default. */
const HANG_BELOW: readonly string[] = ['runner', 'gate'];

const UNDO_LIMIT = 50;
/** How long the parting line waits for the last nudge before it is saved. */
const PARTING_SAVE_DEBOUNCE_MS = 300;
/** How long the cope/drag split waits for the last change before it is measured. */
const INFO_DEBOUNCE_MS = 350;

type PlacementFields = Pick<
  ProjectPart,
  'position_x' | 'position_y' | 'position_z' | 'rotation_x' | 'rotation_y' | 'rotation_z'
>;
type UndoEntry = { label: string; parts: Array<{ id: string } & PlacementFields>; parting: PartingLine | null };

export type OrientNotify = {
  succeed: (message: string) => void;
  warn: (message: string) => void;
  fail: (error: unknown, fallback?: string) => void;
};

/** The side a part type sits on by default, or null for no suggestion. */
export function suggestedPartingSide(partType: string): PartingSide | null {
  if (SEAT_ABOVE.includes(partType)) return 'above';
  if (HANG_BELOW.includes(partType)) return 'below';
  return null;
}

/** The casting the parting line belongs to: the first visible cast body, else the first cast body. */
export function castingOf(parts: ProjectPart[]): ProjectPart | null {
  const castings = parts.filter((part) => part.part_type === 'cast');
  return castings.find((part) => part.visible !== false) ?? castings[0] ?? null;
}

function placementFields(part: ProjectPart): PlacementFields {
  return {
    position_x: part.position_x,
    position_y: part.position_y,
    position_z: part.position_z,
    rotation_x: part.rotation_x,
    rotation_y: part.rotation_y,
    rotation_z: part.rotation_z
  };
}

/** Float noise off a coordinate, so 1.0000000000002 is stored and shown as 1. */
function tidy(value: number): number {
  const rounded = Math.round(value * 1e6) / 1e6;
  return rounded === 0 ? 0 : rounded;
}

function fieldsFrom(rotation: Vec3, position: Vec3): PlacementFields {
  return {
    rotation_x: rotation[0],
    rotation_y: rotation[1],
    rotation_z: rotation[2],
    position_x: tidy(position[0]),
    position_y: tidy(position[1]),
    position_z: tidy(position[2])
  };
}

export type OrientOptions = {
  projectId: string | null;
  parts: ProjectPart[];
  setParts: Dispatch<SetStateAction<ProjectPart[]>>;
  selectedPartId: string | null;
  setSelectedPartId: (id: string | null) => void;
  partingLine: PartingLine | null;
  setPartingLine: Dispatch<SetStateAction<PartingLine | null>>;
  /** The tool's panel is open. */
  active: boolean;
  notify: OrientNotify;
};

export function useOrient(options: OrientOptions) {
  const { projectId, parts, setParts, selectedPartId, setSelectedPartId, partingLine, setPartingLine, active, notify } =
    options;

  const [rotationStep, setRotationStep] = useState<number>(90);
  const [heightStep, setHeightStep] = useState<number>(0.25);
  const [keepOnFloor, setKeepOnFloor] = useState(true);
  const [pick, setPick] = useState<OrientPickMode | null>(null);
  const [undoStack, setUndoStack] = useState<UndoEntry[]>([]);
  const [info, setInfo] = useState<PartingLineInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [measuring, setMeasuring] = useState(false);

  // Read by the actions, which can fire twice before React re-renders (a
  // double click): each must see the placement the one before it wrote.
  const partsRef = useRef(parts);
  partsRef.current = parts;
  const partingRef = useRef(partingLine);
  partingRef.current = partingLine;
  /** Part writes, chained per part so the server sees them in the order they were made. */
  const writeChains = useRef(new Map<string, Promise<unknown>>());

  const casting = useMemo(() => castingOf(parts), [parts]);
  const target = useMemo(
    () => parts.find((part) => part.id === selectedPartId) ?? casting ?? null,
    [parts, selectedPartId, casting]
  );
  const targetBounds = useMemo(() => (target ? partWorldBounds(target) : null), [target]);
  const castingBounds = useMemo(() => (casting ? partWorldBounds(casting) : null), [casting]);

  // A pick belongs to the open tool and the project it was armed in.
  useEffect(() => {
    if (!active) setPick(null);
  }, [active]);
  useEffect(() => {
    setPick(null);
    setUndoStack([]);
    setInfo(null);
    setInfoError(null);
  }, [projectId]);

  // --- Writes ---------------------------------------------------------------

  const writePart = useCallback(
    (part: ProjectPart) => {
      if (!projectId) return;
      const previous = writeChains.current.get(part.id) ?? Promise.resolve();
      const next = previous
        .catch(() => undefined)
        .then(() => api.updatePart(projectId, part))
        .catch((error: unknown) => notify.fail(error, 'Unable to save the part placement'));
      writeChains.current.set(part.id, next);
    },
    [projectId, notify]
  );

  /** Set placements in state and on the server, after recording them for Undo. */
  const commit = useCallback(
    (label: string, patches: Array<{ id: string } & Partial<PlacementFields>>, parting?: PartingLine | null) => {
      const current = partsRef.current;
      const before = patches
        .map((patch) => current.find((part) => part.id === patch.id))
        .filter((part): part is ProjectPart => Boolean(part))
        .map((part) => ({ id: part.id, ...placementFields(part) }));
      setUndoStack((stack) => [...stack.slice(-(UNDO_LIMIT - 1)), { label, parts: before, parting: partingRef.current }]);

      const byId = new Map(patches.map((patch) => [patch.id, patch]));
      const next = current.map((part) => {
        const patch = byId.get(part.id);
        return patch ? { ...part, ...patch } : part;
      });
      partsRef.current = next;
      setParts((state) =>
        state.map((part) => {
          const patch = byId.get(part.id);
          return patch ? { ...part, ...patch } : part;
        })
      );
      for (const part of next) if (byId.has(part.id)) writePart(part);
      if (parting !== undefined) {
        partingRef.current = parting;
        setPartingLine(parting);
      }
    },
    [setParts, setPartingLine, writePart]
  );

  /** Change only the parting line, recorded for Undo. */
  const commitParting = useCallback(
    (label: string, parting: PartingLine | null) => {
      setUndoStack((stack) => [...stack.slice(-(UNDO_LIMIT - 1)), { label, parts: [], parting: partingRef.current }]);
      partingRef.current = parting;
      setPartingLine(parting);
    },
    [setPartingLine]
  );

  // The parting line is saved on a short debounce: a run of nudges is one write.
  // The shell sets the project id and its stored parting line in one update,
  // so the first value seen for a project is what it was opened with.
  const savedParting = useRef<{ projectId: string | null; key: string }>({ projectId: null, key: '' });
  useEffect(() => {
    const key = JSON.stringify(partingLine);
    if (savedParting.current.projectId !== projectId) {
      savedParting.current = { projectId, key };
      return;
    }
    if (!projectId || key === savedParting.current.key) return;
    const timer = window.setTimeout(() => {
      savedParting.current = { projectId, key };
      api.savePartingLine(projectId, partingLine).catch((error: unknown) => notify.fail(error, 'Unable to save the parting line'));
    }, PARTING_SAVE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [projectId, partingLine, notify]);

  // --- Helpers --------------------------------------------------------------

  const latest = useCallback((id: string | undefined | null) => partsRef.current.find((part) => part.id === id) ?? null, []);

  /** The part a button acts on, re-read so a double click sees the first click's result. */
  const currentTarget = useCallback(() => {
    const parts = partsRef.current;
    const selected = parts.find((part) => part.id === selectedPartId);
    return selected ?? castingOf(parts);
  }, [selectedPartId]);

  const needGeometry = useCallback(
    (part: ProjectPart | null): { part: ProjectPart; bounds: WorldBounds } | null => {
      if (!part) {
        notify.warn('Add a casting first, then orient it.');
        return null;
      }
      const bounds = partWorldBounds(part);
      if (!bounds) {
        notify.warn(`"${part.name}" has no geometry loaded yet.`);
        return null;
      }
      return { part, bounds };
    },
    [notify]
  );

  /** Keep-on-floor: the Z shift that puts the new lowest point where the old one was. */
  const floorKeep = useCallback(
    (part: ProjectPart, before: WorldBounds, rotation: Vec3, position: Vec3): Vec3 => {
      if (!keepOnFloor) return position;
      const after = partWorldBounds({ ...part, ...fieldsFrom(rotation, position) });
      if (!after) return position;
      return [position[0], position[1], position[2] + (before[0][2] - after[0][2])];
    },
    [keepOnFloor]
  );

  /** Move a part straight up or down by `dz`, carrying a locked parting line with the casting. */
  const shiftZ = useCallback(
    (label: string, part: ProjectPart, dz: number) => {
      if (Math.abs(dz) < 1e-9) return;
      const patch = { id: part.id, position_z: tidy(part.position_z + dz) };
      const parting = partingRef.current;
      const isCasting = castingOf(partsRef.current)?.id === part.id;
      if (parting && parting.locked_to_casting && isCasting) {
        commit(label, [patch], { ...parting, z: tidy(parting.z + dz) });
      } else {
        commit(label, [patch]);
      }
    },
    [commit]
  );

  // --- Orientation ----------------------------------------------------------

  const rotate = useCallback(
    (axis: WorldAxis, sign: 1 | -1) => {
      const found = needGeometry(currentTarget());
      if (!found) return;
      const { part, bounds } = found;
      const placement = placementOf(part);
      const turned = rotateAboutWorldAxis(placement.rotation, placement.position, axis, sign * rotationStep, boundsCenter(bounds));
      const position = floorKeep(part, bounds, turned.rotation, turned.position);
      commit(`Rotate ${axis.toUpperCase()} ${sign > 0 ? '+' : '-'}${rotationStep}°`, [
        { id: part.id, ...fieldsFrom(turned.rotation, position) }
      ]);
    },
    [needGeometry, currentTarget, rotationStep, floorKeep, commit]
  );

  const faceTo = useCallback(
    (partId: string | null, normal: Vec3, direction: 'down' | 'up') => {
      const found = needGeometry(latest(partId) ?? currentTarget());
      if (!found) return;
      const { part, bounds } = found;
      const placement = placementOf(part);
      const turned = faceToDirection(
        placement.rotation,
        placement.position,
        normal,
        boundsCenter(bounds),
        direction === 'down' ? [0, 0, -1] : [0, 0, 1]
      );
      const position = floorKeep(part, bounds, turned.rotation, turned.position);
      commit(direction === 'down' ? 'Face down' : 'Face up', [{ id: part.id, ...fieldsFrom(turned.rotation, position) }]);
      if (part.id !== selectedPartId) setSelectedPartId(part.id);
    },
    [needGeometry, latest, currentTarget, floorKeep, commit, selectedPartId, setSelectedPartId]
  );

  const resetOrientation = useCallback(() => {
    const found = needGeometry(currentTarget());
    if (!found) return;
    const { part, bounds } = found;
    const placement = placementOf(part);
    // Undo the whole turn about the part's centre, so it comes back upright where it stands.
    const undoTurn = applyDelta(placement.rotation, placement.position, transpose(eulerToMatrix(placement.rotation)), boundsCenter(bounds));
    const rotation: Vec3 = [0, 0, 0];
    const position = floorKeep(part, bounds, rotation, undoTurn.position);
    commit('Reset orientation', [{ id: part.id, ...fieldsFrom(rotation, position) }]);
  }, [needGeometry, currentTarget, floorKeep, commit]);

  /** Exact Euler angles from the advanced fields. */
  const setExactRotation = useCallback(
    (rotation: Vec3) => {
      const found = needGeometry(currentTarget());
      if (!found) return;
      const { part, bounds } = found;
      const position = floorKeep(part, bounds, rotation, placementOf(part).position);
      commit('Exact rotation', [{ id: part.id, ...fieldsFrom(rotation, position) }]);
    },
    [needGeometry, currentTarget, floorKeep, commit]
  );

  // --- Height ---------------------------------------------------------------

  const nudgeHeight = useCallback(
    (sign: 1 | -1) => {
      const found = needGeometry(currentTarget());
      if (!found) return;
      shiftZ(sign > 0 ? 'Raise' : 'Lower', found.part, sign * heightStep);
    },
    [needGeometry, currentTarget, shiftZ, heightStep]
  );

  const setBottomZ = useCallback(
    (bottomZ: number) => {
      const found = needGeometry(currentTarget());
      if (!found) return;
      shiftZ('Set bottom Z', found.part, dropToFloorDz(found.bounds, bottomZ));
    },
    [needGeometry, currentTarget, shiftZ]
  );

  const dropToFloor = useCallback(() => {
    const found = needGeometry(currentTarget());
    if (!found) return;
    shiftZ('Drop to floor', found.part, dropToFloorDz(found.bounds, 0));
  }, [needGeometry, currentTarget, shiftZ]);

  const snapToParting = useCallback(
    (side: PartingSide) => {
      const parting = partingRef.current;
      if (!parting) {
        notify.warn('Set the parting line first.');
        return;
      }
      const found = needGeometry(currentTarget());
      if (!found) return;
      const dz = snapToPartingDz(found.bounds, parting.z, side);
      if (Math.abs(dz) < 1e-9) return;
      commit(side === 'above' ? 'Seat on parting line' : 'Hang from parting line', [
        { id: found.part.id, position_z: tidy(found.part.position_z + dz) }
      ]);
    },
    [needGeometry, currentTarget, commit, notify]
  );

  // --- Parting line ---------------------------------------------------------

  const suggestions = info?.suggestions ?? null;
  /** Preset heights, from the server when it has measured the casting, else from its box. */
  const presets = useMemo(() => {
    const bottom = castingBounds ? castingBounds[0][2] : null;
    const top = castingBounds ? castingBounds[1][2] : null;
    return {
      widest: suggestions?.max_section ?? null,
      top: top,
      bottom: bottom,
      center: bottom !== null && top !== null ? (bottom + top) / 2 : null
    };
  }, [castingBounds, suggestions]);

  const setPartingZ = useCallback(
    (label: string, z: number) => {
      const current = partingRef.current;
      commitParting(label, {
        z: tidy(z),
        visible: current?.visible ?? true,
        locked_to_casting: current?.locked_to_casting ?? true
      });
    },
    [commitParting]
  );

  const setPartingVisible = useCallback(
    (visible: boolean) => {
      const current = partingRef.current;
      if (current) {
        commitParting(visible ? 'Show parting line' : 'Hide parting line', { ...current, visible });
        return;
      }
      if (!visible) return;
      const z = presets.widest ?? presets.center;
      if (z === null) {
        notify.warn('Add a casting with geometry before setting a parting line.');
        return;
      }
      commitParting('Set parting line', { z: tidy(z), visible: true, locked_to_casting: true });
    },
    [commitParting, presets, notify]
  );

  const setPartingLocked = useCallback(
    (locked: boolean) => {
      const current = partingRef.current;
      if (current) commitParting(locked ? 'Lock to casting' : 'Unlock from casting', { ...current, locked_to_casting: locked });
    },
    [commitParting]
  );

  const nudgeParting = useCallback(
    (sign: 1 | -1) => {
      const current = partingRef.current;
      if (!current) return;
      setPartingZ(sign > 0 ? 'Raise parting line' : 'Lower parting line', current.z + sign * heightStep);
    },
    [setPartingZ, heightStep]
  );

  // --- Picks ----------------------------------------------------------------

  const armPick = useCallback((mode: OrientPickMode | null) => setPick((current) => (current === mode ? null : mode)), []);

  const handlePick = useCallback(
    (event: OrientPick) => {
      setPick(null);
      if (event.mode === 'pickHeight') {
        setPartingZ('Pick parting height', event.point[2]);
        notify.succeed(`Parting line set to Z = ${event.point[2].toFixed(2)} in.`);
        return;
      }
      if (!event.normal) {
        notify.warn('That click landed on a section cut, not a face. Click a face of the part.');
        return;
      }
      faceTo(event.partId, event.normal, event.mode === 'faceDown' ? 'down' : 'up');
    },
    [setPartingZ, faceTo, notify]
  );

  // --- Undo -----------------------------------------------------------------

  const undo = useCallback(() => {
    const entry = undoStack.at(-1);
    if (!entry) return;
    setUndoStack((stack) => stack.slice(0, -1));
    const byId = new Map(entry.parts.map((part) => [part.id, part]));
    const next = partsRef.current.map((part) => {
      const patch = byId.get(part.id);
      return patch ? { ...part, ...patch } : part;
    });
    partsRef.current = next;
    setParts((state) =>
      state.map((part) => {
        const patch = byId.get(part.id);
        return patch ? { ...part, ...patch } : part;
      })
    );
    for (const part of next) if (byId.has(part.id)) writePart(part);
    partingRef.current = entry.parting;
    setPartingLine(entry.parting);
  }, [undoStack, setParts, setPartingLine, writePart]);

  // --- Server measurements --------------------------------------------------

  const castingPlacementKey = casting
    ? `${casting.id}|${casting.filepath}|${casting.rotation_x},${casting.rotation_y},${casting.rotation_z}|` +
      `${casting.position_x},${casting.position_y},${casting.position_z}|${casting.scale}`
    : null;
  const partingZ = partingLine?.z ?? null;
  useEffect(() => {
    if (!active || !projectId || !casting || !castingPlacementKey) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setMeasuring(true);
      api
        .partingLineInfo(
          projectId,
          {
            partId: casting.id,
            rotation: [casting.rotation_x, casting.rotation_y, casting.rotation_z],
            position: [casting.position_x, casting.position_y, casting.position_z],
            scale: casting.scale || 1,
            z: partingZ
          },
          controller.signal
        )
        .then((payload) => {
          if (controller.signal.aborted) return;
          setInfo(payload);
          setInfoError(null);
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          setInfoError(error instanceof Error ? error.message : 'Unable to measure the casting');
        })
        .finally(() => {
          if (!controller.signal.aborted) setMeasuring(false);
        });
    }, INFO_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
    // `castingPlacementKey` stands for every casting field read above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, projectId, castingPlacementKey, partingZ]);

  /** Depths straight from the casting's box, so they follow every nudge without a round trip. */
  const depths = useMemo(() => {
    if (!partingLine || !castingBounds) return null;
    return {
      cope: Math.max(0, castingBounds[1][2] - partingLine.z),
      drag: Math.max(0, partingLine.z - castingBounds[0][2])
    };
  }, [partingLine, castingBounds]);

  /** The server's split, only while it describes the plane and placement on screen. */
  const split =
    info && partingLine && casting && info.part_id === casting.id && Math.abs(info.split_z - partingLine.z) < 1e-3
      ? info.split
      : null;

  const viewState: OrientViewState = useMemo(
    () => ({
      active,
      pick,
      parting: partingLine ? { z: partingLine.z, visible: partingLine.visible } : null
    }),
    [active, pick, partingLine]
  );

  return {
    target,
    targetBounds,
    casting,
    castingBounds,
    partingLine,
    presets,
    depths,
    split,
    measuring,
    infoError,
    rotationStep,
    setRotationStep,
    heightStep,
    setHeightStep,
    keepOnFloor,
    setKeepOnFloor,
    pick,
    armPick,
    cancelPick: () => setPick(null),
    handlePick,
    rotate,
    resetOrientation,
    setExactRotation,
    nudgeHeight,
    setBottomZ,
    dropToFloor,
    snapToParting,
    setPartingZ,
    setPartingVisible,
    setPartingLocked,
    nudgeParting,
    undo,
    undoLabel: undoStack.at(-1)?.label ?? null,
    viewState
  };
}

export type OrientController = ReturnType<typeof useOrient>;
