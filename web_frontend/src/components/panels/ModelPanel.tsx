import {
  ArrowDownUp,
  Box,
  Check,
  Copy,
  Crosshair,
  Layers2,
  Pencil,
  PenLine,
  Plus,
  RefreshCw,
  Ruler,
  Scissors,
  Square,
  Trash2,
  X
} from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';

import { Group, SetupPanel } from '../SetupPanel';
import { SKETCH_TOOLS } from '../sketchTools';
import type { ModelingController } from '../../hooks/useModeling';
import type { PartType, ProjectPart } from '../../lib/api';
import { pickLabel } from '../../lib/modeling/measure';
import {
  CUT_END_CONDITIONS,
  END_CONDITIONS,
  arcEnds,
  cutCandidates,
  entityLabel,
  featureOf,
  planeLabel,
  referencePlaneProblem,
  regionSelected,
  sketchProblem,
  STANDARD_PLANES,
  type CutEndCondition,
  type CutFeature,
  type EndCondition,
  type ExtrudeFeature,
  type SketchEntity
} from '../../lib/modeling/types';
import { PART_TYPES, partTypeColor, partTypeLabel } from '../../lib/partTypes';

export type ModelPanelProps = {
  modeling: ModelingController;
  parts: ProjectPart[];
  selectedPartId: string | null;
  onSelectPart: (partId: string | null) => void;
  onRemovePart: (partId: string) => void;
  onClose: () => void;
};

/** Body types offered for a modeled body, most used first. */
const BODY_TYPES: PartType[] = ['riser', 'chill', 'riser_post', 'core', 'cast', 'pour_cup', 'sprue', 'runner', 'gate', 'filter'];

export function ModelPanel(props: ModelPanelProps) {
  const { modeling } = props;
  if (modeling.measurePicks) return <MeasureManager {...props} />;
  if (modeling.planeSession) return <ReferencePlaneManager {...props} />;
  const session = modeling.session;
  if (!session) return <FeatureTree {...props} />;
  if (session.stage === 'plane') return <PlaneManager {...props} />;
  if (session.stage === 'sketch') return <SketchManager {...props} />;
  return session.feature.kind === 'cut' ? <CutManager {...props} /> : <ExtrudeManager {...props} />;
}

// ---------------------------------------------------------------------------
// FeatureManager tree
// ---------------------------------------------------------------------------

function FeatureTree({ modeling, parts, selectedPartId, onSelectPart, onRemovePart, onClose }: ModelPanelProps) {
  const imported = parts.filter((part) => !featureOf(part));
  const modeled = parts.filter((part) => featureOf(part));
  // Bodies and cuts in history order, the way the FeatureManager lists them.
  const history: Array<{ seq: number; part: ProjectPart; cut?: never } | { seq: number; cut: CutFeature; part?: never }> = [
    ...modeled.map((part) => ({ seq: featureOf(part)?.seq ?? 0, part })),
    ...modeling.cutFeatures.map((cut) => ({ seq: cut.seq, cut }))
  ].sort((a, b) => a.seq - b.seq);
  const partIds = new Set(parts.map((part) => part.id));
  return (
    <SetupPanel title="Model" onClose={onClose}>
      <div className="model-commands">
        <button type="button" className="btn btn-primary" onClick={() => modeling.startExtrude()} title="Sketch a profile on a plane or face, then extrude it">
          <Plus size={14} />
          Extruded Boss
        </button>
        <button
          type="button"
          className="btn"
          disabled={!parts.some((part) => part.meshData)}
          onClick={() => modeling.startCut()}
          title="Sketch a profile and cut it out of the casting or any other body"
        >
          <Scissors size={13} />
          Extruded Cut
        </button>
      </div>
      <div className="model-commands">
        <button type="button" className="btn" onClick={() => modeling.startExtrude(undefined, 'faceExtrude')} title="Click a flat face and extrude it as it is">
          <Copy size={13} />
          Extrude Face
        </button>
        <button type="button" className="btn" onClick={() => modeling.startPlane()} title="Reference plane offset from a plane or face, or midway between two">
          <Layers2 size={13} />
          Plane
        </button>
        <button type="button" className="btn" onClick={modeling.startMeasure} title="Measure distances, angles, diameters and areas">
          <Ruler size={13} />
          Measure
        </button>
        <button
          type="button"
          className="btn btn-icon"
          aria-label="Rebuild"
          disabled={(!modeled.length && !modeling.cutFeatures.length) || modeling.rebuilding}
          onClick={() => void modeling.rebuildAll()}
          title="Rebuild every modeled body and cut against the current assembly"
        >
          <RefreshCw size={13} className={modeling.rebuilding ? 'spin' : undefined} />
        </button>
      </div>
      <p className="field-hint">
        Sketch on a plane or a face and extrude risers, chills, cores and gating. Up To Next makes a body stop on the
        casting and take its shape; Extrude Face turns a face into a chill directly; Extruded Cut removes material
        from the bodies you choose, the casting included.
      </p>

      <Group title="Feature tree">
        <ul className="feature-tree" role="tree">
          <li className="feature-node feature-origin" role="treeitem">
            <Crosshair size={13} />
            <span>Origin</span>
          </li>
          {STANDARD_PLANES.map((plane) => (
            <li key={plane.id} role="treeitem">
              <button
                type="button"
                className="feature-node feature-plane"
                title={`Sketch on the ${plane.label} plane (${plane.hint})`}
                onClick={() => modeling.startExtrude({ kind: 'standard', id: plane.id, offset: 0 })}
              >
                <Square size={13} />
                <span>{plane.label} Plane</span>
                <span className="feature-node-hint">Sketch</span>
              </button>
            </li>
          ))}
          {modeling.referencePlanes.map((ref) => {
            const resolved = modeling.resolvedReferences.find((entry) => entry.id === ref.id);
            return (
              <li key={ref.id} role="treeitem" className="feature-item">
                <div className="feature-row">
                  <button
                    type="button"
                    className="feature-node feature-plane feature-ref-plane"
                    disabled={!resolved}
                    title={resolved ? `Sketch on ${ref.name}` : `${ref.name}: its references are no longer parallel`}
                    onClick={() => resolved && modeling.startExtrude(resolved.plane)}
                  >
                    <Layers2 size={13} />
                    <span className="feature-node-name">{ref.name}</span>
                    <span className="feature-node-hint">
                      {ref.source.kind === 'offset'
                        ? `${ref.source.flip ? '−' : ''}${ref.source.distance.toFixed(3)} in from ${planeLabel(ref.source.base)}`
                        : 'Mid plane'}
                    </span>
                  </button>
                  <div className="feature-actions">
                    <button type="button" className="btn btn-ghost btn-icon" title="Edit the plane" aria-label={`Edit ${ref.name}`} onClick={() => modeling.startPlane(ref.id)}>
                      <Pencil size={12} />
                    </button>
                    <button type="button" className="btn btn-ghost btn-icon" title="Delete the plane" aria-label={`Delete ${ref.name}`} onClick={() => void modeling.deletePlane(ref.id)}>
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              </li>
            );
          })}
          {imported.map((part) => (
            <li key={part.id} role="treeitem" aria-selected={part.id === selectedPartId}>
              <button type="button" className="feature-node" onClick={() => onSelectPart(part.id)}>
                <Box size={13} />
                <span className="feature-node-name">{part.name}</span>
                <TypeChip type={part.part_type} />
              </button>
            </li>
          ))}
          {history.map((entry) => {
            if (entry.cut) {
              const cut = entry.cut;
              const bodies = cut.partIds.filter((id) => partIds.has(id)).length;
              return (
                <li key={cut.id} role="treeitem" className="feature-item">
                  <div className="feature-row">
                    <button
                      type="button"
                      className="feature-node"
                      onClick={() => modeling.editCut(cut.id, 'feature')}
                      title="Edit the cut"
                    >
                      <Scissors size={13} />
                      <span className="feature-node-name">{cut.name}</span>
                      <span className="type-chip type-chip-cut">Cut</span>
                    </button>
                    <div className="feature-actions">
                      <button
                        type="button"
                        className="btn btn-ghost btn-icon"
                        title="Edit the cut"
                        aria-label={`Edit ${cut.name}`}
                        onClick={() => modeling.editCut(cut.id, 'feature')}
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost btn-icon"
                        title="Delete the cut and restore the material it removed"
                        aria-label={`Delete ${cut.name}`}
                        disabled={modeling.rebuilding}
                        onClick={() => void modeling.deleteCut(cut.id)}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="feature-node feature-child"
                    onClick={() => modeling.editCut(cut.id, 'sketch')}
                    title="Edit the sketch"
                  >
                    <PenLine size={12} />
                    <span>
                      Sketch · {planeLabel(cut.sketch.plane)} · {cutEndConditionLabel(cut.endCondition)}
                      {cut.endCondition === 'blind' || cut.endCondition === 'midplane' ? ` ${cut.depth.toFixed(3)} in` : ''} ·{' '}
                      {bodies} {bodies === 1 ? 'body' : 'bodies'}
                    </span>
                  </button>
                </li>
              );
            }
            const part = entry.part;
            const feature = featureOf(part)!;
            return (
              <li key={part.id} role="treeitem" aria-selected={part.id === selectedPartId} className="feature-item">
                <div className={part.id === selectedPartId ? 'feature-row is-selected' : 'feature-row'}>
                  <button
                    type="button"
                    className="feature-node"
                    onClick={() => onSelectPart(part.id)}
                    onDoubleClick={() => modeling.editFeature(part.id, 'feature')}
                    title="Double-click to edit the extrude"
                  >
                    <span className="feature-icon" style={{ background: partTypeColor(part.part_type) }} />
                    <span className="feature-node-name">{part.name}</span>
                    <TypeChip type={part.part_type} />
                  </button>
                  <div className="feature-actions">
                    <button
                      type="button"
                      className="btn btn-ghost btn-icon"
                      title="Edit the extrude"
                      aria-label={`Edit ${part.name}`}
                      onClick={() => modeling.editFeature(part.id, 'feature')}
                    >
                      <Pencil size={12} />
                    </button>
                    <button
                      type="button"
                      className="btn btn-ghost btn-icon"
                      title="Delete the body"
                      aria-label={`Delete ${part.name}`}
                      onClick={() => onRemovePart(part.id)}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
                <button
                  type="button"
                  className="feature-node feature-child"
                  onClick={() => modeling.editFeature(part.id, 'sketch')}
                  title="Edit the sketch"
                >
                  <PenLine size={12} />
                  <span>
                    Sketch · {planeLabel(feature.sketch.plane)} · {endConditionLabel(feature.endCondition)}
                    {feature.endCondition === 'blind' || feature.endCondition === 'midplane'
                      ? ` ${feature.depth.toFixed(3)} in`
                      : ''}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </Group>
    </SetupPanel>
  );
}

/**
 * Which regions of the sketch the feature uses. They are picked in the view,
 * as in Onshape; this says how many and offers the way back to the default.
 */
function RegionsGroup({ modeling }: { modeling: ModelingController }) {
  const regions = modeling.regions;
  const session = modeling.session;
  if (!session || !regions || (regions.length < 2 && !session.feature.regions)) return null;
  const seeds = session.feature.regions;
  const selected = regions.filter((region) => regionSelected(region, seeds)).length;
  return (
    <Group title="Regions">
      <p className="pm-message">
        {selected} of {regions.length} regions. Click regions in the view to add or remove them.
      </p>
      {seeds ? (
        <button type="button" className="btn btn-sm" onClick={modeling.resetRegions} title="Use every region the sketch fills">
          Use the whole sketch
        </button>
      ) : null}
    </Group>
  );
}

function TypeChip({ type }: { type: string }) {
  return (
    <span className="type-chip" style={{ borderColor: partTypeColor(type), color: partTypeColor(type) }}>
      {partTypeLabel(type)}
    </span>
  );
}

function endConditionLabel(id: EndCondition): string {
  return END_CONDITIONS.find((entry) => entry.id === id)?.label ?? id;
}

function cutEndConditionLabel(id: CutEndCondition): string {
  return CUT_END_CONDITIONS.find((entry) => entry.id === id)?.label ?? id;
}

// ---------------------------------------------------------------------------
// Property managers
// ---------------------------------------------------------------------------

function ManagerHeader({
  title,
  onOk,
  okDisabled,
  okLabel,
  onCancel,
  busy
}: {
  title: string;
  onOk?: () => void;
  okDisabled?: boolean;
  okLabel: string;
  onCancel: () => void;
  busy?: boolean;
}) {
  return (
    <div className="pm-head">
      <span className="pm-title">{title}</span>
      <div className="spacer" />
      {onOk ? (
        <button
          type="button"
          className="btn btn-icon pm-ok"
          title={okLabel}
          aria-label={okLabel}
          disabled={okDisabled || busy}
          onClick={onOk}
        >
          <Check size={15} />
        </button>
      ) : null}
      <button type="button" className="btn btn-icon pm-cancel" title="Cancel" aria-label="Cancel" onClick={onCancel}>
        <X size={15} />
      </button>
    </div>
  );
}

function ManagerShell({ children, footer }: { children: ReactNode; footer?: ReactNode }) {
  return (
    <section className="setup-panel property-manager" aria-label="Property manager">
      <div className="panel-body">{children}</div>
      {footer ? <footer className="panel-foot">{footer}</footer> : null}
    </section>
  );
}

function PlaneManager({ modeling }: ModelPanelProps) {
  const faceMode = modeling.session?.purpose === 'faceExtrude';
  const cutting = modeling.session?.feature.kind === 'cut';
  return (
    <ManagerShell>
      <ManagerHeader
        title={faceMode ? 'Extrude Face — select a face' : cutting ? 'Cut-Extrude — choose a sketch plane' : 'Extrude — choose a sketch plane'}
        okLabel="OK"
        onCancel={modeling.cancel}
      />
      <p className="pm-message">
        {faceMode
          ? 'Click a flat face of the model. Its outline, holes included, becomes the profile and the extrude starts from the face.'
          : 'Pick a datum or reference plane, or click any face of the model in the viewport to sketch on it.'}
      </p>
      {faceMode ? null : (
        <Group title="Planes">
          <div className="pm-button-grid">
            {STANDARD_PLANES.map((plane) => (
              <button
                key={plane.id}
                type="button"
                className="btn"
                title={plane.hint}
                onClick={() => modeling.choosePlane({ kind: 'standard', id: plane.id, offset: 0 })}
              >
                <Square size={13} />
                {plane.label}
              </button>
            ))}
            {modeling.resolvedReferences.map((ref) => (
              <button key={ref.id} type="button" className="btn" onClick={() => modeling.choosePlane(ref.plane)}>
                <Layers2 size={13} />
                {ref.name}
              </button>
            ))}
          </div>
        </Group>
      )}
    </ManagerShell>
  );
}

function SketchManager({ modeling }: ModelPanelProps) {
  const session = modeling.session!;
  const sketch = session.feature.sketch;
  const problem = sketchProblem(sketch.entities);
  return (
    <ManagerShell
      footer={
        <div className="pm-foot-actions">
          <button type="button" className="btn btn-primary btn-block" disabled={Boolean(problem)} onClick={modeling.exitSketch}>
            <Check size={14} />
            {session.feature.kind === 'cut' ? 'Exit Sketch → Cut' : 'Exit Sketch → Extrude'}
          </button>
        </div>
      }
    >
      <ManagerHeader
        title={`Sketch on ${planeLabel(sketch.plane)}`}
        okLabel="Exit sketch"
        okDisabled={Boolean(problem)}
        onOk={modeling.exitSketch}
        onCancel={modeling.cancel}
      />
      <Group title="Plane">
        <NumberField
          label="Offset along the normal"
          unit="in"
          value={sketch.plane.offset}
          onCommit={(value) => modeling.setPlaneOffset(value)}
          hint="Moves the sketch plane, and the sketch with it."
        />
        <div className="row">
          <button type="button" className="btn btn-sm" onClick={modeling.normalTo}>
            Normal To
          </button>
        </div>
      </Group>

      <Group title="Tools">
        <div className="pm-tool-grid" role="toolbar" aria-label="Sketch tools">
          {SKETCH_TOOLS.map(({ id, label, hint, Icon }) => (
            <button
              key={id}
              type="button"
              className="pm-tool"
              aria-pressed={session.tool === id}
              title={`${label}: ${hint}`}
              onClick={() => modeling.patchSession({ tool: id })}
            >
              <Icon size={15} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <ul className="pm-howto">
          <li>Click twice, or press and drag, to draw. Type a number while drawing to set the size.</li>
          <li>
            With Select, drag on empty space to box-select: left to right takes what is inside, right to left
            what the box touches. Ctrl- or ⌘-click adds to the selection.
          </li>
          <li>Type a number right after drawing, or click a dimension, to change it.</li>
          <li>Middle-drag or Shift-drag turns the view, right-drag pans, scroll zooms. Esc ends the tool.</li>
        </ul>
        {session.tool === 'polygon' ? (
          <NumberField
            label="Polygon sides"
            value={session.polygonSides}
            step={1}
            onCommit={(value) => modeling.patchSession({ polygonSides: Math.min(Math.max(Math.round(value), 3), 64) })}
          />
        ) : null}
        <label className="check">
          <input
            type="checkbox"
            checked={session.snap}
            onChange={(event) => modeling.patchSession({ snap: event.target.checked })}
          />
          <span className="check-body">
            Snap and infer
            <span className="check-note">
              Grid, ends, midpoints, centres, and horizontal/vertical lines. Hold Alt while clicking to place a point
              freely.
            </span>
          </span>
        </label>
      </Group>

      <Group title={`Entities (${sketch.entities.length})`}>
        {session.selectedEntityIds.length > 1 ? (
          <div className="pm-selection">
            <span>{session.selectedEntityIds.length} selected</span>
            <button type="button" className="btn btn-sm" onClick={() => modeling.patchSession({ selectedEntityIds: [] })}>
              Clear
            </button>
            <button
              type="button"
              className="btn btn-sm btn-danger"
              onClick={() => modeling.deleteEntities(session.selectedEntityIds)}
              title="Delete the selected entities (Del)"
            >
              <Trash2 size={12} />
              Delete
            </button>
          </div>
        ) : null}
        {sketch.entities.length === 0 ? (
          <p className="status-note">Draw a closed profile in the viewport. Click a dimension label to type its value.</p>
        ) : (
          <div className="entity-list">
            {sketch.entities.map((entity) => (
              <EntityEditor
                key={entity.id}
                entity={entity}
                selected={session.selectedEntityIds.includes(entity.id)}
                onSelect={() => modeling.patchSession({ selectedEntityIds: [entity.id] })}
                onChange={modeling.updateEntity}
                onDelete={() => modeling.deleteEntity(entity.id)}
              />
            ))}
          </div>
        )}
        {problem && sketch.entities.length ? <p className="status-note status-note-warning">{problem}</p> : null}
      </Group>
    </ManagerShell>
  );
}

function EntityEditor({
  entity,
  selected,
  onSelect,
  onChange,
  onDelete
}: {
  entity: SketchEntity;
  selected: boolean;
  onSelect: () => void;
  onChange: (entity: SketchEntity) => void;
  onDelete: () => void;
}) {
  const field = (label: string, key: string, unit = 'in', min?: number) => (
    <NumberField
      key={key}
      compact
      label={label}
      unit={unit}
      value={(entity as unknown as Record<string, number>)[key]}
      onCommit={(value) => {
        if (min !== undefined && !(value > min)) return;
        onChange({ ...entity, [key]: value } as SketchEntity);
      }}
    />
  );
  let fields: ReactNode;
  switch (entity.type) {
    case 'rectangle':
      fields = [field('Center X', 'cx'), field('Center Y', 'cy'), field('Width', 'w', 'in', 0), field('Height', 'h', 'in', 0), field('Angle', 'angle', '°')];
      break;
    case 'circle':
      fields = [field('Center X', 'cx'), field('Center Y', 'cy'), field('Diameter', 'd', 'in', 0)];
      break;
    case 'polygon':
      fields = [field('Center X', 'cx'), field('Center Y', 'cy'), field('Diameter', 'd', 'in', 0), field('Sides', 'sides', '', 2.5), field('Angle', 'angle', '°')];
      break;
    case 'slot':
      fields = [field('Center X', 'cx'), field('Center Y', 'cy'), field('Length', 'length', 'in', -1e-9), field('Width', 'width', 'in', 0), field('Angle', 'angle', '°')];
      break;
    case 'line': {
      const length = Math.hypot(entity.x2 - entity.x1, entity.y2 - entity.y1);
      const angle = (Math.atan2(entity.y2 - entity.y1, entity.x2 - entity.x1) * 180) / Math.PI;
      fields = [
        field('Start X', 'x1'),
        field('Start Y', 'y1'),
        field('End X', 'x2'),
        field('End Y', 'y2'),
        <NumberField
          key="length"
          compact
          label="Length"
          unit="in"
          value={length}
          onCommit={(value) => {
            if (!(value > 1e-4) || length < 1e-12) return;
            const k = value / length;
            onChange({ ...entity, x2: entity.x1 + (entity.x2 - entity.x1) * k, y2: entity.y1 + (entity.y2 - entity.y1) * k });
          }}
        />,
        <NumberField
          key="angle"
          compact
          label="Angle"
          unit="°"
          value={angle}
          onCommit={(value) => {
            const a = (value * Math.PI) / 180;
            onChange({ ...entity, x2: entity.x1 + Math.cos(a) * length, y2: entity.y1 + Math.sin(a) * length });
          }}
        />,
        entity.relation ? (
          <div key="relation" className="entity-relation" onClick={(event) => event.stopPropagation()}>
            <span className="type-chip">{entity.relation === 'horizontal' ? 'Horizontal' : 'Vertical'}</span>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              title="Delete the relation so the line can be angled"
              onClick={() => {
                const { relation: _removed, ...rest } = entity;
                void _removed;
                onChange(rest);
              }}
            >
              Delete relation
            </button>
          </div>
        ) : null
      ];
      break;
    }
    case 'arc': {
      const [start, end] = arcEnds(entity);
      fields = [
        field('Center X', 'cx'),
        field('Center Y', 'cy'),
        field('Radius', 'r', 'in', 0),
        field('Start angle', 'start', '°'),
        field('End angle', 'end', '°'),
        <div key="ends" className="field-hint entity-ends">
          ({start[0].toFixed(3)}, {start[1].toFixed(3)}) → ({end[0].toFixed(3)}, {end[1].toFixed(3)})
        </div>
      ];
      break;
    }
    case 'polyline':
      fields = entity.points.map((point, index) => (
        <div className="entity-point" key={index}>
          <span className="entity-point-label">P{index + 1}</span>
          <NumberField
            compact
            label="X"
            unit="in"
            value={point[0]}
            onCommit={(value) =>
              onChange({ ...entity, points: entity.points.map((p, i) => (i === index ? [value, p[1]] : p)) })
            }
          />
          <NumberField
            compact
            label="Y"
            unit="in"
            value={point[1]}
            onCommit={(value) =>
              onChange({ ...entity, points: entity.points.map((p, i) => (i === index ? [p[0], value] : p)) })
            }
          />
        </div>
      ));
      break;
  }
  return (
    <div className={selected ? 'entity-card is-selected' : 'entity-card'} onClick={onSelect}>
      <div className="entity-head">
        <span>{entityLabel(entity)}</span>
        <div className="spacer" />
        <label className="check entity-construction" title="For construction: drawn dashed, never part of a profile" onClick={(event) => event.stopPropagation()}>
          <input
            type="checkbox"
            checked={Boolean(entity.construction)}
            onChange={(event) => onChange({ ...entity, construction: event.target.checked })}
          />
          <span className="check-body">Construction</span>
        </label>
        <button
          type="button"
          className="btn btn-ghost btn-icon"
          aria-label={`Delete ${entityLabel(entity)}`}
          title="Delete (Del)"
          onClick={(event) => {
            event.stopPropagation();
            onDelete();
          }}
        >
          <Trash2 size={12} />
        </button>
      </div>
      <div className={entity.type === 'polyline' ? 'entity-points' : 'entity-fields'}>{fields}</div>
    </div>
  );
}

function ReferencePlaneManager({ modeling }: ModelPanelProps) {
  const plane = modeling.planeSession!;
  const source =
    plane.mode === 'offset'
      ? plane.refs[0]
        ? { kind: 'offset' as const, base: plane.refs[0], distance: plane.distance, flip: plane.flip }
        : null
      : plane.refs.length === 2
        ? { kind: 'mid' as const, a: plane.refs[0], b: plane.refs[1] }
        : null;
  const problem = referencePlaneProblem(source);
  return (
    <ManagerShell
      footer={
        <div className="pm-foot-actions">
          <button type="button" className="btn btn-primary btn-block" disabled={Boolean(problem)} onClick={() => void modeling.commitPlane()}>
            <Check size={14} />
            {plane.editingId ? 'Update plane' : plane.mode === 'offset' && plane.count > 1 ? `Create ${plane.count} planes` : 'Create plane'}
          </button>
        </div>
      }
    >
      <ManagerHeader
        title={plane.editingId ? `Edit ${plane.name}` : 'Plane'}
        okLabel="OK"
        okDisabled={Boolean(problem)}
        onOk={() => void modeling.commitPlane()}
        onCancel={modeling.cancel}
      />
      <div className="segmented segmented-block" role="radiogroup" aria-label="Plane type">
        <button type="button" aria-pressed={plane.mode === 'offset'} onClick={() => modeling.patchPlane({ mode: 'offset' })}>
          Offset distance
        </button>
        <button type="button" aria-pressed={plane.mode === 'mid'} onClick={() => modeling.patchPlane({ mode: 'mid' })}>
          Mid plane
        </button>
      </div>
      <Group title={plane.mode === 'offset' ? 'First reference' : 'References'}>
        <p className="pm-message">
          {plane.mode === 'offset'
            ? 'Click a datum plane, a reference plane or a face in the viewport.'
            : 'Click two parallel planes or faces. The new plane lies midway between them.'}
        </p>
        <ul className="pm-ref-list">
          {plane.refs.length === 0 ? <li className="pm-ref-empty">Nothing selected</li> : null}
          {plane.refs.map((ref, index) => (
            <li key={index} className="pm-ref">
              <Layers2 size={12} />
              <span>{planeLabel(ref)}</span>
              <button
                type="button"
                className="btn btn-ghost btn-icon"
                aria-label="Remove reference"
                onClick={() => modeling.patchPlane({ refs: plane.refs.filter((_, i) => i !== index) })}
              >
                <X size={12} />
              </button>
            </li>
          ))}
        </ul>
        <div className="pm-button-grid">
          {STANDARD_PLANES.map((datum) => (
            <button
              key={datum.id}
              type="button"
              className="btn btn-sm"
              onClick={() =>
                modeling.patchPlane({
                  refs:
                    plane.mode === 'offset'
                      ? [{ kind: 'standard', id: datum.id, offset: 0 }]
                      : [...plane.refs, { kind: 'standard' as const, id: datum.id, offset: 0 }].slice(-2)
                })
              }
            >
              {datum.label}
            </button>
          ))}
        </div>
      </Group>
      {plane.mode === 'offset' ? (
        <Group title="Offset">
          <NumberField label="Distance" unit="in" value={plane.distance} onCommit={(value) => modeling.patchPlane({ distance: value })} />
          <label className="check">
            <input type="checkbox" checked={plane.flip} onChange={(event) => modeling.patchPlane({ flip: event.target.checked })} />
            <span className="check-body">Flip offset</span>
          </label>
          {plane.editingId ? null : (
            <NumberField
              label="Number of planes"
              value={plane.count}
              step={1}
              onCommit={(value) => modeling.patchPlane({ count: Math.min(Math.max(Math.round(value), 1), 20) })}
              hint="Each plane is one more distance further out."
            />
          )}
        </Group>
      ) : null}
      <Group title="Name">
        <input className="input" value={plane.name} onChange={(event) => modeling.patchPlane({ name: event.target.value })} />
      </Group>
      {problem && plane.refs.length ? <p className="status-note status-note-warning">{problem}</p> : null}
      {modeling.planePreview.length ? (
        <p className="status-note status-note-info">
          {modeling.planePreview.map((preview) => `${preview.offset >= 0 ? '' : '−'}${Math.abs(preview.offset).toFixed(3)} in`).join(', ')}{' '}
          from the origin along the plane normal.
        </p>
      ) : null}
    </ManagerShell>
  );
}

function MeasureManager({ modeling }: ModelPanelProps) {
  const picks = modeling.measurePicks ?? [];
  const result = modeling.measureResult;
  return (
    <ManagerShell
      footer={
        <div className="pm-foot-actions">
          <button type="button" className="btn" disabled={!picks.length} onClick={modeling.clearMeasure}>
            Clear selections
          </button>
          <button type="button" className="btn btn-primary" onClick={modeling.cancel}>
            Done
          </button>
        </div>
      }
    >
      <ManagerHeader title="Measure" okLabel="Done" onOk={modeling.cancel} onCancel={modeling.cancel} />
      <p className="pm-message">
        Click vertices, edges or faces. One selection reports its size; two report the distance or angle between
        them. A third click starts over; Esc clears.
      </p>
      <Group title="Selections">
        <ul className="pm-ref-list">
          {picks.length === 0 ? <li className="pm-ref-empty">Nothing selected</li> : null}
          {picks.map((pick, index) => (
            <li key={index} className="pm-ref">
              <Ruler size={12} />
              <span>{pickLabel(pick)}</span>
            </li>
          ))}
        </ul>
      </Group>
      {result ? (
        <Group title={result.title}>
          <div className="property-sheet measure-sheet">
            {result.rows.map((row) => (
              <div className="property-row" key={row.label}>
                <span className="property-label">{row.label}</span>
                <span className="property-value">{row.value}</span>
              </div>
            ))}
          </div>
        </Group>
      ) : null}
    </ManagerShell>
  );
}

function ExtrudeManager({ modeling, parts }: ModelPanelProps) {
  const session = modeling.session!;
  const feature = session.feature as ExtrudeFeature;
  const body = session.body;
  const preview = modeling.preview;
  const ownId = session.editingPartId;
  const usesDepth = feature.endCondition === 'blind' || feature.endCondition === 'midplane';
  const bodyChoices = parts.filter((part) => part.id !== ownId && part.meshData);
  const okDisabled = preview.status !== 'ok' || !body.name.trim();

  return (
    <ManagerShell
      footer={
        <div className="pm-foot-actions">
          <button type="button" className="btn" onClick={modeling.backToSketch}>
            <PenLine size={13} />
            Edit Sketch
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={okDisabled || modeling.committing}
            onClick={() => void modeling.commit()}
          >
            <Check size={14} />
            {modeling.committing ? 'Building...' : session.editingPartId ? 'Update body' : 'Create body'}
          </button>
        </div>
      }
    >
      <ManagerHeader
        title={session.editingPartId ? `Edit ${body.name}` : 'Boss-Extrude'}
        okLabel="OK"
        okDisabled={okDisabled}
        busy={modeling.committing}
        onOk={() => void modeling.commit()}
        onCancel={modeling.cancel}
      />
      <p className="pm-message">From {planeLabel(feature.sketch.plane)}</p>

      <RegionsGroup modeling={modeling} />

      <Group title="Direction 1">
        <label className="field">
          <span className="field-label">End condition</span>
          <select
            className="select"
            value={feature.endCondition}
            onChange={(event) => modeling.patchFeature({ endCondition: event.target.value as EndCondition })}
          >
            {END_CONDITIONS.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.label}
              </option>
            ))}
          </select>
          <span className="field-hint">{END_CONDITIONS.find((entry) => entry.id === feature.endCondition)?.hint}</span>
        </label>
        <button
          type="button"
          className="btn btn-sm"
          aria-pressed={feature.reverse}
          onClick={() => modeling.patchFeature({ reverse: !feature.reverse })}
          title="Reverse the extrude direction"
        >
          <ArrowDownUp size={13} />
          Reverse direction
        </button>
        {usesDepth ? (
          <NumberField
            label={feature.endCondition === 'midplane' ? 'Total depth' : 'Depth'}
            unit="in"
            value={feature.depth}
            onCommit={(value) => value > 0 && modeling.patchFeature({ depth: value })}
            hint="Or drag the arrow in the view. Hold Alt for no snapping."
          />
        ) : null}
        {feature.endCondition === 'upToBody' ? (
          <label className="field">
            <span className="field-label">Body</span>
            <select
              className="select"
              value={feature.targetPartId ?? ''}
              onChange={(event) => modeling.patchFeature({ targetPartId: event.target.value || null })}
            >
              <option value="">Choose a body...</option>
              {bodyChoices.map((part) => (
                <option key={part.id} value={part.id}>
                  {part.name} ({partTypeLabel(part.part_type)})
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <div className="pm-inline">
          <NumberField
            compact
            label="Draft"
            unit="°"
            value={feature.draftDeg}
            onCommit={(value) => value >= 0 && value < 60 && modeling.patchFeature({ draftDeg: value })}
          />
          <label className="check">
            <input
              type="checkbox"
              checked={feature.draftOutward}
              disabled={feature.draftDeg <= 0}
              onChange={(event) => modeling.patchFeature({ draftOutward: event.target.checked })}
            />
            <span className="check-body">Draft outward</span>
          </label>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={feature.trimOverlap}
            onChange={(event) => modeling.patchFeature({ trimOverlap: event.target.checked })}
          />
          <span className="check-body">
            Trim where it overlaps other bodies
            <span className="check-note">
              Keeps the body out of the casting, so a chill does not take casting volume.
            </span>
          </span>
        </label>
      </Group>

      <Group title="Body">
        <label className="field">
          <span className="field-label">Name</span>
          <input className="input" value={body.name} onChange={(event) => modeling.patchBody({ name: event.target.value })} />
        </label>
        <div className="field">
          <span className="field-label">Type</span>
          <div className="pm-type-grid" role="radiogroup" aria-label="Body type">
            {BODY_TYPES.map((type) => {
              const def = PART_TYPES.find((entry) => entry.id === type)!;
              return (
                <button
                  key={type}
                  type="button"
                  role="radio"
                  aria-checked={body.partType === type}
                  className="pm-type"
                  title={def.hint}
                  onClick={() => modeling.patchBody({ partType: type })}
                >
                  <span className="pm-type-swatch" style={{ background: def.color }} />
                  {def.label}
                </button>
              );
            })}
          </div>
          <span className="field-hint">{PART_TYPES.find((entry) => entry.id === body.partType)?.hint}</span>
        </div>
        {body.partType === 'riser' ? (
          <label className="check">
            <input
              type="checkbox"
              checked={body.isOpenTop}
              onChange={(event) => modeling.patchBody({ isOpenTop: event.target.checked })}
            />
            <span className="check-body">
              Open top
              <span className="check-note">Riser is open to atmosphere and can vent during filling.</span>
            </span>
          </label>
        ) : null}
      </Group>

      <Group title="Preview">
        {preview.status === 'building' ? <p className="status-note">Building...</p> : null}
        {preview.status === 'error' ? <p className="status-note status-note-warning">{preview.error}</p> : null}
        {preview.status === 'ok' ? (
          <div className="property-sheet">
            <div className="property-row">
              <span className="property-label">Volume</span>
              <span className="property-value">{preview.body.volume.toFixed(3)} in³</span>
            </div>
            <div className="property-row">
              <span className="property-label">Size</span>
              <span className="property-value">
                {(preview.body.bbox.max[0] - preview.body.bbox.min[0]).toFixed(2)} ×{' '}
                {(preview.body.bbox.max[1] - preview.body.bbox.min[1]).toFixed(2)} ×{' '}
                {(preview.body.bbox.max[2] - preview.body.bbox.min[2]).toFixed(2)} in
              </span>
            </div>
            <div className="property-row">
              <span className="property-label">Triangles</span>
              <span className="property-value">{preview.body.triangles.toLocaleString()}</span>
            </div>
            {preview.body.warnings.map((warning) => (
              <p key={warning} className="status-note status-note-info">
                {warning}
              </p>
            ))}
          </div>
        ) : null}
      </Group>
    </ManagerShell>
  );
}

function CutManager({ modeling, parts }: ModelPanelProps) {
  const session = modeling.session!;
  const feature = session.feature as CutFeature;
  const preview = modeling.preview;
  const usesDepth = feature.endCondition === 'blind' || feature.endCondition === 'midplane';
  const candidates = cutCandidates(parts, feature);
  const scope = new Set(feature.partIds);
  const allSelected = candidates.length > 0 && candidates.every((part) => scope.has(part.id));
  const removed = preview.status === 'ok' ? (preview.removed ?? {}) : {};
  const okDisabled = preview.status !== 'ok' || !feature.name.trim();
  const editing = session.editingCutId !== null;
  const setScope = (ids: string[]) => modeling.patchFeature({ partIds: ids });

  return (
    <ManagerShell
      footer={
        <div className="pm-foot-actions">
          <button type="button" className="btn" onClick={modeling.backToSketch}>
            <PenLine size={13} />
            Edit Sketch
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={okDisabled || modeling.committing}
            onClick={() => void modeling.commit()}
          >
            <Check size={14} />
            {modeling.committing ? 'Cutting...' : editing ? 'Update cut' : 'Cut'}
          </button>
        </div>
      }
    >
      <ManagerHeader
        title={editing ? `Edit ${feature.name}` : 'Cut-Extrude'}
        okLabel="OK"
        okDisabled={okDisabled}
        busy={modeling.committing}
        onOk={() => void modeling.commit()}
        onCancel={modeling.cancel}
      />
      <p className="pm-message">From {planeLabel(feature.sketch.plane)}</p>

      <RegionsGroup modeling={modeling} />

      <Group title="Direction 1">
        <label className="field">
          <span className="field-label">End condition</span>
          <select
            className="select"
            value={feature.endCondition}
            onChange={(event) => modeling.patchFeature({ endCondition: event.target.value as CutEndCondition })}
          >
            {CUT_END_CONDITIONS.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.label}
              </option>
            ))}
          </select>
          <span className="field-hint">{CUT_END_CONDITIONS.find((entry) => entry.id === feature.endCondition)?.hint}</span>
        </label>
        {feature.endCondition === 'throughAllBoth' || feature.endCondition === 'midplane' ? null : (
          <button
            type="button"
            className="btn btn-sm"
            aria-pressed={feature.reverse}
            onClick={() => modeling.patchFeature({ reverse: !feature.reverse })}
            title="Reverse the cut direction"
          >
            <ArrowDownUp size={13} />
            Reverse direction
          </button>
        )}
        {usesDepth ? (
          <NumberField
            label={feature.endCondition === 'midplane' ? 'Total depth' : 'Depth'}
            unit="in"
            value={feature.depth}
            onCommit={(value) => value > 0 && modeling.patchFeature({ depth: value })}
            hint="Or drag the arrow in the view. Hold Alt for no snapping."
          />
        ) : null}
        <div className="pm-inline">
          <NumberField
            compact
            label="Draft"
            unit="°"
            value={feature.draftDeg}
            onCommit={(value) => value >= 0 && value < 60 && modeling.patchFeature({ draftDeg: value })}
          />
          <label className="check">
            <input
              type="checkbox"
              checked={feature.draftOutward}
              disabled={feature.draftDeg <= 0}
              onChange={(event) => modeling.patchFeature({ draftOutward: event.target.checked })}
            />
            <span className="check-body">Draft outward</span>
          </label>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={feature.flipSide}
            onChange={(event) => modeling.patchFeature({ flipSide: event.target.checked })}
          />
          <span className="check-body">
            Flip side to cut
            <span className="check-note">Removes the material outside the profile instead of inside it.</span>
          </span>
        </label>
      </Group>

      <Group title="Feature scope">
        {candidates.length === 0 ? (
          <p className="status-note">No body comes before this cut in the history.</p>
        ) : (
          <>
            <label className="check">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={(event) => setScope(event.target.checked ? candidates.map((part) => part.id) : [])}
              />
              <span className="check-body">All bodies</span>
            </label>
            <ul className="pm-scope-list">
              {candidates.map((part) => (
                <li key={part.id}>
                  <label className="check pm-scope">
                    <input
                      type="checkbox"
                      checked={scope.has(part.id)}
                      onChange={(event) =>
                        setScope(
                          event.target.checked
                            ? [...feature.partIds, part.id]
                            : feature.partIds.filter((id) => id !== part.id)
                        )
                      }
                    />
                    <span className="pm-scope-name">{part.name}</span>
                    <TypeChip type={part.part_type} />
                    {removed[part.id] ? <span className="pm-scope-removed">−{removed[part.id].toFixed(3)} in³</span> : null}
                  </label>
                </li>
              ))}
            </ul>
            <p className="field-hint">Only the checked bodies are cut. A body the cut misses is left as it is.</p>
          </>
        )}
      </Group>

      <Group title="Name">
        <input className="input" value={feature.name} onChange={(event) => modeling.patchFeature({ name: event.target.value })} />
      </Group>

      <Group title="Preview">
        {preview.status === 'building' ? <p className="status-note">Building...</p> : null}
        {preview.status === 'error' ? <p className="status-note status-note-warning">{preview.error}</p> : null}
        {preview.status === 'ok' ? (
          <div className="property-sheet">
            <div className="property-row">
              <span className="property-label">Removed</span>
              <span className="property-value">{preview.body.volume.toFixed(3)} in³</span>
            </div>
            <div className="property-row">
              <span className="property-label">Bodies cut</span>
              <span className="property-value">{Object.keys(removed).length}</span>
            </div>
            {preview.body.warnings.map((warning) => (
              <p key={warning} className="status-note status-note-info">
                {warning}
              </p>
            ))}
          </div>
        ) : null}
      </Group>
    </ManagerShell>
  );
}

// ---------------------------------------------------------------------------
// A number input that commits on Enter or blur, like a CAD modify box
// ---------------------------------------------------------------------------

function NumberField({
  label,
  unit,
  value,
  onCommit,
  hint,
  step,
  compact
}: {
  label: string;
  unit?: string;
  value: number;
  onCommit: (value: number) => void;
  hint?: string;
  step?: number;
  compact?: boolean;
}) {
  const [text, setText] = useState(formatValue(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setText(formatValue(value));
  }, [value, focused]);
  const commit = () => {
    const parsed = Number(text.trim());
    if (Number.isFinite(parsed) && parsed !== value) onCommit(parsed);
    else setText(formatValue(value));
  };
  return (
    <label className={compact ? 'field field-compact' : 'field'}>
      <span className="field-label">
        {label}
        {unit ? <span className="field-unit">{unit}</span> : null}
      </span>
      <input
        className="input input-num"
        inputMode="decimal"
        value={text}
        step={step}
        onFocus={() => setFocused(true)}
        onBlur={() => {
          setFocused(false);
          commit();
        }}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            commit();
            (event.target as HTMLInputElement).blur();
          } else if (event.key === 'Escape') {
            setText(formatValue(value));
            (event.target as HTMLInputElement).blur();
          }
        }}
      />
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return '';
  return Number(value.toFixed(4)).toString();
}
