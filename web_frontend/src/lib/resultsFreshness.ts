/**
 * Whether the loaded results still describe the parts on screen.
 *
 * A finished run records the parts it simulated in its summary
 * (`simulated_parts`, see `simulated_parts_snapshot` in web/payloads.py). Any
 * difference in a field the run depends on -- a part deleted, added, moved,
 * hidden, retyped, or a modeled body rebuilt into a new file -- means the field
 * on screen belongs to a model that no longer exists: a deleted riser is still
 * in it, solidifying where nothing is drawn.
 */
import type { ProjectPart } from './api';

export type SimulatedPart = Partial<PartFields>;

type PartFields = Pick<
  ProjectPart,
  | 'id'
  | 'filepath'
  | 'part_type'
  | 'visible'
  | 'is_open_top'
  | 'riser_sleeve'
  | 'position_x'
  | 'position_y'
  | 'position_z'
  | 'rotation_x'
  | 'rotation_y'
  | 'rotation_z'
  | 'scale'
>;

/** Positions and angles equal to a millionth are equal; the server stores floats. */
const round = (value: unknown, fallback: number) => {
  const number = Number(value ?? fallback);
  return Number.isFinite(number) ? Math.round(number * 1e6) / 1e6 : fallback;
};

/** A key that is equal exactly when two part lists would simulate the same. */
export function simulatedPartsKey(parts: SimulatedPart[]): string {
  return parts
    .map((part) =>
      [
        String(part.id ?? ''),
        String(part.filepath ?? ''),
        String(part.part_type ?? ''),
        part.visible === false ? 0 : 1,
        part.is_open_top ? 1 : 0,
        // Runs recorded before the sleeve choice are insulating.
        part.riser_sleeve === 'none' ? 'none' : 'insulating',
        round(part.position_x, 0),
        round(part.position_y, 0),
        round(part.position_z, 0),
        round(part.rotation_x, 0),
        round(part.rotation_y, 0),
        round(part.rotation_z, 0),
        round(part.scale, 1)
      ].join('|')
    )
    .sort()
    .join('\n');
}

/** The parts a run simulated, or null for a run that predates the record. */
export function simulatedPartsFromSummary(summary: Record<string, unknown> | null | undefined): SimulatedPart[] | null {
  const recorded = summary?.simulated_parts;
  return Array.isArray(recorded) ? (recorded as SimulatedPart[]) : null;
}

/** What changed, in a few words, for the banner. */
export function describePartChanges(simulated: SimulatedPart[], current: SimulatedPart[]): string {
  const before = new Map(simulated.map((part) => [String(part.id), part]));
  const now = new Map(current.map((part) => [String(part.id), part]));
  let removed = 0;
  let added = 0;
  let changed = 0;
  for (const id of before.keys()) if (!now.has(id)) removed += 1;
  for (const [id, part] of now) {
    const old = before.get(id);
    if (!old) added += 1;
    else if (simulatedPartsKey([old]) !== simulatedPartsKey([part])) changed += 1;
  }
  const phrase = (count: number, verb: string) => (count ? `${count} ${count === 1 ? 'part' : 'parts'} ${verb}` : '');
  return [phrase(removed, 'deleted'), phrase(added, 'added'), phrase(changed, 'changed')].filter(Boolean).join(', ');
}
