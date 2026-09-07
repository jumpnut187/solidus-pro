/**
 * The ten part types the simulation engine understands, mirroring
 * `core/part_manager.py` (`PartType`, `PART_TYPE_COLORS`, `PART_TYPE_KEYWORDS`,
 * `infer_part_type_from_name`).
 *
 * The web client previously exposed only five, which meant a gating system
 * uploaded as sprue/runner/gate silently became five castings and every one of
 * them got scored for shrinkage.
 */
import type { PartType } from './api';

export type PartTypeDef = {
  id: PartType;
  label: string;
  /** `PART_TYPE_COLORS` converted from 0..1 floats to hex. */
  color: string;
  /** One-line description for the type picker's tooltip. */
  hint: string;
};

export const PART_TYPES: PartTypeDef[] = [
  { id: 'cast', label: 'Cast', color: '#b3bfcc', hint: 'The casting itself. Scored for shrinkage and porosity.' },
  { id: 'riser', label: 'Riser', color: '#e69933', hint: 'Feeder. Slows cooling and supplies feed metal.' },
  { id: 'riser_post', label: 'Riser Post', color: '#cc804d', hint: 'Neck connecting a riser to the casting.' },
  { id: 'chill', label: 'Chill', color: '#4d4d80', hint: 'External chill. Accelerates local solidification.' },
  { id: 'core', label: 'Core', color: '#998066', hint: 'Sand core forming an internal cavity. Not metal.' },
  { id: 'pour_cup', label: 'Pour Cup', color: '#73ccbf', hint: 'Pouring basin at the top of the sprue.' },
  { id: 'sprue', label: 'Sprue', color: '#4dadb3', hint: 'Vertical downsprue.' },
  { id: 'runner', label: 'Runner', color: '#338ca6', hint: 'Horizontal channel feeding the gates.' },
  { id: 'filter', label: 'Filter', color: '#d9d959', hint: 'Ceramic foam filter. Metal flows through it.' },
  { id: 'gate', label: 'Gate', color: '#267399', hint: 'Ingate where metal enters the cavity.' }
];

const BY_ID = new Map(PART_TYPES.map((type) => [type.id, type]));

export function partTypeDef(id: string): PartTypeDef {
  return BY_ID.get(id as PartType) ?? PART_TYPES[0];
}

export function partTypeLabel(id: string): string {
  return partTypeDef(id).label;
}

export function partTypeColor(id: string): string {
  return partTypeDef(id).color;
}

/**
 * Only a riser can be marked open-top, matching the desktop parts dialog:
 * the flag feeds the filling stage's riser venting, which is defined for
 * risers alone.
 */
export const OPEN_TOP_TYPES: readonly PartType[] = ['riser'];

export function supportsOpenTop(partType: string): boolean {
  return OPEN_TOP_TYPES.includes(partType as PartType);
}

/**
 * Keyword table from `PART_TYPE_KEYWORDS`, in the same precedence order
 * `infer_part_type_from_name` walks: core, chill, the gating group (whose names
 * would otherwise be swallowed by the generic 'post' / 'pad' / 'bridge'
 * keywords), riser post, riser, then cast.
 */
const KEYWORD_ORDER: { type: PartType; keywords: string[] }[] = [
  { type: 'core', keywords: ['core', 'sand_core', 'sand core'] },
  { type: 'chill', keywords: ['chill'] },
  { type: 'filter', keywords: ['filter', 'stelex', 'sedex'] },
  { type: 'pour_cup', keywords: ['pour_cup', 'pour cup', 'pouring cup', 'pouring basin', 'basin'] },
  { type: 'sprue', keywords: ['downsprue', 'down_sprue', 'down sprue', 'sprue'] },
  { type: 'runner', keywords: ['runner'] },
  { type: 'gate', keywords: ['ingate', 'in_gate', 'in gate', 'gate'] },
  { type: 'riser_post', keywords: ['riser_post', 'riser post', 'post', 'neck'] },
  { type: 'riser', keywords: ['feeder', 'riser', 'bridge', 'pad'] },
  { type: 'cast', keywords: ['casting', 'part'] }
];

/**
 * Infer a part type from a file or body name, matching the server's inference
 * exactly so the type the user sees in the table is the type the run uses.
 * Defaults to `cast` when nothing matches.
 */
export function inferPartType(name: string): PartType {
  const base = name
    .split(/[\\/]/)
    .pop()!
    .replace(/\.[^.]*$/, '')
    .toLowerCase();
  for (const entry of KEYWORD_ORDER) {
    if (entry.keywords.some((keyword) => base.includes(keyword))) return entry.type;
  }
  return 'cast';
}
