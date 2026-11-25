import type { PartType } from './api';

export type PartTypeDef = {
  id: PartType;
  label: string;
  color: string;
  hint: string;
};

export const PART_TYPES: PartTypeDef[] = [
  { id: 'cast', label: 'Cast', color: '#b3bfcc', hint: 'The casting itself. Scored for shrinkage and porosity.' },
  { id: 'riser', label: 'Riser', color: '#e69933', hint: 'Feeder. Slows cooling and supplies feed metal.' },
  { id: 'riser_post', label: 'Riser Post', color: '#cc804d', hint: 'Neck connecting a riser to the casting.' },
  { id: 'chill', label: 'Chill', color: '#4d4d80', hint: 'External chill. Accelerates local solidification.' },
  { id: 'core', label: 'Core', color: '#4dc766', hint: 'Sand core forming an internal cavity. Not metal.' },
  { id: 'pour_cup', label: 'Pour Cup', color: '#73ccbf', hint: 'Pouring basin at the top of the sprue.' },
  { id: 'sprue', label: 'Sprue', color: '#4dadb3', hint: 'Vertical downsprue.' },
  { id: 'runner', label: 'Runner', color: '#338ca6', hint: 'Horizontal channel feeding the gates.' },
  { id: 'filter', label: 'Filter', color: '#d9d959', hint: 'Ceramic foam filter. Metal flows through it.' },
  { id: 'gate', label: 'Gate', color: '#267399', hint: 'Ingate where metal enters the cavity.' },
  { id: 'mold', label: 'Mold', color: '#9e66db', hint: 'Sand mold (flask, cope, drag) around the assembly. Not metal.' }
];

export const SAND_TYPES: readonly PartType[] = ['core', 'mold'];

export function isSandType(partType: string): boolean {
  return SAND_TYPES.includes(partType as PartType);
}

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

export const OPEN_TOP_TYPES: readonly PartType[] = ['riser'];

export function supportsOpenTop(partType: string): boolean {
  return OPEN_TOP_TYPES.includes(partType as PartType);
}

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
  { type: 'cast', keywords: ['casting', 'part'] },
  { type: 'mold', keywords: ['mold', 'mould', 'flask', 'cope', 'drag'] }
];

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
