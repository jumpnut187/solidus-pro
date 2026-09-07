/**
 * Reading and writing the nested `solidification_params` structure.
 *
 * The Filling tab is the awkward one: its filter fields are shown alongside the
 * rest but persist one level deeper, at `filling.filter.*`. Every access goes
 * through here so that nesting rule lives in one place rather than being
 * re-derived at each call site.
 */
import type { SolidificationParams } from './api';
import { FILTER_KEYS, PARAM_DEFS, type ParamDef, type ParamGroup } from './solidificationFields';

function isFilterField(def: ParamDef): boolean {
  return def.group === 'filling' && FILTER_KEYS.has(def.key);
}

export function readParam(params: SolidificationParams, def: ParamDef): number {
  const group = params[def.group];
  if (!group) return def.default;
  if (isFilterField(def)) {
    const filter = group.filter;
    if (typeof filter !== 'object' || filter === null) return def.default;
    const value = (filter as Record<string, number>)[def.key];
    return typeof value === 'number' && Number.isFinite(value) ? value : def.default;
  }
  const value = group[def.key];
  return typeof value === 'number' && Number.isFinite(value) ? value : def.default;
}

export function writeParam(
  params: SolidificationParams,
  def: ParamDef,
  value: number
): SolidificationParams {
  const group = { ...(params[def.group] ?? {}) };
  if (isFilterField(def)) {
    const filter = { ...((group.filter as Record<string, number> | undefined) ?? {}) };
    filter[def.key] = value;
    group.filter = filter;
  } else {
    group[def.key] = value;
  }
  return { ...params, [def.group]: group };
}

/** Restore one tab's visible fields to their defaults, leaving other tabs alone. */
export function resetGroup(params: SolidificationParams, group: ParamGroup): SolidificationParams {
  let next = params;
  for (const def of PARAM_DEFS) {
    if (def.group !== group) continue;
    next = writeParam(next, def, def.default);
  }
  return next;
}

/** How many fields in a tab differ from their default — shown as a badge. */
export function changedCount(params: SolidificationParams, group: ParamGroup): number {
  let count = 0;
  for (const def of PARAM_DEFS) {
    if (def.group !== group) continue;
    if (readParam(params, def) !== def.default) count += 1;
  }
  return count;
}
