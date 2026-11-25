import Module from 'manifold-3d';

import {
  applyCuts,
  bodyFromMesh,
  buildExtrude,
  FeatureBuildError,
  partWorldMatrix,
  previewCut,
  sketchRegions,
  type CutTarget,
  type KernelBody
} from '../src/lib/modeling/kernel';
import { measure } from '../src/lib/modeling/measure';
import {
  arcThrough,
  defaultCut,
  defaultExtrude,
  moveEnd,
  parallelOffset,
  pointInContours,
  replaceEntity,
  referencePlanePreview,
  resolveReferencePlane,
  simplifyLoop,
  sketchProblem,
  sketchProfiles,
  withDimension,
  type CutFeature,
  type ExtrudeFeature,
  type SketchEntity,
  type SketchPlane
} from '../src/lib/modeling/types';

let failures = 0;
function check(name: string, condition: boolean, detail = ''): void {
  if (condition) console.log(`  ok   ${name}`);
  else {
    failures += 1;
    console.log(`  FAIL ${name} ${detail}`);
  }
}
function near(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

const wasm = await Module();
wasm.setup();
const { Manifold } = wasm;

const dome = Manifold.sphere(5, 96).translate([0, 0, 4]).trimByPlane([0, 0, 1], 4);
const castSolid = Manifold.cylinder(4, 5, 5, 96).add(dome);
const castMesh = castSolid.getMesh();
const soup = new Float32Array(castMesh.triVerts.length * 3);
const soupIdx = new Uint32Array(castMesh.triVerts.length);
for (let i = 0; i < castMesh.triVerts.length; i += 1) {
  for (let k = 0; k < 3; k += 1) soup[3 * i + k] = castMesh.vertProperties[castMesh.triVerts[i] * castMesh.numProp + k];
  soupIdx[i] = i;
}
const identity = partWorldMatrix([0, 0, 0], [0, 0, 0], 1);
const cast: KernelBody = bodyFromMesh(wasm, 'cast', 'Casting', soup, soupIdx, identity);
check('casting welds into a solid', cast.manifold !== null, cast.error ?? '');

function feature(plane: SketchPlane, entities: SketchEntity[], patch: Partial<ExtrudeFeature>): ExtrudeFeature {
  return { ...defaultExtrude(1, plane), ...patch, sketch: { plane, entities } };
}
const top = (offset: number): SketchPlane => ({ kind: 'standard', id: 'top', offset });
const circle = (cx: number, cy: number, d: number): SketchEntity => ({ id: 'c', type: 'circle', cx, cy, d });

console.log('blind');
{
  const out = buildExtrude(wasm, feature(top(12), [circle(20, 0, 2)], { depth: 3 }), [cast]);
  check('volume = pi r^2 h', near(out.volume, Math.PI * 3, 0.02), String(out.volume));
  check('starts at plane', near(out.bbox.min[2], 12, 1e-4) && near(out.bbox.max[2], 15, 1e-4));
}
{
  const out = buildExtrude(wasm, feature(top(12), [circle(20, 0, 2)], { depth: 3, reverse: true }), [cast]);
  check('reverse goes down', near(out.bbox.min[2], 9, 1e-4) && near(out.bbox.max[2], 12, 1e-4));
  check('reverse volume positive', near(out.volume, Math.PI * 3, 0.02), String(out.volume));
}

console.log('up to next (riser onto the dome)');
{
  const out = buildExtrude(wasm, feature(top(12), [circle(2, 0, 3)], { endCondition: 'upToNext', reverse: true }), [cast]);
  check('top stays on the sketch plane', near(out.bbox.max[2], 12, 1e-4));
  check('bottom conforms to the dome', near(out.bbox.min[2], 4 + Math.sqrt(25 - 3.5 * 3.5), 0.05), String(out.bbox.min[2]));
  const lowest = 4 + Math.sqrt(25 - 3.5 * 3.5);
  const highest = 4 + Math.sqrt(25 - 0.5 * 0.5);
  check('volume between the bounds', out.volume > Math.PI * 2.25 * (12 - highest) && out.volume < Math.PI * 2.25 * (12 - lowest));
  const riser = new wasm.Manifold(new wasm.Mesh({ numProp: 3, vertProperties: out.positions, triVerts: out.indices }));
  const overlap = riser.intersect(cast.manifold!).volume();
  check('no overlap with the casting', overlap < 1e-3, String(overlap));
  riser.delete();
}
{
  const below: SketchPlane = { kind: 'standard', id: 'top', offset: -2 };
  const out = buildExtrude(wasm, feature(below, [{ id: 'r', type: 'rectangle', cx: 0, cy: 0, w: 4, h: 2, angle: 0 }], { endCondition: 'upToNext' }), [cast]);
  check('chill stops at the bottom face', near(out.bbox.max[2], 0, 1e-4) && near(out.volume, 16, 1e-3), `${out.bbox.max[2]} ${out.volume}`);
}
for (const [label, entities] of [
  ['profile hanging off the body', [circle(5.5, 0, 3)]],
  ['profile wholly off the body', [circle(20, 0, 3)]]
] as const) {
  try {
    buildExtrude(wasm, feature(top(12), [...entities], { endCondition: 'upToNext', reverse: true }), [cast]);
    check(`refuses: ${label}`, false);
  } catch (error) {
    check(`refuses: ${label}`, error instanceof FeatureBuildError, String(error));
  }
}
try {
  buildExtrude(wasm, feature(top(2), [circle(0, 0, 2)], { endCondition: 'upToNext' }), [cast]);
  check('refuses: plane inside the body', false);
} catch (error) {
  check('refuses: plane inside the body', error instanceof FeatureBuildError && /inside a body/.test((error as Error).message), String(error));
}

console.log('up to next with draft');
{
  const out = buildExtrude(wasm, feature(top(12), [circle(0, 0, 4)], { endCondition: 'upToNext', reverse: true, draftDeg: 5, draftOutward: true }), [cast]);
  const run = 12 - (4 + Math.sqrt(25 - 4));
  check('outward draft widens toward the casting', out.bbox.max[0] > 2.1 && out.bbox.max[0] < 2 + run * Math.tan((5 * Math.PI) / 180) + 0.05, String(out.bbox.max[0]));
}

console.log('watertight once stored (bodies across a seam between two bodies)');
{
  const weldProblem = (positions: Float32Array, indices: Uint32Array): string | null => {
    const ids = new Map<string, number>();
    const remap = new Uint32Array(positions.length / 3);
    for (let i = 0; i < remap.length; i += 1) {
      const key = [0, 1, 2].map((k) => Math.round(positions[3 * i + k] * 1e8)).join();
      remap[i] = ids.get(key) ?? i;
      if (!ids.has(key)) ids.set(key, i);
    }
    const uses = new Map<string, number>();
    for (let t = 0; t < indices.length; t += 3) {
      const [a, b, c] = [remap[indices[t]], remap[indices[t + 1]], remap[indices[t + 2]]];
      if (a === b || b === c || a === c) return 'collapsed triangle';
      for (const [p, q] of [[a, b], [b, c], [c, a]]) {
        const key = p < q ? `${p}_${q}` : `${q}_${p}`;
        uses.set(key, (uses.get(key) ?? 0) + 1);
      }
    }
    const bad = [...uses.values()].filter((n) => n !== 2).length;
    return bad ? `${bad} edges not on exactly two triangles` : null;
  };
  const stored = (name: string, out: { positions: Float32Array; indices: Uint32Array }) =>
    bodyFromMesh(wasm, name, name, out.positions, out.indices, identity);

  const legacyCut = wasm.CrossSection.circle(1.25, 64).translate([-0.75, 0.25]).extrude(10).translate([0, 0, 1]);
  const legacyPieces = legacyCut.subtract(cast.manifold!).decompose();
  const legacyTop = legacyPieces.find((piece) => piece.boundingBox().max[2] > 10.9)!;
  const legacyMesh = legacyTop.getMesh();
  const legacy = bodyFromMesh(wasm, 'legacy', 'Legacy riser', legacyMesh.vertProperties, legacyMesh.triVerts, identity);
  for (const piece of legacyPieces) piece.delete();
  legacyCut.delete();
  check('legacy riser loads', legacy.manifold !== null, legacy.error ?? '');

  const a = buildExtrude(wasm, feature(top(11), [circle(-0.75, 0.25, 2.5)], { endCondition: 'upToNext', reverse: true }), [cast]);
  const bodyA = stored('a', a);
  const b = buildExtrude(wasm, feature(top(11), [circle(1, 0.25, 2.5)], { endCondition: 'upToNext', reverse: true }), [cast, bodyA]);
  const bodyB = stored('b', b);
  check('riser beside a stored riser welds closed', weldProblem(b.positions, b.indices) === null, String(weldProblem(b.positions, b.indices)));
  check('a stored riser overlaps the casting by a hair, not a seam', (() => {
    const riser = new wasm.Manifold(new wasm.Mesh({ numProp: 3, vertProperties: a.positions, triVerts: a.indices }));
    const overlap = riser.intersect(cast.manifold!).volume();
    riser.delete();
    return overlap > 0 && overlap < 1e-3;
  })());

  const scenes: Array<[string, KernelBody[], number]> = [
    ['over the legacy riser seam', [cast, legacy], 12],
    ['over the riser seam', [cast, bodyA], 12],
    ['over two touching risers', [cast, bodyA, bodyB], 12]
  ];
  for (const [label, targets, plane] of scenes) {
    let built = 0;
    const failures: string[] = [];
    for (const cx of [-1.5, -0.25, 0.5, 1.25]) {
      for (const cy of [-0.75, 0.25, 1]) {
        for (const shape of ['circle', 'square'] as const) {
          const entity: SketchEntity =
            shape === 'circle' ? circle(cx, cy, 2.25) : { id: 'r', type: 'rectangle', cx, cy, w: 2, h: 2, angle: 0 };
          for (const trimOverlap of [true, false]) {
            try {
              const out = buildExtrude(wasm, feature(top(plane), [entity], { endCondition: 'upToNext', reverse: true, trimOverlap }), targets);
              built += 1;
              const problem = weldProblem(out.positions, out.indices);
              if (problem) failures.push(`${shape} ${cx},${cy}: ${problem}`);
              const back = stored('x', out);
              if (!back.manifold) failures.push(`${shape} ${cx},${cy}: does not reload (${back.error})`);
              back.manifold?.delete();
            } catch (error) {
              failures.push(`${shape} ${cx},${cy}: ${String(error)}`);
            }
          }
        }
      }
    }
    check(`up to next ${label}: ${built} bodies weld closed`, built === 48 && failures.length === 0, failures.slice(0, 3).join('; '));
  }
  for (const body of [legacy, bodyA, bodyB]) body.manifold?.delete();
}

console.log('trim overlap (blind chill pushed into the casting)');
{
  const side: SketchPlane = { kind: 'standard', id: 'right', offset: 6 };
  const rect: SketchEntity = { id: 'r', type: 'rectangle', cx: 0, cy: 2, w: 3, h: 2, angle: 0 };
  const trimmed = buildExtrude(wasm, feature(side, [rect], { depth: 2, reverse: true, trimOverlap: true }), [cast]);
  const raw = buildExtrude(wasm, feature(side, [rect], { depth: 2, reverse: true, trimOverlap: false }), [cast]);
  check('untrimmed is the full block', near(raw.volume, 12, 1e-3), String(raw.volume));
  check('trimmed loses the overlap', trimmed.volume < raw.volume - 0.5 && trimmed.warnings.length > 0, String(trimmed.volume));
}

console.log('mid plane, through all, polygon, slot, polyline, hole');
{
  const out = buildExtrude(wasm, feature(top(20), [circle(0, 0, 2)], { endCondition: 'midplane', depth: 4 }), [cast]);
  check('mid plane is centred', near(out.bbox.min[2], 18, 1e-4) && near(out.bbox.max[2], 22, 1e-4));
}
{
  const out = buildExtrude(wasm, feature(top(12), [circle(20, 0, 2)], { endCondition: 'throughAll', reverse: true, trimOverlap: false }), [cast]);
  check('through all passes the casting', out.bbox.min[2] < 0, String(out.bbox.min[2]));
}
{
  const hex: SketchEntity = { id: 'p', type: 'polygon', cx: 0, cy: 0, d: 2, sides: 6, angle: 0 };
  const out = buildExtrude(wasm, feature(top(20), [hex], { depth: 1 }), [cast]);
  check('hexagon area', near(out.volume, (3 * Math.sqrt(3) / 2) * 1, 1e-3), String(out.volume));
}
{
  const slot: SketchEntity = { id: 's', type: 'slot', cx: 0, cy: 0, length: 4, width: 2, angle: 30 };
  const out = buildExtrude(wasm, feature(top(20), [slot], { depth: 1 }), [cast]);
  check('slot area', near(out.volume, 8 + Math.PI, 0.02), String(out.volume));
}
{
  const tri: SketchEntity = { id: 'l', type: 'polyline', points: [[0, 0], [0, 3], [4, 0]] };
  const out = buildExtrude(wasm, feature(top(20), [tri], { depth: 2 }), [cast]);
  check('clockwise polyline still builds', near(out.volume, 12, 1e-4), String(out.volume));
}
{
  const ring = [circle(0, 0, 4), { ...circle(0, 0, 2), id: 'c2' }];
  const out = buildExtrude(wasm, feature(top(20), ring, { depth: 1 }), [cast]);
  check('nested circle is a hole', near(out.volume, Math.PI * 3, 0.02), String(out.volume));
}
{
  const front: SketchPlane = { kind: 'standard', id: 'front', offset: 10 };
  const out = buildExtrude(wasm, feature(front, [circle(0, 8, 2)], { depth: 2 }), [cast]);
  check('front plane sits at y = -10 and grows to -12', near(out.bbox.max[1], -10, 1e-4) && near(out.bbox.min[1], -12, 1e-4), JSON.stringify(out.bbox));
  check('front plane u is +X, v is +Z', near((out.bbox.min[2] + out.bbox.max[2]) / 2, 8, 1e-3));
}

console.log('transforms');
{
  const moved = bodyFromMesh(wasm, 'm', 'Moved', soup, soupIdx, partWorldMatrix([0, 0, 10], [90, 0, 0], 1));
  check('moved body box', moved.bbox !== null && near(moved.bbox.min[2], 5, 0.01) && near(moved.bbox.max[2], 15, 0.01), JSON.stringify(moved.bbox));
  moved.manifold?.delete();
}
{
  const open = bodyFromMesh(wasm, 'o', 'Open', new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), new Uint32Array([0, 1, 2]), identity);
  check('an open mesh is reported, not thrown', open.manifold === null && open.error !== null);
  try {
    buildExtrude(wasm, feature(top(5), [circle(0.3, 0.3, 0.2)], { endCondition: 'upToNext', reverse: true }), [open]);
    check('an open body in the way fails the feature', false);
  } catch (error) {
    check('an open body in the way fails the feature', /closed solid/.test(String(error)), String(error));
  }
}

console.log('lines, arcs and profiles');
{
  const line = (id: string, x1: number, y1: number, x2: number, y2: number): SketchEntity => ({ id, type: 'line', x1, y1, x2, y2 });
  const square = [line('a', 0, 0, 4, 0), line('b', 4, 0, 4, 3), line('c', 4, 3, 0, 3), line('d', 0, 3, 0, 0)];
  const profiles = sketchProfiles(square);
  check('four lines close one loop', profiles.loops.length === 1 && profiles.openEnds.length === 0, JSON.stringify(profiles));
  const out = buildExtrude(wasm, feature(top(20), square, { depth: 2 }), [cast]);
  check('four-line rectangle extrudes', near(out.volume, 24, 1e-4), String(out.volume));

  const split = [...square, line('e', 2, 0, 2, 3)];
  const splitProfiles = sketchProfiles(split);
  check('a dividing line makes two regions', splitProfiles.loops.length === 2, String(splitProfiles.loops.length));
  const splitOut = buildExtrude(wasm, feature(top(20), split, { depth: 2 }), [cast]);
  check('two regions extrude to the whole', near(splitOut.volume, 24, 1e-3), String(splitOut.volume));

  const open = square.slice(0, 3);
  check('an open chain is refused', /open contour/.test(sketchProblem(open) ?? ''), String(sketchProblem(open)));
  check('two open ends found', sketchProfiles(open).openEnds.length === 2);
  const withCenterline = [...square, { ...line('f', -1, 1.5, 5, 1.5), construction: true }];
  check('a centerline does not open the sketch', sketchProblem(withCenterline) === null, String(sketchProblem(withCenterline)));
  const dangling = [...square, line('g', 4, 3, 6, 5)];
  check('a stray line is an open contour', /open contour/.test(sketchProblem(dangling) ?? ''));

  const arcR = arcThrough([4, 1], [4, -1], [5, 0], 'r')!;
  const arcL = arcThrough([0, -1], [0, 1], [-1, 0], 'l')!;
  const slot = [line('t', 0, 1, 4, 1), arcR, line('u', 4, -1, 0, -1), arcL];
  check('3 point arc radius', arcR.type === 'arc' && near(arcR.r, 1, 1e-9));
  check('lines and arcs close a loop', sketchProfiles(slot).loops.length === 1 && sketchProfiles(slot).openEnds.length === 0);
  const slotOut = buildExtrude(wasm, feature(top(20), slot, { depth: 1 }), [cast]);
  check('line-and-arc slot area', near(slotOut.volume, 8 + Math.PI, 0.01), String(slotOut.volume));

  const moved = moveEnd(square, 'a', 1, [5, -1]);
  const b2 = moved.find((e) => e.id === 'b') as Extract<SketchEntity, { type: 'line' }>;
  check('dragging a corner drags the joined line', near(b2.x1, 5, 1e-12) && near(b2.y1, -1, 1e-12));
  check('the sketch stays closed after the drag', sketchProblem(moved) === null);
  const longer = withDimension(square[0], 'lineLength', 6) as Extract<SketchEntity, { type: 'line' }>;
  check('line length dimension moves the second end', near(longer.x2, 6, 1e-12) && near(longer.y2, 0, 1e-12));

  const hv = [
    { ...line('a', 0, 0, 4, 0), relation: 'horizontal' as const },
    { ...line('b', 4, 0, 4, 3), relation: 'vertical' as const },
    { ...line('c', 4, 3, 0, 3), relation: 'horizontal' as const },
    { ...line('d', 0, 3, 0, 0), relation: 'vertical' as const }
  ];
  const wider = replaceEntity(hv, withDimension(hv[0], 'lineLength', 6)!);
  const side = wider.find((e) => e.id === 'b') as Extract<SketchEntity, { type: 'line' }>;
  const topLine = wider.find((e) => e.id === 'c') as Extract<SketchEntity, { type: 'line' }>;
  check('a vertical side stays vertical when the base grows', near(side.x1, 6, 1e-12) && near(side.x2, 6, 1e-12), JSON.stringify(side));
  check('the top follows and stays horizontal', near(topLine.x1, 6, 1e-12) && near(topLine.y1, 3, 1e-12));
  check('still closed and a 6 x 3 rectangle', sketchProblem(wider) === null && near(Math.abs(sketchProfiles(wider).loops.reduce((s2, l) => s2 + Math.abs(l.reduce((acc, p, i) => { const q = l[(i + 1) % l.length]; return acc + p[0] * q[1] - q[0] * p[1]; }, 0) / 2), 0)), 18, 1e-9));
  const broken = replaceEntity(hv, { ...hv[0], y2: 1 });
  check('an edit that breaks a relation drops it', !(broken.find((e) => e.id === 'a') as { relation?: string }).relation);

  const simplified = simplifyLoop([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [0, 2]], 1e-6);
  check('collinear points dropped from a traced face', simplified.length === 4, JSON.stringify(simplified));
}

console.log('reference planes');
{
  const base: SketchPlane = { kind: 'standard', id: 'top', offset: 0 };
  const plane = resolveReferencePlane({ id: 'p', name: 'Plane1', source: { kind: 'offset', base, distance: 3, flip: false } });
  check('offset plane', plane?.kind === 'standard' && near(plane.offset, 3, 1e-12) && plane.refId === 'p');
  const flipped = resolveReferencePlane({ id: 'p', name: 'Plane1', source: { kind: 'offset', base, distance: 3, flip: true } });
  check('flipped offset', flipped !== null && near(flipped.offset, -3, 1e-12));
  const face: SketchPlane = { kind: 'custom', normal: [0, 0, -1], offset: -8, label: 'bottom' };
  check('opposed normals are parallel', parallelOffset(base, face) === 8);
  const mid = resolveReferencePlane({ id: 'm', name: 'Mid', source: { kind: 'mid', a: base, b: face } });
  check('mid plane between z=0 and z=8', mid !== null && near(mid.offset, 4, 1e-12));
  check('three offset planes', referencePlanePreview({ kind: 'offset', base, distance: 2, flip: false }, 3, 'P').map((p) => p.offset).join() === '2,4,6');
  const tilted: SketchPlane = { kind: 'custom', normal: [1, 0, 1], offset: 0, label: 't' };
  check('non-parallel mid plane refused', resolveReferencePlane({ id: 'x', name: 'x', source: { kind: 'mid', a: base, b: tilted } }) === null);
  const out = buildExtrude(wasm, feature(plane!, [circle(20, 0, 2)], { depth: 1 }), [cast]);
  check('extrude on an offset plane starts on it', near(out.bbox.min[2], 3, 1e-4));
}

console.log('measure');
{
  const a = measure([{ kind: 'vertex', point: [0, 0, 0], partName: null }, { kind: 'vertex', point: [3, 4, 0], partName: null }]);
  check('vertex to vertex distance', a?.rows[0].value === '5.0000 in', JSON.stringify(a?.rows));
  const tri = new Float32Array(9);
  const faceA = { kind: 'face' as const, point: [0, 0, 0] as [number, number, number], normal: [0, 0, 1] as [number, number, number], planar: true, area: 4, centroid: [0, 0, 0] as [number, number, number], triangles: tri, partName: null };
  const faceB = { ...faceA, normal: [0, 0, -1] as [number, number, number], centroid: [1, 1, 2.5] as [number, number, number] };
  const f = measure([faceA, faceB]);
  check('parallel faces normal distance', f?.rows[0].label === 'Normal distance' && f.rows[0].value === '2.5000 in', JSON.stringify(f?.rows));
  const faceC = { ...faceA, normal: [1, 0, 0] as [number, number, number], centroid: [3, 0, 1] as [number, number, number] };
  const g = measure([faceA, faceC]);
  check('perpendicular faces angle 90', g?.rows[0].value === '90.00°', JSON.stringify(g?.rows));
  const c = measure([{ kind: 'circle', center: [0, 0, 0], axis: [0, 0, 1], radius: 1.5, full: true, length: 3 * Math.PI, points: [[1.5, 0, 0]], partName: null }]);
  check('circle diameter', c?.rows[0].value === '3.0000 in');
  const e = measure([
    { kind: 'line', a: [0, 0, 0], b: [4, 0, 0], points: [], partName: null },
    { kind: 'line', a: [0, 2, 0], b: [4, 2, 0], points: [], partName: null }
  ]);
  check('parallel edges distance', e?.rows[1].value === '2.0000 in' && e.rows[0].value === '0.00°', JSON.stringify(e?.rows));
}

console.log('sketch regions');
{
  const plane = top(0);
  const sketchOf = (entities: SketchEntity[]) => ({ plane, entities });
  const pair: SketchEntity[] = [
    { id: 'a', type: 'circle', cx: 0, cy: 0, d: 4 },
    { id: 'b', type: 'circle', cx: 2, cy: 0, d: 4 }
  ];
  const regions = sketchRegions(wasm, sketchOf(pair));
  check('two overlapping circles make three regions', regions.length === 3, String(regions.length));
  const lens = regions.find((region) => !region.inDefault);
  check('the even-odd default leaves the lens out', regions.filter((region) => region.inDefault).length === 2 && lens !== undefined);
  check('every seed lies in its own region only', regions.every((region) => regions.filter((other) => pointInContours(other.contours, region.seed)).length === 1 && pointInContours(region.contours, region.seed)));
  check('regions carry triangles', regions.every((region) => region.triangles.length >= 18));
  const areaSum = regions.reduce((sum, region) => sum + region.area, 0);
  const extrude = (patch: Partial<ExtrudeFeature>) => buildExtrude(wasm, feature(plane, pair, { depth: 1, ...patch }), []);
  const all = extrude({ regions: regions.map((region) => region.seed) });
  check('all regions extrude the union', near(all.volume, areaSum, 1e-3), `${all.volume} vs ${areaSum}`);
  const lensOnly = extrude({ regions: [lens!.seed] });
  check('the lens alone', near(lensOnly.volume, lens!.area, 1e-4), `${lensOnly.volume} vs ${lens!.area}`);
  const byDefault = extrude({});
  check('no picks = even-odd', near(byDefault.volume, areaSum - lens!.area, 1e-3), String(byDefault.volume));
  try {
    extrude({ regions: [[40, 40]] });
    check('refuses: picks outside every region', false);
  } catch (error) {
    check('refuses: picks outside every region', error instanceof FeatureBuildError && /picked regions/.test((error as Error).message), String(error));
  }

  const ring = sketchRegions(wasm, sketchOf([
    { id: 'r', type: 'rectangle', cx: 0, cy: 0, w: 6, h: 6, angle: 0 },
    { id: 'c', type: 'circle', cx: 0, cy: 0, d: 2 }
  ]));
  check('rectangle with a circle: ring and disc', ring.length === 2 && ring.filter((region) => region.inDefault).length === 1);
  const ringRegion = ring.find((region) => region.inDefault)!;
  check('the ring keeps its hole', ringRegion.contours.length === 2 && !pointInContours(ringRegion.contours, [0, 0]));

  const lines: SketchEntity[] = [
    { id: 'l1', type: 'line', x1: 0, y1: 0, x2: 4, y2: 0 },
    { id: 'l2', type: 'line', x1: 4, y1: 0, x2: 4, y2: 4 },
    { id: 'l3', type: 'line', x1: 4, y1: 4, x2: 0, y2: 4 },
    { id: 'l4', type: 'line', x1: 0, y1: 4, x2: 0, y2: 0 },
    { id: 'l5', type: 'line', x1: 1, y1: 0, x2: 1, y2: 4 }
  ];
  const split = sketchRegions(wasm, sketchOf(lines));
  check('a line across a square splits it in two', split.length === 2 && split.every((region) => region.inDefault), String(split.length));
  const strip = split.find((region) => near(region.area, 4, 1e-6))!;
  const stripOnly = buildExtrude(wasm, feature(plane, lines, { depth: 2, regions: [strip.seed] }), []);
  check('one of the two extrudes alone', near(stripOnly.volume, 8, 1e-4), String(stripOnly.volume));

  const slab = Manifold.cube([10, 10, 1]).translate([-5, -5, 0]);
  const slabMesh = slab.getMesh();
  const cutTarget: CutTarget = { name: 'Slab', positions: slabMesh.vertProperties, indices: slabMesh.triVerts, matrix: identity };
  const lensCut = applyCuts(wasm, cutTarget, [
    { ...defaultCut(5, top(2), 'Cut'), partIds: ['x'], sketch: { plane: top(2), entities: pair }, endCondition: 'throughAll', reverse: true, regions: [lens!.seed] }
  ]);
  check('a cut through the lens only', lensCut.changed && near(100 - lensCut.volume, lens!.area, 1e-3), lensCut.changed ? String(100 - lensCut.volume) : '');
  slab.delete();
}

console.log('extrude cut');
{
  const cutOf = (plane: SketchPlane, entities: SketchEntity[], patch: Partial<CutFeature>): CutFeature => ({
    ...defaultCut(5, plane, 'Cut-Extrude1'),
    partIds: ['cast'],
    ...patch,
    sketch: { plane, entities }
  });
  const target = (matrix = identity): CutTarget => ({ name: 'Casting', positions: soup, indices: soupIdx, matrix });
  const closed = (out: { positions: Float32Array; indices: Uint32Array }) => {
    const back = bodyFromMesh(wasm, 'x', 'x', out.positions, out.indices, identity);
    const ok = back.manifold !== null && back.manifold.status() === 'NoError';
    back.manifold?.delete();
    return ok;
  };
  const castVolume = castSolid.volume();
  const refuses = (label: string, run: () => unknown, pattern: RegExp) => {
    try {
      run();
      check(`refuses: ${label}`, false);
    } catch (error) {
      check(`refuses: ${label}`, error instanceof FeatureBuildError && pattern.test((error as Error).message), String(error));
    }
  };

  const hole = cutOf(top(12), [circle(0, 0, 2)], { endCondition: 'throughAll', reverse: true });
  const expected = 2 * Math.PI * (2 + (125 - Math.pow(24, 1.5)) / 3);
  const preview = previewCut(wasm, hole, [cast]);
  check('preview removes the hole volume', near(preview.removed.cast, expected, expected * 0.01), `${preview.removed.cast} vs ${expected}`);
  const holed = applyCuts(wasm, target(), [hole]);
  check('through all cuts the body', holed.changed);
  if (holed.changed) {
    check('cut volume = body - removed', near(holed.volume, castVolume - preview.removed.cast, 1e-3), `${holed.volume}`);
    check('cut body is closed once stored', closed(holed));
    check('cut body keeps its outline', near(holed.bbox.min[2], 0, 1e-6) && near(holed.bbox.max[0] - holed.bbox.min[0], 10, 1e-3));
  }

  const moved = partWorldMatrix([10, -3, 2], [0, 0, 90], 1);
  const movedHole = cutOf({ kind: 'standard', id: 'top', offset: 14 }, [circle(10, -3, 2)], { endCondition: 'throughAll', reverse: true });
  const movedOut = applyCuts(wasm, target(moved), [movedHole]);
  check(
    'a placed body is cut in its own frame',
    movedOut.changed && near(movedOut.volume, castVolume - preview.removed.cast, 1e-3) && near(movedOut.bbox.min[2], 0, 1e-6),
    movedOut.changed ? `${movedOut.volume} ${JSON.stringify(movedOut.bbox)}` : 'unchanged'
  );
  const scaled = partWorldMatrix([0, 0, 0], [0, 0, 0], 2);
  const scaledOut = applyCuts(wasm, target(scaled), [cutOf(top(30), [circle(0, 0, 4)], { endCondition: 'throughAll', reverse: true })]);
  check(
    'a scaled body is cut in its own frame',
    scaledOut.changed && near(scaledOut.volume, castVolume - preview.removed.cast, expected * 0.01),
    scaledOut.changed ? String(scaledOut.volume) : 'unchanged'
  );

  const pocket = applyCuts(wasm, target(), [cutOf(top(0), [{ id: 'r', type: 'rectangle', cx: 0, cy: 0, w: 2, h: 2, angle: 0 }], { endCondition: 'blind', depth: 1, reverse: false })]);
  check('blind pocket removes w*h*d', pocket.changed && near(castVolume - pocket.volume, 4, 1e-3), pocket.changed ? String(castVolume - pocket.volume) : 'unchanged');

  const cylinder = Manifold.cylinder(4, 5, 5, 96);
  const cylMesh = cylinder.getMesh();
  const cylTarget: CutTarget = { name: 'Cylinder', positions: cylMesh.vertProperties, indices: cylMesh.triVerts, matrix: identity };
  const cylVolume = cylinder.volume();
  const mid = applyCuts(wasm, cylTarget, [cutOf(top(2), [circle(0, 0, 2)], { endCondition: 'midplane', depth: 2 })]);
  const holeArea = cylVolume / 4 / 25;
  check('mid plane cuts both ways from the sketch', mid.changed && near(cylVolume - mid.volume, holeArea * 2, 0.02), mid.changed ? String(cylVolume - mid.volume) : 'unchanged');
  check('a cut that stops inside leaves a void, and says so', mid.changed && closed(mid) && mid.warnings.some((w) => /enclosed void/.test(w)), JSON.stringify(mid.warnings));
  const both = applyCuts(wasm, cylTarget, [cutOf(top(2), [circle(0, 0, 2)], { endCondition: 'throughAllBoth' })]);
  check('through all both cuts the whole height', both.changed && near(cylVolume - both.volume, holeArea * 4, 0.03), both.changed ? String(cylVolume - both.volume) : 'unchanged');
  const drafted = applyCuts(wasm, cylTarget, [cutOf(top(4), [circle(0, 0, 4)], { endCondition: 'blind', depth: 2, reverse: true, draftDeg: 10 })]);
  const r2 = 2 - 2 * Math.tan((10 * Math.PI) / 180);
  const frustum = (Math.PI * 2 * (4 + 2 * r2 + r2 * r2)) / 3;
  check('drafted cut tapers inward', drafted.changed && near(cylVolume - drafted.volume, frustum, frustum * 0.02), drafted.changed ? `${cylVolume - drafted.volume} vs ${frustum}` : 'unchanged');

  const flipped = applyCuts(wasm, cylTarget, [cutOf(top(2), [circle(0, 0, 4)], { endCondition: 'throughAllBoth', flipSide: true })]);
  check('flip side keeps the inside of the profile', flipped.changed && near(flipped.volume, holeArea * 4 * 4, 0.1), flipped.changed ? String(flipped.volume) : 'unchanged');
  if (flipped.changed) check('flipped cut is closed once stored', closed(flipped));

  const twice = applyCuts(wasm, cylTarget, [
    cutOf(top(2), [circle(0, 0, 2)], { endCondition: 'throughAllBoth', seq: 7 }),
    cutOf(top(2), [circle(3, 0, 1)], { endCondition: 'throughAllBoth', seq: 6 })
  ]);
  check('two cuts both apply', twice.changed && near(cylVolume - twice.volume, holeArea * 4 + holeArea * 4 * 0.25, 0.05), twice.changed ? String(cylVolume - twice.volume) : '');
  const missed = applyCuts(wasm, cylTarget, [cutOf(top(10), [circle(0, 0, 2)], { endCondition: 'throughAll', reverse: false })]);
  check('a cut pointing away leaves the body unchanged', !missed.changed);
  const aside = applyCuts(wasm, cylTarget, [cutOf(top(10), [circle(20, 0, 2)], { endCondition: 'throughAll', reverse: true })]);
  check('a cut beside the body leaves it unchanged', !aside.changed);

  refuses('cut that removes the whole body', () => applyCuts(wasm, cylTarget, [cutOf(top(10), [circle(0, 0, 30)], { endCondition: 'throughAll', reverse: true })]), /removes all/);
  refuses('preview that removes nothing', () => previewCut(wasm, cutOf(top(10), [circle(20, 0, 2)], { reverse: true }), [cast]), /removes no material/);
  refuses('preview with no bodies selected', () => previewCut(wasm, cutOf(top(10), [circle(0, 0, 2)], { partIds: [] }), [cast]), /Select the bodies/);
  cylinder.delete();
}

cast.manifold?.delete();
castSolid.delete();
if (failures) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('all kernel checks passed');
