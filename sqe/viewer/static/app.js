/* Spatial query engine viewer.
 *
 * Plain three.js, no build step. The one thing worth explaining is the frame
 * table: after a query it lists what each reference frame would have answered,
 * and clicking a row re-runs under that frame. Watching the highlighted mug
 * jump when you switch from "egocentric" to "intrinsic" is the fastest way to
 * see what the whole project is about.
 *
 * Orbiting the camera with "viewpoint = my camera" ticked makes the egocentric
 * frame follow the camera, so "the second from the left" changes as you walk
 * round the room. That is not a gimmick: it is the honest behaviour, and the
 * usual alternative is a fixed convention nobody tells you about.
 */
'use strict';

const S = {
  scene: null, camera: null, renderer: null, root: null,
  cloud: null, boxGroup: null, axesGroup: null, frontGroup: null,
  trajGroup: null, roomGroup: null, linkGroup: null,
  sceneData: null, sceneId: null, objects: new Map(),
  selected: null, lastResult: null, annTargets: [],
  tags: [], annotating: false,
  orbit: { theta: 0.9, phi: 1.0, radius: 8, target: new THREE.Vector3() },
};

const COL = { dim: 0x39414d, target: 0xffd166, anchor: 0xef476f,
              cand: 0x5aa9e6, sel: 0xffffff,
              right: 0xff5d5d, front: 0x5dff8f, up: 0x6db3ff };

/* ---------------------------------------------------------------- utils */
const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => { const e = document.createElement(tag);
  if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

async function getJSON(url, body) {
  const opt = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' },
                       body: JSON.stringify(body) } : {};
  const r = await fetch(url, opt);
  return r.json();
}

/* ------------------------------------------------------------- three.js */
function initGL() {
  const canvas = $('gl');
  S.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  S.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  S.scene = new THREE.Scene();
  S.scene.background = new THREE.Color(0x101317);
  S.camera = new THREE.PerspectiveCamera(55, 1, 0.02, 400);

  S.root = new THREE.Group();
  // world is Z-up; three.js is Y-up, so rotate the whole scene once here and
  // never think about it again
  S.root.rotation.x = -Math.PI / 2;
  S.scene.add(S.root);

  for (const k of ['boxGroup', 'axesGroup', 'frontGroup', 'trajGroup',
                   'roomGroup', 'linkGroup']) {
    S[k] = new THREE.Group();
    S.root.add(S[k]);
  }
  S.scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.0));
  bindOrbit(canvas);
  onResize();
  window.addEventListener('resize', onResize);
  animate();
}

function onResize() {
  const w = $('canvas-wrap').clientWidth, h = $('canvas-wrap').clientHeight;
  S.renderer.setSize(w, h, false);
  S.camera.aspect = w / Math.max(h, 1);
  S.camera.updateProjectionMatrix();
}

function bindOrbit(canvas) {
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, shift: e.shiftKey, moved: 0 };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy);
    drag.x = e.clientX; drag.y = e.clientY;
    const o = S.orbit;
    if (drag.shift) {
      const right = new THREE.Vector3(-Math.sin(o.theta), Math.cos(o.theta), 0);
      const upv = new THREE.Vector3(0, 0, 1);
      o.target.addScaledVector(right, -dx * o.radius * 0.0016);
      o.target.addScaledVector(upv, dy * o.radius * 0.0016);
    } else {
      o.theta -= dx * 0.006;
      o.phi = Math.min(Math.PI - 0.05, Math.max(0.05, o.phi - dy * 0.006));
    }
    if (S.useCameraViewpoint()) scheduleReRun();
  });
  canvas.addEventListener('pointerup', (e) => {
    if (drag && drag.moved < 5) pick(e);
    drag = null;
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    S.orbit.radius = Math.min(90, Math.max(0.4,
      S.orbit.radius * Math.exp(e.deltaY * 0.0013)));
    if (S.useCameraViewpoint()) scheduleReRun();
  }, { passive: false });
}

function cameraEye() {
  const o = S.orbit;
  return new THREE.Vector3(
    o.target.x + o.radius * Math.sin(o.phi) * Math.cos(o.theta),
    o.target.y + o.radius * Math.sin(o.phi) * Math.sin(o.theta),
    o.target.z + o.radius * Math.cos(o.phi));
}

function updateCamera() {
  const eye = cameraEye();
  // convert world Z-up into three's Y-up view space
  S.camera.position.set(eye.x, eye.z, -eye.y);
  S.camera.up.set(0, 1, 0);
  S.camera.lookAt(S.orbit.target.x, S.orbit.target.z, -S.orbit.target.y);
}

let frames = 0, lastFps = performance.now();
function animate() {
  requestAnimationFrame(animate);
  updateCamera();
  S.renderer.render(S.scene, S.camera);
  drawTags();
  if (++frames % 30 === 0) {
    const now = performance.now();
    $('fps').textContent = (30000 / (now - lastFps)).toFixed(0) + ' fps';
    lastFps = now;
  }
}

/* ------------------------------------------------------------- geometry */
function clearGroup(g) {
  while (g.children.length) {
    const c = g.children.pop();
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }
}

async function loadCloud(sceneId) {
  const r = await fetch('/api/cloud/' + encodeURIComponent(sceneId));
  const buf = await r.arrayBuffer();
  const n = new Uint32Array(buf, 0, 1)[0];
  const xyz = new Float32Array(buf, 8, n * 3);
  const rgb = new Uint8Array(buf, 8 + n * 12, n * 3);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) col[i] = Math.pow(rgb[i] / 255, 1.6);
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  if (S.cloud) { S.root.remove(S.cloud); S.cloud.geometry.dispose(); }
  S.cloud = new THREE.Points(geom, new THREE.PointsMaterial({
    size: parseFloat($('psize').value) * 0.01, vertexColors: true,
    sizeAttenuation: true }));
  S.cloud.visible = $('show-cloud').checked;
  S.root.add(S.cloud);
  return n;
}

function boxLines(o, color, width) {
  const g = new THREE.BoxGeometry(o.extent[0], o.extent[1], o.extent[2]);
  const edges = new THREE.EdgesGeometry(g);
  g.dispose();
  const m = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color }));
  const R = o.R;
  const mat = new THREE.Matrix4().set(
    R[0][0], R[0][1], R[0][2], o.center[0],
    R[1][0], R[1][1], R[1][2], o.center[1],
    R[2][0], R[2][1], R[2][2], o.center[2],
    0, 0, 0, 1);
  m.applyMatrix4(mat);
  m.userData.id = o.id;
  return m;
}

function arrow(from, dir, len, color) {
  const d = new THREE.Vector3(dir[0], dir[1], dir[2]).normalize();
  const a = new THREE.ArrowHelper(d,
    new THREE.Vector3(from[0], from[1], from[2]), len, color,
    len * 0.22, len * 0.12);
  return a;
}

function rebuildBoxes() {
  clearGroup(S.boxGroup);
  clearGroup(S.frontGroup);
  if (!S.sceneData) return;
  const res = S.lastResult;
  const targetId = res ? res.target_id : null;
  const anchorIds = new Set((res && res.anchors || [])
    .map((a) => a.object_id).filter((x) => x != null));
  const candIds = new Set((res && res.candidates || [])
    .slice(0, 8).map((c) => c.object_id));

  for (const o of S.sceneData.objects) {
    let color = COL.dim, show = $('show-boxes').checked;
    if (o.id === targetId) { color = COL.target; show = true; }
    else if (anchorIds.has(o.id)) { color = COL.anchor; show = true; }
    else if (candIds.has(o.id)) { color = COL.cand; show = true; }
    if (S.selected === o.id) { color = COL.sel; show = true; }
    if (S.annTargets.includes(o.id)) { color = COL.target; show = true; }
    if (!show) continue;
    S.boxGroup.add(boxLines(o, color));
  }
  if ($('show-fronts').checked) {
    for (const o of S.sceneData.objects) {
      if (!o.front) continue;
      const len = Math.max(0.25, Math.min(0.8, o.extent[0] * 0.8));
      S.frontGroup.add(arrow(o.center, o.front, len,
        o.front_confidence > 0.5 ? 0x06d6a0 : 0xf4a261));
    }
  }
}

function rebuildFrameAxes(frameDict, origin, chosen) {
  clearGroup(S.axesGroup);
  if (!$('show-axes').checked || !frameDict || !origin) return;
  const order = ['egocentric', 'intrinsic', 'addressee', 'world'];
  let i = 0;
  for (const k of order) {
    const f = frameDict[k];
    if (!f || !f.available) continue;
    const scale = k === chosen ? 1.0 : 0.55;
    const o = [origin[0], origin[1], origin[2] + i * 0.06];
    S.axesGroup.add(arrow(o, f.right, 0.7 * scale, COL.right));
    S.axesGroup.add(arrow(o, f.front, 0.7 * scale, COL.front));
    if (k === chosen) S.axesGroup.add(arrow(o, f.up, 0.5, COL.up));
    i++;
  }
}

function rebuildLinks() {
  clearGroup(S.linkGroup);
  const res = S.lastResult;
  if (!res || res.target_id == null) return;
  const t = S.objects.get(res.target_id);
  if (!t) return;
  for (const a of res.anchors || []) {
    if (a.object_id == null) continue;
    const o = S.objects.get(a.object_id);
    if (!o) continue;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...t.center), new THREE.Vector3(...o.center)]);
    S.linkGroup.add(new THREE.Line(g, new THREE.LineDashedMaterial({
      color: COL.anchor, dashSize: 0.08, gapSize: 0.05 })));
    S.linkGroup.children[S.linkGroup.children.length - 1].computeLineDistances();
  }
}

function rebuildRoom() {
  clearGroup(S.roomGroup);
  const r = S.sceneData && S.sceneData.room;
  if (!r || !$('show-room').checked) return;
  const c = r.bounds.center;
  if (r.canonical_forward) {
    S.roomGroup.add(arrow([c[0], c[1], r.floor_z + 0.05],
      r.canonical_forward, 1.2, 0xffffff));
  }
  for (const cand of r.forward_candidates || []) {
    S.roomGroup.add(arrow([c[0], c[1], r.floor_z + 0.05], cand, 0.6, 0x4a5563));
  }
  const b = r.bounds;
  S.roomGroup.add(boxLines({ id: -1, center: b.center, extent: b.extent,
                             R: b.R }, 0x2a323c));
}

function rebuildTraj() {
  clearGroup(S.trajGroup);
  const t = S.sceneData && S.sceneData.trajectory;
  if (!t || !$('show-traj').checked) return;
  const pts = t.centers.map((c) => new THREE.Vector3(c[0], c[1], c[2]));
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  S.trajGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({
    color: 0x3d6b8c })));
}

/* ------------------------------------------------------------- picking */
function pick(ev) {
  if (!S.sceneData) return;
  const rect = $('gl').getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, S.camera);
  // pick by distance from the ray to each object centre, scaled by size
  let best = null, bestD = Infinity;
  const p = new THREE.Vector3();
  for (const o of S.sceneData.objects) {
    p.set(o.center[0], o.center[2], -o.center[1]);   // into view space
    const d = ray.ray.distanceToPoint(p);
    const r = Math.max(0.08, 0.6 * Math.hypot(o.extent[0], o.extent[1],
                                              o.extent[2]));
    if (d < r && d / r < bestD) { bestD = d / r; best = o; }
  }
  if (!best) return;
  if (S.annotating) {
    if (ev.shiftKey) {
      if (!S.annTargets.includes(best.id)) S.annTargets.push(best.id);
    } else S.annTargets = [best.id];
    $('ann-targets').textContent = 'targets: ' +
      (S.annTargets.length ? S.annTargets.join(', ') : '(none)');
  }
  selectObject(best.id);
}

async function selectObject(id) {
  S.selected = id;
  const o = S.objects.get(id);
  if (!o) return;
  $('inspect-box').classList.remove('hidden');
  $('inspect').textContent =
    `#${o.id}  ${o.label}\n` +
    `centre  (${o.center.map((v) => v.toFixed(2)).join(', ')})\n` +
    `extent  ${o.extent.map((v) => v.toFixed(2)).join(' x ')} m\n` +
    `front   ${o.front ? o.front.map((v) => v.toFixed(2)).join(', ')
                        : 'none'}  conf ${o.front_confidence.toFixed(2)}\n` +
    `method  ${o.front_method}\n` +
    (o.levels && o.levels.length
      ? `shelves ${o.levels.map((z) => z.toFixed(2)).join(', ')}\n` : '') +
    `points  ${o.point_count}`;
  const body = { scene_id: S.sceneId, object_id: id };
  Object.assign(body, viewpointBody());
  const fr = await getJSON('/api/frames', body);
  if (fr.frames) {
    const lines = Object.entries(fr.frames).map(([k, f]) =>
      f.available
        ? `${k.padEnd(19)} right (${f.right[0].toFixed(2)}, ${f.right[1].toFixed(2)})` +
          `  conf ${f.confidence.toFixed(2)}  ${f.handedness > 0 ? 'RH' : 'LH'}`
        : `${k.padEnd(19)} unavailable: ${f.reason}`);
    $('obj-frames').textContent = 'viewpoint: ' + fr.viewpoint.source + '\n'
      + lines.join('\n');
    rebuildFrameAxes(fr.frames, o.center, null);
  }
  renderObjectList();
  rebuildBoxes();
}

/* --------------------------------------------------------------- tags */
function drawTags() {
  const ov = $('overlay');
  if (S.tags.length === 0) { if (ov.childElementCount) ov.textContent = ''; return; }
  while (ov.childElementCount < S.tags.length) ov.appendChild(el('div', 'tag'));
  while (ov.childElementCount > S.tags.length) ov.removeChild(ov.lastChild);
  const rect = $('gl').getBoundingClientRect();
  S.tags.forEach((t, i) => {
    const node = ov.children[i];
    node.className = 'tag ' + (t.cls || '');
    node.textContent = t.text;
    const v = new THREE.Vector3(t.pos[0], t.pos[2], -t.pos[1]);
    v.project(S.camera);
    if (v.z > 1) { node.style.display = 'none'; return; }
    node.style.display = '';
    node.style.left = ((v.x * 0.5 + 0.5) * rect.width) + 'px';
    node.style.top = ((-v.y * 0.5 + 0.5) * rect.height - 6) + 'px';
  });
}

/* ------------------------------------------------------------- queries */
function viewpointBody() {
  if ($('use-camera').checked) {
    const e = cameraEye();
    return { viewpoint_mode: 'position', viewpoint_position: [e.x, e.y, e.z],
             look_at: [S.orbit.target.x, S.orbit.target.y, S.orbit.target.z] };
  }
  return { viewpoint_mode: 'best_view' };
}

let reRunTimer = null;
function scheduleReRun() {
  if (!$('q').value.trim()) return;
  clearTimeout(reRunTimer);
  reRunTimer = setTimeout(runQuery, 180);
}
S.useCameraViewpoint = () => $('use-camera').checked;

async function runQuery(forceFrame) {
  const text = $('q').value.trim();
  if (!text) return;
  const body = { scene_id: S.sceneId, text,
                 force_frame: forceFrame != null ? forceFrame
                                                 : ($('force-frame').value || null) };
  Object.assign(body, viewpointBody());
  const res = await getJSON('/api/query', body);
  if (res.error) {
    $('answer-box').classList.remove('hidden');
    $('answer').textContent = 'error: ' + res.error;
    return;
  }
  S.lastResult = res;
  renderAnswer(res);
  rebuildBoxes();
  rebuildLinks();
  const anchor = (res.anchors || []).find((a) => a.object_id != null);
  if (res.frame && anchor) {
    const o = S.objects.get(anchor.object_id);
    if (o) rebuildFrameAxes(res.frame.frames, o.center, res.frame.chosen);
  } else {
    clearGroup(S.axesGroup);
  }
  S.tags = [];
  if (res.target_id != null) {
    const o = S.objects.get(res.target_id);
    if (o) S.tags.push({ pos: o.center, text: `${o.label} #${o.id}`,
                          cls: 'target' });
  }
  for (const a of res.anchors || []) {
    if (a.object_id == null) continue;
    const o = S.objects.get(a.object_id);
    if (o) S.tags.push({ pos: o.center, text: `anchor: ${o.label}`,
                          cls: 'anchor' });
  }
}

function renderAnswer(res) {
  $('answer-box').classList.remove('hidden');
  $('explain-box').classList.remove('hidden');
  const a = $('answer');
  a.textContent = '';
  if (res.target_id == null) {
    a.appendChild(el('div', null, 'no object matched'));
  } else {
    const o = S.objects.get(res.target_id);
    const top = res.candidates[0];
    a.appendChild(el('div', 'big',
      `${o.label} #${o.id}   score ${top.score.toFixed(3)}`));
    if (res.candidates[1]) {
      const r = S.objects.get(res.candidates[1].object_id);
      a.appendChild(el('div', 'small',
        `runner-up: ${r.label} #${r.id} (${res.candidates[1].score.toFixed(3)})`));
    }
  }
  if (res.frame && res.frame.chosen) {
    a.appendChild(el('div', 'small',
      `frame: ${res.frame.chosen} ` +
      `(${res.frame.explicit ? 'stated in the query' : 'policy default'})` +
      `  ·  viewpoint: ${res.frame.viewpoint.source}`));
  }

  const ft = $('frame-table');
  ft.textContent = '';
  const answers = res.frame_answers || {};
  if (Object.keys(answers).length) {
    const t = el('table');
    const scores = (res.ambiguity.detail || {}).frame_scores || {};
    for (const [k, v] of Object.entries(answers)) {
      const tr = el('tr', 'clickable' + (k === (res.frame && res.frame.chosen)
        ? ' chosen' : ''));
      tr.appendChild(el('td', null, k));
      const o = v != null ? S.objects.get(v) : null;
      tr.appendChild(el('td', null, o ? `${o.label} #${o.id}` : 'no answer'));
      tr.appendChild(el('td', null, scores[k] != null
        ? scores[k].toFixed(2) : ''));
      tr.onclick = () => { $('force-frame').value = k; runQuery(k); };
      t.appendChild(tr);
    }
    ft.appendChild(el('div', 'small', 'answer under each frame (click to force)'));
    ft.appendChild(t);
  }

  const amb = $('ambiguity');
  amb.textContent = '';
  if (res.ambiguity && res.ambiguity.ambiguous) {
    for (const k of res.ambiguity.kinds) amb.appendChild(el('span', 'kind', k));
    for (const m of res.ambiguity.messages) amb.appendChild(el('div', null, m));
  }
  $('explain').textContent = res.explanation || '';
}

/* ---------------------------------------------------------------- lists */
function renderObjectList() {
  const box = $('objects');
  const f = $('filter').value.trim().toLowerCase();
  box.textContent = '';
  let n = 0;
  for (const o of S.sceneData.objects) {
    if (f && !o.label.toLowerCase().includes(f)) continue;
    n++;
    const d = el('div', 'o' + (S.selected === o.id ? ' sel' : ''));
    d.appendChild(el('span', 'id', '#' + o.id));
    d.appendChild(el('span', null, o.label));
    d.appendChild(el('span', o.front ? 'f' : 'id',
      o.front ? 'front ' + o.front_confidence.toFixed(2)
              : (o.has_intrinsic_front ? 'no front' : '')));
    d.onclick = () => selectObject(o.id);
    box.appendChild(d);
  }
  $('obj-count').textContent = `${n}/${S.sceneData.objects.length}`;
}

const EXAMPLES = [
  'the second mug from the left on the middle shelf',
  'the mug to the left of the laptop',
  'the object in front of the sofa',
  'the leftmost monitor',
  'the trash can nearest to the door',
  'the keyboard on the table',
  'the monitor to the left of the keyboard',
  'the third chair from the door',
];

function renderExamples() {
  const box = $('examples');
  box.textContent = '';
  box.appendChild(el('div', null, 'try:'));
  for (const t of EXAMPLES) {
    const a = el('a', null, t);
    a.onclick = () => { $('q').value = t; runQuery(); };
    box.appendChild(a);
  }
}

/* ----------------------------------------------------------------- boot */
async function loadScene(sceneId) {
  S.sceneId = sceneId;
  S.sceneData = await getJSON('/api/scene/' + encodeURIComponent(sceneId));
  S.objects = new Map(S.sceneData.objects.map((o) => [o.id, o]));
  S.lastResult = null; S.selected = null; S.annTargets = []; S.tags = [];
  $('answer-box').classList.add('hidden');
  $('explain-box').classList.add('hidden');
  $('inspect-box').classList.add('hidden');

  const n = await loadCloud(sceneId);
  const r = S.sceneData.room;
  if (r) {
    S.orbit.target.set(r.bounds.center[0], r.bounds.center[1],
                       r.floor_z + 1.0);
    S.orbit.radius = Math.max(4, 1.15 * Math.hypot(r.bounds.extent[0],
                                                   r.bounds.extent[1]));
    const nf = S.sceneData.objects.filter((o) => o.front).length;
    const nfc = S.sceneData.objects.filter((o) => o.has_intrinsic_front).length;
    $('scene-info').textContent =
      `${S.sceneData.objects.length} objects · ${n.toLocaleString()} points\n` +
      `room ${r.bounds.extent[0].toFixed(1)} x ${r.bounds.extent[1].toFixed(1)} m` +
      ` · manhattan ${r.axis_confidence.toFixed(2)}\n` +
      `canonical forward '${r.forward_convention}' margin ` +
      `${r.forward_margin.toFixed(3)}` +
      (r.forward_margin < 0.12 ? '  (UNDETERMINED)' : '') + '\n' +
      `fronts ${nf}/${nfc} of front-bearing objects`;
  }
  rebuildBoxes(); rebuildRoom(); rebuildTraj(); renderObjectList();
}

async function boot() {
  if (typeof THREE === 'undefined') {
    $('boot').textContent =
      'three.js could not be loaded from the CDN. The viewer needs it; '
      + 'the rest of the toolkit works offline.';
    return;
  }
  initGL();
  $('boot').style.display = 'none';
  const info = await getJSON('/api/scenes');
  S.annotating = !!info.annotating;
  if (S.annotating) $('annotate-box').classList.remove('hidden');
  const sel = $('scene');
  for (const s of info.scenes) sel.appendChild(el('option', null, s));
  sel.onchange = () => loadScene(sel.value);
  renderExamples();

  $('run').onclick = () => runQuery();
  $('q').addEventListener('keydown', (e) => { if (e.key === 'Enter') runQuery(); });
  $('force-frame').onchange = () => runQuery();
  $('use-camera').onchange = () => runQuery();
  $('filter').oninput = renderObjectList;
  for (const id of ['show-boxes', 'show-fronts', 'show-axes']) {
    $(id).onchange = () => { rebuildBoxes();
      if (id === 'show-axes' && !$(id).checked) clearGroup(S.axesGroup); };
  }
  $('show-cloud').onchange = () => { if (S.cloud) S.cloud.visible = $('show-cloud').checked; };
  $('show-room').onchange = rebuildRoom;
  $('show-traj').onchange = rebuildTraj;
  $('psize').oninput = () => {
    $('psize-val').textContent = $('psize').value;
    if (S.cloud) S.cloud.material.size = parseFloat($('psize').value) * 0.01;
  };
  $('psize-val').textContent = $('psize').value;

  $('ann-save').onclick = async () => {
    const r = await getJSON('/api/annotate', Object.assign({
      scene_id: S.sceneId, text: $('q').value.trim(),
      target_ids: S.annTargets, frame: $('ann-frame').value,
      frame_stated: $('ann-stated').checked,
      ambiguous: $('ann-amb').checked,
      ambiguity_kind: $('ann-kind').value,
      difficulty: $('ann-diff').value }, viewpointBody()));
    $('ann-status').textContent = r.error
      ? 'error: ' + r.error
      : `saved ${r.saved} (${r.n_items} items)` +
        (r.problems && r.problems.length ? ' — ' + r.problems.join('; ') : '');
  };

  if (info.scenes.length) await loadScene(info.scenes[0]);
}

boot();
