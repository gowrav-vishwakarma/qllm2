// Three.js scene setup for the architecture overview
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { clamp, getDtSchedule } from '../presets.js';

let scene, camera, renderer, labelRenderer, controls, modelGroup;
let gridHelper, dirLight;
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hoveredObj = null;
const clickableObjects = [];
const objectData = new Map();
let originalColors = new Map();
let _animId = null;
let _onBlockClick = null;

// Data-flow particle system
let particlesEnabled = false;
let particleGroup = null;
let particles = [];
let flowWaypoints = []; // [{y, color}] set by builder
const PARTICLE_COUNT = 12;
const PARTICLE_SPEED = 0.03;

export const COLORS = {
  embed:0xf0c674,cgu:0x4a90d9,pam:0x2ecc71,attn:0x00bcd4,
  semBank:0x5dade2,ctxBank:0x2e86c1,coupler:0x8e44ad,memory:0x9b59b6,
  norm:0x5a6070,output:0xe67e22,aux:0xe74c3c,container:0x1e2a38,
  residual:0xf39c12,drift:0xe91e63,
};

export function dtColor(dt) {
  const t = Math.max(0, Math.min(1, (dt + 7) / 7));
  const h = (240 - t * 180) / 360;
  return new THREE.Color().setHSL(h, 0.75, 0.55);
}

export function dimScale(cfg) {
  const dim = cfg.dim || 384;
  const heads = cfg.n_heads || cfg.pam_num_heads || 6;
  const headDim = cfg.head_dim || cfg.pam_head_dim || 64;
  const expand = cfg.expand || cfg.bank_expand || 3;
  return {
    BW: clamp(2.0 + 4.5 * Math.sqrt(dim / 512), 2.5, 9),
    BD: clamp(1.2 + 1.8 * Math.sqrt(heads / 8), 1.2, 4.5),
    SUB_H: clamp(0.3 + 0.35 * Math.sqrt(headDim / 64), 0.3, 0.9),
    CGU_D_MULT: clamp(0.6 + 0.35 * (expand / 3), 0.5, 1.4),
  };
}

export function makeBox(w, h, d, color, opacity = 1.0) {
  const geo = new THREE.BoxGeometry(w, h, d);
  const mat = new THREE.MeshPhongMaterial({
    color, transparent: opacity < 1, opacity, shininess: 40, specular: 0x222244,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export function makeLabel(text, cls = 'label3d') {
  const el = document.createElement('div');
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

export function addClickable(mesh, data) {
  clickableObjects.push(mesh);
  objectData.set(mesh, data);
}

export function makeLine(points, color) {
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
}

export function makeCurve(from, to, color, midOffset) {
  const mid = new THREE.Vector3(
    (from.x + to.x) / 2 + (midOffset?.x || 0),
    (from.y + to.y) / 2 + (midOffset?.y || 0),
    (from.z + to.z) / 2 + (midOffset?.z || 0),
  );
  return makeLine(new THREE.QuadraticBezierCurve3(from, mid, to).getPoints(20), color);
}

export function addHeadStripes(parent, w, h, d, nHeads) {
  const mat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25 });
  for (let i = 1; i < nHeads; i++) {
    const frac = (i / nHeads - 0.5) * d;
    const pts = [new THREE.Vector3(-w/2, 0, frac), new THREE.Vector3(w/2, 0, frac)];
    parent.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
  }
}

export function initScene(viewport, onBlockClickCb) {
  _onBlockClick = onBlockClickCb;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.FogExp2(0x0d1117, 0.008);

  camera = new THREE.PerspectiveCamera(50, 1, 0.1, 300);
  camera.position.set(8, 12, 14);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  viewport.appendChild(renderer.domElement);

  labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.left = '0';
  labelRenderer.domElement.style.pointerEvents = 'none';
  viewport.appendChild(labelRenderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 3;
  controls.maxDistance = 120;

  scene.add(new THREE.AmbientLight(0x404060, 0.6));
  dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(8, 20, 10);
  dirLight.castShadow = true;
  scene.add(dirLight);
  const fill = new THREE.DirectionalLight(0x8888ff, 0.3);
  fill.position.set(-6, 5, -8);
  scene.add(fill);

  gridHelper = new THREE.GridHelper(60, 60, 0x1a2030, 0x1a2030);
  gridHelper.position.y = -0.5;
  scene.add(gridHelper);

  modelGroup = new THREE.Group();
  scene.add(modelGroup);

  viewport.addEventListener('mousemove', e => {
    const r = viewport.getBoundingClientRect();
    mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  });

  viewport.addEventListener('click', () => {
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(clickableObjects, false);
    if (hits.length) {
      const d = objectData.get(hits[0].object);
      if (d && _onBlockClick) _onBlockClick(d);
    }
  });

  resizeScene(viewport);
}

export function resizeScene(viewport) {
  if (!renderer) return;
  const w = viewport.clientWidth, h = viewport.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
}

export function clearModel() {
  const labelDom = labelRenderer.domElement;
  while (labelDom.firstChild) labelDom.removeChild(labelDom.firstChild);
  while (modelGroup.children.length) {
    const c = modelGroup.children[0];
    modelGroup.remove(c);
    c.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    });
  }
  clickableObjects.length = 0;
  objectData.clear();
  originalColors = new Map();
  hoveredObj = null;
}

function updateHover(viewport) {
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(clickableObjects, false);
  if (hoveredObj && (!hits.length || hits[0].object !== hoveredObj)) {
    hoveredObj.material.emissive.setHex(0x000000);
    hoveredObj = null;
    viewport.style.cursor = 'default';
  }
  if (hits.length) {
    const obj = hits[0].object;
    if (obj !== hoveredObj) {
      hoveredObj = obj;
      obj.material.emissive.setHex(0x333333);
      viewport.style.cursor = 'pointer';
    }
  }
}

export function startLoop(viewport) {
  if (_animId) return;
  function loop() {
    _animId = requestAnimationFrame(loop);
    controls.update();
    updateHover(viewport);
    updateParticles();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }
  loop();
}

export function stopLoop() {
  if (_animId) { cancelAnimationFrame(_animId); _animId = null; }
}

export function setFlowWaypoints(wps) {
  flowWaypoints = wps;
  resetParticles();
}

export function setParticlesEnabled(on) {
  particlesEnabled = on;
  if (!on && particleGroup) {
    particleGroup.visible = false;
  } else if (on && particleGroup) {
    particleGroup.visible = true;
  }
}

function resetParticles() {
  if (particleGroup) {
    scene.remove(particleGroup);
    particleGroup.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
  }
  particleGroup = new THREE.Group();
  particleGroup.visible = particlesEnabled;
  scene.add(particleGroup);
  particles = [];
  if (flowWaypoints.length < 2) return;
  const totalDist = flowWaypoints[flowWaypoints.length - 1].y - flowWaypoints[0].y;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const geo = new THREE.SphereGeometry(0.08, 8, 8);
    const mat = new THREE.MeshBasicMaterial({ color: 0xf0c674 });
    const mesh = new THREE.Mesh(geo, mat);
    const t = i / PARTICLE_COUNT;
    mesh.userData = { t, speed: PARTICLE_SPEED * (0.8 + Math.random() * 0.4) };
    particleGroup.add(mesh);
    particles.push(mesh);
  }
}

function updateParticles() {
  if (!particlesEnabled || !particleGroup || flowWaypoints.length < 2) return;
  const wps = flowWaypoints;
  const minY = wps[0].y, maxY = wps[wps.length - 1].y;
  const range = maxY - minY;
  if (range <= 0) return;

  for (const p of particles) {
    p.userData.t += p.userData.speed / range;
    if (p.userData.t > 1) p.userData.t -= 1;
    const targetY = minY + p.userData.t * range;
    // Find surrounding waypoints for color interpolation
    let wpIdx = 0;
    for (let i = 0; i < wps.length - 1; i++) {
      if (wps[i + 1].y >= targetY) { wpIdx = i; break; }
      wpIdx = i;
    }
    const wp = wps[wpIdx];
    const wpNext = wps[Math.min(wpIdx + 1, wps.length - 1)];
    const localT = (wpNext.y - wp.y) > 0 ? (targetY - wp.y) / (wpNext.y - wp.y) : 0;
    const c1 = new THREE.Color(wp.color);
    const c2 = new THREE.Color(wpNext.color);
    p.material.color.copy(c1).lerp(c2, localT);
    p.material.emissive = p.material.color.clone().multiplyScalar(0.5);
    const x = (wp.x || 0) + ((wpNext.x || 0) - (wp.x || 0)) * localT;
    const z = (wp.z || 0) + ((wpNext.z || 0) - (wp.z || 0)) * localT;
    p.position.set(x, targetY, z);
    const pulse = 0.08 + 0.03 * Math.sin(Date.now() * 0.005 + p.userData.t * 20);
    p.scale.setScalar(pulse / 0.08);
  }
}

export function getSceneRefs() {
  return { scene, camera, controls, modelGroup, THREE };
}
