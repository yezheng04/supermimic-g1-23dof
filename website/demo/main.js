import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});
const pending = new Map();
let nextId = 0,
  catalog = null,
  state = null,
  running = false,
  ready = false;
const categories = {
  walk: "行走",
  frog: "蛙跳",
  jog: "慢跑",
  dance: "街舞",
  kick: "踢腿",
  punch: "出拳",
};
function rpc(type, args = {}) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(Error("加载超时，请检查网络后重试"));
    }, 120000);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ type, id, ...args });
  });
}
function failure(error) {
  worker.terminate();
  ready = false;
  running = false;
  $("loading").hidden = true;
  $("error").hidden = false;
  $("error-text").textContent = error.message || String(error);
  for (const p of pending.values()) {
    clearTimeout(p.timer);
    p.reject(error);
  }
  pending.clear();
  lock(true);
}
worker.onmessage = ({ data: m }) => {
  if (m.type === "reply") {
    const p = pending.get(m.id);
    if (p) {
      clearTimeout(p.timer);
      p.resolve(m.value);
      pending.delete(m.id);
    }
  } else if (m.type === "progress") {
    $("loading-text").textContent = m.message;
  } else if (m.type === "error") failure(Error(m.message));
  else if (m.type === "state") {
    state = m.state;
    running = m.running;
    paintState();
  }
};
worker.onerror = (e) =>
  failure(
    Error(e.message || "浏览器无法初始化仿真，请尝试新版 Chrome 或 Edge"),
  );
$("retry").onclick = () => location.reload();
function lock(value) {
  for (const id of ["category", "motion", "policy", "play", "reset", "push"])
    $(id).disabled = value;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color("#f5f5f7");
scene.fog = new THREE.Fog("#f5f5f7", 9, 22);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
} catch (e) {
  failure(
    Error(
      "此浏览器无法创建 WebGL 画面，请启用硬件加速或使用新版 Chrome / Edge",
    ),
  );
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
$("viewport").append(renderer.domElement);
const camera = new THREE.PerspectiveCamera(36, 1, 0.02, 100);
camera.up.set(0, 0, 1);
camera.position.set(2.6, -2, 1.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.65);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 1.3;
controls.maxDistance = 7;
controls.maxPolarAngle = Math.PI * 0.49;
scene.add(new THREE.HemisphereLight(0xffffff, 0xc8ccd5, 1.3));
const key = new THREE.DirectionalLight(0xffffff, 2.4);
key.position.set(3, -4, 7);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, {
  left: -3,
  right: 3,
  top: 3,
  bottom: -3,
  near: 0.1,
  far: 20,
});
key.shadow.bias = -0.0001;
key.shadow.normalBias = 0.015;
scene.add(key, key.target);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(200, 200),
  new THREE.MeshBasicMaterial({ color: 0xf5f5f7, toneMapped: false }),
);
floor.receiveShadow = false;
floor.position.z = -0.002;
scene.add(floor);
const shadow = new THREE.Mesh(
  new THREE.PlaneGeometry(200, 200),
  new THREE.ShadowMaterial({ opacity: 0.16 }),
);
shadow.position.z = -0.0018;
shadow.receiveShadow = true;
scene.add(shadow);
const grid = new THREE.GridHelper(80, 160, 0xd9dbe2, 0xe5e6eb);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.001;
grid.material.transparent = true;
grid.material.opacity = 0.35;
scene.add(grid);
const bodies = [];
let previousRoot = null,
  view = "quarter";
function fitCamera() {
  const box = $("viewport").getBoundingClientRect();
  if (!box.width || !box.height) return;
  renderer.setSize(box.width, box.height);
  camera.aspect = box.width / box.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(fitCamera).observe($("viewport"));
function setView(name) {
  view = name;
  const q = state?.qpos || [0, 0, 0.8, 1, 0, 0, 0];
  const [w, x, y, z] = q.slice(3, 7);
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const angle =
    yaw + { front: 0, quarter: -Math.PI / 4, side: -Math.PI / 2 }[name];
  controls.target.set(q[0], q[1], 0.65);
  const distance = innerWidth < 740 ? 3.25 : 2.9;
  camera.position.set(
    q[0] + distance * Math.cos(angle),
    q[1] + distance * Math.sin(angle),
    1.25,
  );
  controls.update();
  previousRoot = new THREE.Vector3(q[0], q[1], 0.65);
  document
    .querySelectorAll("[data-view]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.view === name)),
    );
}
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => setView(b.dataset.view)));
async function visuals() {
  const [meta, binary] = await Promise.all([
    fetch("./assets/robot.json").then((r) => {
      if (!r.ok) throw Error("无法加载机器人外观");
      return r.json();
    }),
    fetch("./assets/robot.bin").then((r) => r.arrayBuffer()),
  ]);
  const geometries = meta.meshes.map((m) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(binary, ...m.vertices), 3),
    );
    g.setIndex(
      new THREE.BufferAttribute(new Uint32Array(binary, ...m.indices), 1),
    );
    g.computeVertexNormals();
    return g;
  });
  for (let i = 0; i < meta.nbody; i++) {
    const b = new THREE.Group();
    bodies.push(b);
    scene.add(b);
  }
  for (const g of meta.geoms) {
    const color = new THREE.Color()
      .setRGB(...g.color.slice(0, 3))
      .convertSRGBToLinear();
    const mesh = new THREE.Mesh(
      geometries[g.mesh],
      new THREE.MeshStandardMaterial({
        color,
        roughness: 0.55,
        metalness: 0.12,
      }),
    );
    mesh.position.fromArray(g.pos);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    bodies[g.body].add(mesh);
  }
}
function paintState() {
  if (!state) return;
  for (let i = 0; i < bodies.length; i++) {
    bodies[i].position.fromArray(state.positions, i * 3);
    const q = state.quaternions;
    bodies[i].quaternion.set(
      q[i * 4 + 1],
      q[i * 4 + 2],
      q[i * 4 + 3],
      q[i * 4],
    );
  }
  const root = new THREE.Vector3(state.qpos[0], state.qpos[1], 0.65);
  if (previousRoot) {
    const delta = root.clone().sub(previousRoot);
    camera.position.add(delta);
    controls.target.add(delta);
  }
  previousRoot = root;
  key.position.set(root.x + 3, root.y - 4, 7);
  key.target.position.set(root.x, root.y, 0);
  $("clock").textContent =
    `${(state.frame / 50).toFixed(2)} / ${((state.frames - 1) / 50).toFixed(2)} s`;
  $("progress").max = state.frames - 1;
  $("progress").value = state.frame;
  $("state-label").textContent = state.done
    ? state.reason
    : running
      ? "策略正在运行"
      : "已就绪";
  $("play").textContent = running
    ? "暂停"
    : state.done
      ? "再次运行"
      : "运行策略";
  $("performance").textContent = state.stepMs
    ? `${state.stepMs.toFixed(1)} ms / 控制步`
    : "";
}
function options() {
  const cat = $("category").value;
  $("motion").replaceChildren(
    ...catalog.motions
      .filter((m) => m.category === cat)
      .map((m) => new Option(m.title, m.id)),
  );
}
async function select() {
  lock(true);
  $("loading").hidden = false;
  try {
    await rpc("select", {
      motion: $("motion").value,
      policy: $("policy").value,
    });
    const m = catalog.motions.find((m) => m.id === $("motion").value);
    $("clip-description").textContent =
      `${m.name} · ${((m.frames - 1) / 50).toFixed(2)} 秒`;
    setView(view);
    lock(false);
    $("loading").hidden = true;
  } catch (e) {
    failure(e);
  }
}
$("category").onchange = () => {
  options();
  select();
};
$("motion").onchange = select;
$("policy").onchange = select;
$("play").onclick = () => rpc(running ? "pause" : "play").catch(failure);
$("reset").onclick = () =>
  rpc("reset")
    .then(() => setView(view))
    .catch(failure);
$("push").onclick = () => rpc("push").catch(failure);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && ready) rpc("pause").catch(failure);
});
addEventListener("message", (e) => {
  if (e.origin === location.origin && e.data?.type === "pause-demo" && ready)
    rpc("pause").catch(failure);
});
function animate() {
  requestAnimationFrame(animate);
  if (document.hidden) return;
  controls.update();
  renderer.render(scene, camera);
}
fitCamera();
animate();
try {
  const results = await Promise.all([rpc("init"), visuals()]);
  catalog = results[0];
  $("category").replaceChildren(
    ...[...new Set(catalog.motions.map((m) => m.category))].map(
      (c) => new Option(categories[c], c),
    ),
  );
  $("policy").replaceChildren(
    ...catalog.policies.map((p) => new Option(p.label + " checkpoint", p.id)),
  );
  options();
  await select();
  ready = true;
  window.sm23 = {
    rpc,
    get state() {
      return state;
    },
    get catalog() {
      return catalog;
    },
  };
} catch (e) {
  failure(e);
}
