import loadMujoco from "./vendor/mujoco.js";
import * as ort from "./vendor/ort.wasm.min.mjs";

const asset = (name) => new URL(`./assets/${name}`, import.meta.url).href;
async function fetchOK(url) {
  const r = await fetch(url);
  if (!r.ok) throw Error(`无法加载资源 (${r.status})`);
  return r;
}
export function observation(q, v, previous, q0) {
  const o = new Float32Array(75);
  for (let j = 0; j < 23; j++) {
    o[j] = q[7 + j] - q0[j];
    o[23 + j] = v[6 + j];
    o[52 + j] = previous[j];
  }
  o.set(v.slice(3, 6), 46);
  const [w, x, y, z] = q.slice(3, 7);
  o[49] = 2 * (w * y - x * z);
  o[50] = -2 * (y * z + w * x);
  o[51] = 2 * (x * x + y * y) - 1;
  return o;
}
export function flattenHistory(history) {
  const result = new Float32Array(750);
  let src = 0,
    dst = 0;
  for (const width of [23, 23, 3, 3, 23]) {
    for (const row of history) {
      result.set(row.subarray(src, src + width), dst);
      dst += width;
    }
    src += width;
  }
  return result;
}
export class Simulation {
  async init(progress = () => {}) {
    progress("加载机器人与物理引擎");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
    this.catalog = await (await fetchOK(asset("catalog.json"))).json();
    this.mj = await loadMujoco({
      locateFile: (name) => new URL(`./vendor/${name}`, import.meta.url).href,
    });
    this.model = this.mj.MjModel.from_xml_string(
      await (await fetchOK(asset("physics.xml"))).text(),
    );
    this.data = new this.mj.MjData(this.model);
    if (
      this.model.nq !== 30 ||
      this.model.nu !== 23 ||
      Math.abs(this.model.opt.timestep - 0.005) > 1e-9
    )
      throw Error("机器人模型接口不匹配");
    this.sessions = new Map();
    this.control = this.catalog.control;
    this.progress = progress;
    await this.select(this.catalog.motions[0].id, this.catalog.policies[0].id);
    return this.catalog;
  }
  async select(motionId, policyId) {
    this.motion = this.catalog.motions.find((m) => m.id === motionId);
    const p = this.catalog.policies.find((p) => p.id === policyId);
    if (!this.motion || !p) throw Error("动作或策略不存在");
    if (!this.sessions.has(policyId)) {
      this.progress(`加载 ${p.label} 策略`);
      const session = await ort.InferenceSession.create(asset(p.file), {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      const latents = new Float32Array(
        await (await fetchOK(asset(p.latents))).arrayBuffer(),
      );
      this.sessions.set(policyId, { session, latents });
    }
    this.policyId = policyId;
    this.active = this.sessions.get(policyId);
    this.reset();
  }
  reset() {
    this.mj.mj_resetData(this.model, this.data);
    this.data.qpos.set(this.motion.initial.slice(0, 30));
    this.data.qvel.set(this.motion.initial.slice(30));
    this.previous = new Float32Array(23);
    this.frame = 0;
    this.done = false;
    this.reason = "";
    this.mj.mj_forward(this.model, this.data);
    const o = observation(
      this.data.qpos,
      this.data.qvel,
      this.previous,
      this.control.q0,
    );
    this.history = Array.from({ length: 10 }, () => o.slice());
    this.action = this.previous.slice();
  }
  async infer(history, latent) {
    const feeds = {
      history: new ort.Tensor("float32", history, [1, 750]),
      latent: new ort.Tensor("float32", latent, [1, 32]),
    };
    const result = await this.active.session.run(feeds);
    const a = Float32Array.from(result.action.data);
    for (const t of Object.values(feeds)) t.dispose();
    for (const t of Object.values(result)) t.dispose();
    return a;
  }
  async step() {
    if (this.done) return this.snapshot();
    const start = performance.now(),
      n = (this.motion.offset + this.frame) * 32;
    const a = await this.infer(
      flattenHistory(this.history),
      this.active.latents.slice(n, n + 32),
    );
    if (!a.every(Number.isFinite))
      throw Error("策略输出出现非有限值，仿真已停止");
    const c = this.control;
    for (let j = 0; j < 23; j++) {
      const k = c.actuator_action_index[j];
      this.data.ctrl[j] = c.offset[k] + c.scale[k] * a[k];
    }
    for (let j = 0; j < c.substeps; j++) this.mj.mj_step(this.model, this.data);
    this.frame++;
    this.action = a;
    this.previous = a;
    const q = this.data.qpos,
      v = this.data.qvel;
    if (!q.every(Number.isFinite) || !v.every(Number.isFinite))
      throw Error("物理状态出现非有限值，仿真已停止");
    const o = observation(q, v, a, c.q0);
    this.history.shift();
    this.history.push(o);
    const height = Math.abs(q[2] - this.motion.height[this.frame]);
    const tilt = Math.abs(o[51] - this.motion.gravity_z[this.frame]);
    if (height > 0.25 || tilt > 0.8) {
      this.done = true;
      this.reason = "本次执行触发终止";
    } else if (this.frame >= this.motion.frames - 1) {
      this.done = true;
      this.reason = "完成动作";
    }
    this.lastStepMs = performance.now() - start;
    // Recompute kinematics at the new state; rendering never substitutes a reference pose.
    this.mj.mj_forward(this.model, this.data);
    return this.snapshot();
  }
  push() {
    this.data.qvel[0] += 0.2;
    this.data.qvel[1] += 0.2;
  }
  snapshot() {
    return {
      frame: this.frame,
      frames: this.motion.frames,
      done: this.done,
      reason: this.reason,
      policy: this.policyId,
      motion: this.motion.id,
      stepMs: this.lastStepMs || 0,
      qpos: Array.from(this.data.qpos),
      action: Array.from(this.action),
      positions: Float32Array.from(this.data.xpos),
      quaternions: Float32Array.from(this.data.xquat),
    };
  }
  async dispose() {
    this.data?.delete();
    this.model?.delete();
    for (const v of this.sessions?.values() || []) await v.session.release();
  }
}
