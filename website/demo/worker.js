import { Simulation } from "./simulation.js";
let sim,
  running = false,
  timer = null,
  queue = Promise.resolve();
const postState = () =>
  postMessage({ type: "state", state: sim.snapshot(), running });
function halt() {
  running = false;
  clearTimeout(timer);
  timer = null;
}
function schedule(delay = 0) {
  if (running)
    timer = setTimeout(
      () =>
        enqueue(async () => {
          if (!running) return;
          const start = performance.now();
          await sim.step();
          if (sim.done) running = false;
          postState();
          if (running) schedule(Math.max(0, 20 - (performance.now() - start)));
        }),
      delay,
    );
}
function enqueue(fn) {
  queue = queue.then(fn).catch((e) => {
    halt();
    postMessage({ type: "error", message: e.message });
  });
}
onmessage = ({ data: m }) =>
  enqueue(async () => {
    const reply = (value) => postMessage({ type: "reply", id: m.id, value });
    switch (m.type) {
      case "init":
        sim = new Simulation();
        reply(
          await sim.init((message) =>
            postMessage({ type: "progress", message }),
          ),
        );
        postState();
        break;
      case "select":
        halt();
        await sim.select(m.motion, m.policy);
        postState();
        reply(true);
        break;
      case "reset":
        halt();
        sim.reset();
        postState();
        reply(true);
        break;
      case "play":
        if (!running) {
          if (sim.done) sim.reset();
          running = true;
          schedule();
        }
        reply(true);
        postState();
        break;
      case "pause":
        halt();
        reply(true);
        postState();
        break;
      case "push":
        sim.push();
        reply(true);
        break;
      case "steps":
        halt();
        for (let i = 0; i < (m.count || 1) && !sim.done; i++) await sim.step();
        reply(sim.snapshot());
        postState();
        break;
      case "infer":
        reply(
          Array.from(
            await sim.infer(
              Float32Array.from(m.history),
              Float32Array.from(m.latent),
            ),
          ),
        );
        break;
      case "dispose":
        halt();
        await sim.dispose();
        reply(true);
        close();
        break;
    }
  });
