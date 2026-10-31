const $ = (id) => document.getElementById(id);
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
let catalog = null,
  motion = "walk",
  wmMotion = "jog",
  wmMode = "on",
  wmChanging = false;
const hero = $("hero-video"),
  mv = $("motion-video"),
  wv = $("wm-video");
const sources = new WeakMap();
const descriptions = {
  on: "每一帧从当前仿真状态重新预测。四个蓝色姿态依次对应未来 0.16、0.32、0.48、0.64 秒。",
  hold: "2.00 秒后保持最后一个策略动作，PD 仍在工作。世界模型也保持该动作；这不是关闭电机，也不冒充官方 Policy off 的底层实现。",
  slip: "2.00 秒后足部摩擦降至 0.3，世界模型同步接收实际摩擦。干预前的预测不会预先知道未来的摩擦变化。",
};
function clip(slug, mode = "on", kind = "actual") {
  return catalog?.clips.find(
    (c) => c.slug === slug && c.mode === mode && c.kind === kind,
  );
}
function error() {
  $("load-error").hidden = false;
}
async function play(v) {
  try {
    await v.play();
  } catch (e) {
    if (e.name !== "AbortError" && e.name !== "NotAllowedError") error();
  }
}
function stopOthers(except) {
  for (const v of [hero, mv, wv]) if (!except.includes(v)) v.pause();
}
function playGroup(videos) {
  stopOthers(videos);
  videos.forEach(play);
}
function toggle(v, group = [v]) {
  if (v.paused) {
    if (v.ended) group.forEach((x) => (x.currentTime = 0));
    playGroup(group);
  } else group.forEach((x) => x.pause());
}
function bindButton(v, id, group = [v]) {
  $(id).addEventListener("click", () => toggle(v, group));
  v.addEventListener("play", () => {
    $(id).textContent = "Ⅱ";
    $(id).setAttribute("aria-label", "暂停视频");
  });
  v.addEventListener("pause", () => {
    $(id).textContent = "▶";
    $(id).setAttribute("aria-label", "播放视频");
  });
}
bindButton(hero, "hero-play");
bindButton(mv, "motion-play");
bindButton(wv, "wm-play");
function still(v) {
  if (v.readyState < 2) return null;
  try {
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    return c.toDataURL("image/jpeg", 0.85);
  } catch {
    return null;
  }
}
async function swapVideo(v, file, poster, time = 0, resume = false) {
  const token = (sources.get(v) || 0) + 1;
  sources.set(v, token);
  const snapshot = still(v);
  v.pause();
  v.poster = snapshot || `media/${poster}`;
  await new Promise((resolve, reject) => {
    const controller = new AbortController(),
      signal = controller.signal;
    const timer = setTimeout(() => finish(new Error("media timeout")), 45000);
    function finish(e) {
      clearTimeout(timer);
      controller.abort();
      e ? reject(e) : resolve();
    }
    v.addEventListener("error", () => finish(new Error("media load failed")), {
      signal,
    });
    v.addEventListener(
      "canplaythrough",
      () => {
        if (sources.get(v) !== token) {
          finish();
          return;
        }
        const target = Math.max(0, Math.min(time, v.duration - 0.02));
        if (target === 0) {
          finish();
          return;
        }
        v.addEventListener(
          "seeked",
          () => {
            if (Math.abs(v.currentTime - target) > 0.025)
              finish(new Error("Could not seek to requested frame"));
            else finish();
          },
          { signal, once: true },
        );
        v.currentTime = target;
      },
      { signal, once: true },
    );
    v.preload = "auto";
    v.src = `media/${file}`;
    v.load();
  });
  if (sources.get(v) !== token) return;
  v.poster = `media/${poster}`;
  if (resume) await play(v);
}
function press(selector, attribute, value) {
  document
    .querySelectorAll(selector)
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset[attribute] === value)),
    );
}
function seconds(v) {
  return Number.isFinite(v) ? v.toFixed(2) : "0.00";
}
function duration(c) {
  return (c.reference_frames - 1) / 50;
}
function stateText(c, t) {
  return c.failed && t >= c.steps / 50 ? "已终止 · 冻结失败状态" : "";
}
let motionVersion = 0;
async function chooseMotion(slug) {
  if (!catalog) return;
  const version = ++motionVersion;
  const resume = !mv.paused;
  motion = slug;
  press("[data-motion]", "motion", slug);
  const c = clip(slug);
  $("motion-label").textContent = c.title;
  mv.setAttribute("aria-label", `${c.title}完整仿真片段`);
  $("motion-time").max = duration(c);
  $("motion-caption").textContent =
    `${c.name} · 80k · 从首帧执行 ${seconds(c.steps / 50)} 秒，${c.failed ? "触发终止" : "完成片段"}。`;
  await swapVideo(mv, c.file, c.poster, 0, false);
  if (version === motionVersion) {
    mv.playbackRate = Number($("motion-speed").value);
    if (resume) playGroup([mv]);
  }
}
document
  .querySelectorAll("[data-motion]")
  .forEach((b) =>
    b.addEventListener("click", () =>
      chooseMotion(b.dataset.motion).catch(error),
    ),
  );
$("motion-speed").addEventListener(
  "change",
  () => (mv.playbackRate = Number($("motion-speed").value)),
);
$("motion-time").addEventListener("input", (e) => {
  mv.pause();
  mv.currentTime = Number(e.target.value);
});
let wmVersion = 0;
async function chooseWorld() {
  if (!catalog) return;
  const version = ++wmVersion;
  const t = wv.currentTime;
  const resume = !wv.paused;
  wmChanging = true;
  const c = clip(wmMotion, wmMode, "future");
  press("[data-mode]", "mode", wmMode);
  $("wm-description").textContent = descriptions[wmMode];
  $("wm-title").textContent = c.title;
  $("wm-time").max = duration(c);
  $("wm-end").textContent = seconds(duration(c)) + " s";
  $("wm-event").hidden = wmMode === "on";
  $("wm-event").style.left = `${(100 * 2) / duration(c)}%`;
  $("wm-event-label").textContent =
    wmMode === "hold" ? "开始保持动作" : "摩擦 0.3";
  wv.setAttribute(
    "aria-label",
    `${c.title}，${wmMode === "on" ? "策略执行" : wmMode === "hold" ? "保持动作" : "低摩擦"}模式`,
  );
  $("wm-status").textContent = "正在载入";
  await swapVideo(wv, c.file, c.poster, Math.min(t, duration(c)), false);
  if (version === wmVersion) {
    wmChanging = false;
    if (resume) playGroup([wv]);
  }
}
document.querySelectorAll("[data-mode]").forEach((b) =>
  b.addEventListener("click", () => {
    wmMode = b.dataset.mode;
    chooseWorld().catch(error);
  }),
);
$("wm-motion").addEventListener("change", (e) => {
  wmMotion = e.target.value;
  chooseWorld().catch(error);
});
$("wm-time").addEventListener("input", (e) => {
  wv.pause();
  wv.currentTime = Number(e.target.value);
});
$("wm-step").addEventListener("click", () => {
  wv.pause();
  wv.currentTime = Math.min(
    Math.round(wv.currentTime * 50 + 1) / 50,
    Number($("wm-time").max),
  );
});
function animate() {
  if (catalog) {
    const m = clip(motion),
      w = clip(wmMotion, wmMode, "future");
    const mt = Math.min(mv.currentTime, duration(m));
    $("motion-time").value = mt;
    $("motion-clock").textContent =
      `${seconds(mt)} / ${seconds(duration(m))} s`;
    $("motion-status").textContent = stateText(m, mt);
    if (!wmChanging) {
      const wt = Math.min(wv.currentTime, duration(w));
      $("wm-time").value = wt;
      $("wm-clock").textContent = seconds(wt) + " s";
      $("wm-frame").textContent =
        `Frame ${Math.min(Math.floor(wt * 50 + 1e-5), w.reference_frames - 1)} / ${w.reference_frames - 1}`;
      $("wm-status").textContent =
        stateText(w, wt) ||
        (wt >= 2
          ? wmMode === "slip"
            ? "当前摩擦 0.3"
            : wmMode === "hold"
              ? "动作已保持 · PD 工作"
              : ""
          : "");
    }
  }
  requestAnimationFrame(animate);
}
fetch("data/showcase.json")
  .then((r) => {
    if (!r.ok) throw Error(r.status);
    return r.json();
  })
  .then(async (data) => {
    catalog = data;
    await Promise.all([chooseMotion("walk"), chooseWorld()]);
    requestAnimationFrame(animate);
  })
  .catch(error);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopOthers([]);
});
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) =>
      entries.forEach((e) => {
        if (!e.isIntersecting) {
          e.target.pause();
        }
      }),
    { threshold: 0.15 },
  );
  [hero, mv, wv].forEach((v) => observer.observe(v));
}
if (!reduced.matches) play(hero);
document.querySelectorAll("[role=group]").forEach((group) =>
  group.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const buttons = [...group.querySelectorAll("button")],
      i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    e.preventDefault();
    const next =
      buttons[
        (i + (e.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length
      ];
    next.focus();
    next.click();
  }),
);
let checkpoints = [];
const groups = [
  "push_up",
  "crawl",
  "inchworm",
  "lying_recovery",
  "flip",
  "side_roll",
];
function drawChart(selected) {
  const svg = $("return-chart");
  const x = (it) => 52 + ((it - 80000) / 76000) * 610,
    y = (v) => 249 - v * 210;
  let out = "";
  for (const v of [0, 0.25, 0.5, 0.75, 1])
    out += `<line x1="52" x2="662" y1="${y(v)}" y2="${y(v)}" stroke="#e8e8ed"/><text x="38" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="#6e6e73">${v.toFixed(2)}</text>`;
  out += `<polyline points="${checkpoints.map((r) => `${x(r.iteration)},${y(r.groups.uniform.mean_norm_return)}`).join(" ")}" fill="none" stroke="#0071e3" stroke-width="2.5"/>`;
  for (const r of checkpoints) {
    const active = r.iteration === selected;
    out += `<circle cx="${x(r.iteration)}" cy="${y(r.groups.uniform.mean_norm_return)}" r="${active ? 7 : 4}" fill="${active ? "#0071e3" : "#fff"}" stroke="#0071e3" stroke-width="2"><title>${r.iteration / 1000}k: ${r.groups.uniform.mean_norm_return.toFixed(4)}</title></circle><text x="${x(r.iteration)}" y="274" text-anchor="middle" font-size="11" fill="${active ? "#0071e3" : "#6e6e73"}">${r.iteration / 1000}k</text>`;
  }
  svg.innerHTML = out;
}
function updateResults() {
  const iteration = Number($("checkpoint").value),
    r = checkpoints.find((v) => v.iteration === iteration);
  if (!r) return;
  const g = r.groups;
  $("selected-label").textContent = `${iteration / 1000}K CHECKPOINT`;
  $("metric-return").textContent = g.uniform.mean_norm_return.toFixed(4);
  $("metric-uniform").textContent = `${g.uniform.failed} / 2,000`;
  $("metric-challenge").textContent = `${g.challenge_union.failed} / 462`;
  $("table-checkpoint").textContent = `${iteration / 1000}k`;
  $("group-row").replaceChildren(
    ...groups.map((k) => {
      const td = document.createElement("td");
      td.textContent = `${g[k].failed} / ${g[k].clips}`;
      if (g[k].failed === g[k].clips) td.className = "bad";
      return td;
    }),
  );
  $("metric-note").textContent =
    iteration === 80000
      ? "基准 checkpoint：当前所评测主样本回报最高。"
      : iteration === 104000
        ? "困难动作失败数最少；主样本回报仍低于 80k。"
        : `主样本平均回报较 80k ${((g.uniform.mean_norm_return - checkpoints[0].groups.uniform.mean_norm_return) * 100).toFixed(2)} 个百分点。`;
  drawChart(iteration);
}
$("checkpoint").addEventListener("change", updateResults);
fetch("data/checkpoints.json")
  .then((r) => {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  })
  .then((data) => {
    checkpoints = data;
    updateResults();
  })
  .catch(() => {
    error();
    $("checkpoint").disabled = true;
  });

// The live engine is loaded only after an explicit start; removing the iframe frees its workers.
let liveFrame = null;
$("live-start").onclick = () => {
  if (liveFrame) return;
  stopOthers([]);
  liveFrame = document.createElement("iframe");
  liveFrame.src = "demo/";
  liveFrame.title = "在浏览器里运行 G1 策略";
  liveFrame.allow = "fullscreen";
  $("live-slot").append(liveFrame);
  $("live-cover").hidden = true;
  $("live-stop").hidden = false;
};
$("live-stop").onclick = () => {
  liveFrame?.remove();
  liveFrame = null;
  $("live-cover").hidden = false;
  $("live-stop").hidden = true;
};
new IntersectionObserver(
  (entries) => {
    if (!entries[0].isIntersecting)
      liveFrame?.contentWindow?.postMessage(
        { type: "pause-demo" },
        location.origin,
      );
  },
  { threshold: 0.1 },
).observe($("live-slot"));
