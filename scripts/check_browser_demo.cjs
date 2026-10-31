/* Optional integration QA: NODE_PATH must resolve an installed Playwright package. */
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      reducedMotion: "reduce",
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("response", (r) => {
      if (r.status() >= 400) errors.push(r.status() + " " + r.url());
    });
    const base = process.env.DEMO_URL || "http://127.0.0.1:8767/";
    await page.goto(base, { waitUntil: "networkidle" });
    await page.waitForFunction(() => catalog && !wmChanging);
    assert.equal(
      await page
        .locator('#failures, [data-motion="crawl"], [data-motion="pushup"]')
        .count(),
      0,
    );
    assert.equal(await page.locator("#live-slot iframe").count(), 0);
    await page.click('[data-motion="jump"]');
    await page.waitForFunction(() =>
      document
        .querySelector("#motion-caption")
        .textContent.includes("frog_jump"),
    );
    await page.locator("#wm-time").evaluate((e) => {
      e.value = "1.2";
      e.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.click('[data-mode="hold"]');
    await page.waitForFunction(() => !wmChanging);
    assert.ok(
      Math.abs(
        (await page.locator("#wm-video").evaluate((v) => v.currentTime)) - 1.2,
      ) < 0.025,
    );
    await page.click("#live-start");
    const frame = page.frameLocator("#live-slot iframe");
    await frame.locator("#play").waitFor();
    const child = await (
      await page.locator("#live-slot iframe").elementHandle()
    ).contentFrame();
    await child.waitForFunction(() => window.sm23, { timeout: 120000 });
    await frame.locator("#category").selectOption("frog");
    await frame.locator("#play").waitFor({ state: "visible" });
    await child.waitForFunction(
      () =>
        sm23.state.motion === "frog" &&
        !document.querySelector("#play").disabled,
    );
    await frame.locator("#policy").selectOption("104000");
    await child.waitForFunction(
      () =>
        sm23.state.policy === "104000" &&
        !document.querySelector("#play").disabled,
    );
    await frame.locator("#play").click();
    await child.waitForFunction(() => sm23.state.frame >= 10);
    await frame.locator("#play").click();
    await child.waitForFunction(
      () => document.querySelector("#play").textContent === "运行策略",
    );
    const paused = await child.evaluate(() => sm23.state.frame);
    await page.waitForTimeout(150);
    assert.equal(await child.evaluate(() => sm23.state.frame), paused);
    await frame.locator("#reset").click();
    await child.waitForFunction(() => sm23.state.frame === 0);
    await frame.locator('[data-view="front"]').click();
    assert.equal(
      await frame.locator('[data-view="front"]').getAttribute("aria-pressed"),
      "true",
    );
    // Perturbation must change the trajectory: the scene is simulated, not replayed.
    const normal = await child.evaluate(() => sm23.rpc("steps", { count: 5 }));
    await child.evaluate(() => sm23.rpc("reset"));
    await frame.locator("#push").click();
    const pushed = await child.evaluate(() => sm23.rpc("steps", { count: 5 }));
    assert.ok(Math.abs(normal.qpos[0] - pushed.qpos[0]) > 0.001);
    if (process.env.SCREENSHOT_DIR) {
      await page.screenshot({
        path: process.env.SCREENSHOT_DIR + "/showcase-desktop.png",
        fullPage: true,
      });
    }
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      assert.ok(
        await child.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
    }
    if (process.env.SCREENSHOT_DIR) {
      await page.setViewportSize({ width: 390, height: 844 });
      await page.screenshot({
        path: process.env.SCREENSHOT_DIR + "/showcase-mobile.png",
        fullPage: true,
      });
    }
    await page.click("#live-stop");
    assert.equal(await page.locator("#live-slot iframe").count(), 0);
    assert.deepEqual(errors, []);
    console.log(
      "PASS gallery, synchronous modes, lazy live demo, policy/motion choices, pause/reset, perturbation, views, responsive layout and teardown",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
