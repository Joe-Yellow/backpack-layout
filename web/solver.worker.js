// Runs the WebAssembly solver off the main thread. rayon spawns its own nested workers from here,
// and blocks while waiting on them, which browsers only allow inside a worker.
import init, { initThreadPool, solve } from "./pkg/backpack_solver.js";

let ready = null;

async function setup(threads) {
  if (!self.crossOriginIsolated) {
    throw new Error("页面未启用跨源隔离（COOP/COEP 响应头），浏览器不允许多线程计算");
  }
  await init();
  await initThreadPool(threads || navigator.hardwareConcurrency || 4);
}

self.onmessage = async (event) => {
  try {
    ready ??= setup(event.data.threads);
    await ready;
    const t0 = performance.now();
    const json = solve(JSON.stringify(event.data.problem), (round, pos, cells) => {
      self.postMessage({ type: "progress", round, pos, cells });
    });
    const result = JSON.parse(json);
    result.seconds = (performance.now() - t0) / 1000;
    self.postMessage({ type: "done", result });
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message ?? err) });
  }
};
