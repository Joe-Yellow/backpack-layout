// Runs the WebAssembly solver off the main thread. rayon spawns its own nested workers from here,
// and blocks while waiting on them, which browsers only allow inside a worker.
// Messages: {type: "init", threads?} warms up the engine; {type: "solve", problem} solves.
// The engine (module + thread pool) is set up once and reused for every solve.
import init, { initThreadPool, solve } from "./pkg/backpack_solver.js";

let ready = null;
let wasm = null;

async function setup(threads) {
  if (!self.crossOriginIsolated) {
    throw new Error("页面未启用跨源隔离（COOP/COEP 响应头），浏览器不允许多线程计算");
  }
  wasm = await init();
  await initThreadPool(threads || navigator.hardwareConcurrency || 4);
}

self.onmessage = async ({ data }) => {
  try {
    ready ??= setup(data.threads);
    await ready;
    if (data.type !== "solve") return;
    const json = solve(JSON.stringify(data.problem), (round, pos, cells) => {
      self.postMessage({ type: "progress", round, pos, cells });
    });
    self.postMessage({ type: "done", result: JSON.parse(json), memoryBytes: wasm.memory.buffer.byteLength });
  } catch (err) {
    // a failed warm-up stays in `ready` and is reported by the next solve
    if (data.type === "solve") self.postMessage({ type: "error", message: String(err?.message ?? err) });
  }
};
