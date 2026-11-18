// Rebuilds the WebAssembly solver into ./pkg.
// Needs: Rust nightly with rust-src and the wasm32-unknown-unknown target, plus wasm-bindgen-cli
// matching the wasm-bindgen version pinned in solver-rs/Cargo.toml.
// Threads in WASM need atomics + shared memory, which means rebuilding std (-Z build-std, nightly).
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import os from "node:os";

const here = path.dirname(fileURLToPath(import.meta.url));
const crate = path.join(here, "..", "solver-rs");
const flags = [
  "-C", "target-feature=+atomics,+bulk-memory,+mutable-globals",
  // shared, imported memory so rayon's worker threads can all map the same heap (max 4 GiB)
  "-C", "link-arg=--shared-memory",
  "-C", "link-arg=--import-memory",
  "-C", "link-arg=--max-memory=4294967296",
  // thread-local storage hooks wasm-bindgen needs to start each thread
  "-C", "link-arg=--export=__wasm_init_tls",
  "-C", "link-arg=--export=__tls_size",
  "-C", "link-arg=--export=__tls_align",
  "-C", "link-arg=--export=__tls_base",
  // panic locations embed source paths: don't publish the builder's home or checkout directory
  "--remap-path-prefix", `${os.homedir()}=~`,
  "--remap-path-prefix", `${path.join(here, "..")}=.`,
];
// encoded form (0x1f-separated) survives spaces in paths
const env = { ...process.env, CARGO_ENCODED_RUSTFLAGS: flags.join("\x1f") };
delete env.RUSTFLAGS;

execFileSync("cargo", ["build", "--release", "--target", "wasm32-unknown-unknown",
  "-Z", "build-std=panic_abort,std", "--lib"], { cwd: crate, env, stdio: "inherit" });
execFileSync("wasm-bindgen", [
  path.join(crate, "target", "wasm32-unknown-unknown", "release", "backpack_solver.wasm"),
  "--out-dir", path.join(here, "pkg"), "--target", "web",
], { stdio: "inherit" });
