# 背包排布问题 · Backpack Layout

在带禁用格的小棋盘上摆放不规则形状的物品和不限数量的 1×2 特殊矩形，求**全局最优**的摆法，并**证明**不存在价值更高的合法方案。

**在线使用：<https://packsolver.site>**

计算全部在浏览器本地完成（Rust 编译成 WebAssembly，多线程），数据不会上传。

## 规则

- 棋盘 W×H，部分格子禁用。物品可旋转、镜像，不能重叠、不能越界、不能占禁用格；每个物品最多用一次。
- 内置 6 种形状：`U` 凹形、`F`、`L`、`T`、`O` 方形、`V` 小 L。
- **边界奖励**：物品接触指定的那条棋盘边（上/下/左/右），基础价值 +20%。
- **邻接奖励**：开启后，每与一个不同的特殊矩形相邻（上下左右，对角不算），基础价值 +10%。
- 一个物品的边界奖励与邻接奖励**先合并再向下取整**：`价值 + ⌊价值 × (2·贴边 + 相邻矩形数) / 10⌋`。
- **特殊矩形**（1×2，可横可竖，不限数量，本身无价值）：相邻普通物品的基础价值**先求和**，再取 20% 向下取整。
- 目标：所有物品与特殊矩形的价值总和最大。

## 算法

1. **逐格扫描的动态规划**：沿棋盘短边逐格决定"留空 / 放特殊矩形 / 放某物品的某姿态"，每种布局只生成一次。邻接关系出现时立即按增量取整计分（`⌊(a+b)/q⌋ = ⌊a/q⌋ + ⌊(a mod q + b)/q⌋`），状态里只保留余数。
2. **支配剪枝**：同一几何局面下，用过的物品更少、未结算实体价值不更低、得分足够高的状态支配其他状态（Pareto 前沿）。
3. **懒约束**：先允许物品重复使用求松弛上界，只对实际被超量使用的物品加"最多一次"的限制再求解，直到松弛最优解本身合法——它就是可证明的全局最优。
4. **并行**：状态按键哈希分 64 片，每层分批用 rayon 并行展开、按分片无锁插入。

正确性由独立校验器（按规则直接重算总分）和小棋盘上与暴力穷举的随机对拍保证。

## 使用

### 网页版（推荐）

```bash
cd web
node serve.mjs          # 打开 http://localhost:8080
```

网页需要 COOP/COEP 响应头才能启用 WASM 多线程，所以请用自带的 `serve.mjs`，不要直接双击打开 `index.html`。
部署到 Cloudflare Pages / Netlify 时 `_headers` 会自动生效；其他平台（如腾讯云 COS + CDN）需要在 CDN 上配置：

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

且必须使用 HTTPS。网址后加 `?threads=N` 可以限制使用的 CPU 核数。

### 重新编译 WASM

需要 Rust nightly（含 `rust-src`、`wasm32-unknown-unknown` 目标）以及与 `solver-rs/Cargo.toml` 中版本一致的 `wasm-bindgen-cli`：

```bash
cd web
node build-wasm.mjs     # 输出到 web/pkg
```

### 原生命令行（测速 / 测试）

```bash
cd solver-rs
cargo test --release
cargo run --release --bin bench -- ../examples/shapes20_1.json
```

## 输入格式

```json
{
  "width": 7, "height": 5,
  "blocked": [[0,0],[6,0],[0,4],[6,4]],
  "items": [
    {"name": "凹", "value": 56, "type": "U", "adjBonus": true},
    {"name": "F1", "value": 45, "type": "F", "border": "top"}
  ]
}
```

坐标为 `[x, y]`，原点在左上角。物品用 `type`（内置形状）或 `shape`（格子坐标列表）二选一。

## 目录

| 目录 | 内容 |
|---|---|
| `web/` | 静态网站：页面、Web Worker、编译好的 WASM（`pkg/`） |
| `solver-rs/` | Rust 求解器（原生 + WebAssembly）、测试、测速程序 |
| `examples/` | 测试实例 |
| `src/`、`pom.xml` | 早期的 Java 实现（参考用） |

## 许可证

[MIT](LICENSE)
