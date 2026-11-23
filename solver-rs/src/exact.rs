//! Exact solver with lazily added "each item at most once" constraints.
//!
//! Each round runs a forward DP over the Sweep in which only tracked items are limited to one use;
//! all other classes may be reused. That is a relaxation, so its optimum is an upper bound on the
//! true optimum. If the best relaxed layout uses every class within its count, it is feasible and
//! therefore optimal. Otherwise the over-used classes become tracked and the round repeats. Tracking
//! every item gives the original problem, so this terminates with a proven optimum.
//!
//! Required items are tracked from the start; targets and required items are checked on complete
//! layouts. A relaxation with no layout meeting them proves the original has none either.
//!
//! Within one layer, partial layouts with the same Sweep key form a Pareto front, kept sorted by the
//! maximized slot descending. A dominates B if, in every slot, A is ahead by at least the most B's
//! open entities can still gain over A's (Sweep::penalty), A used a subset of B's tracked items
//! (every completion of B stays open to A) and exactly the same required items (B's completions
//! don't place the ones B already placed). Target slots whose values are never negative are capped
//! at the target: beyond it nothing changes.
//!
//! Parallelism: fronts live in SHARDS shards chosen by key hash. A layer is expanded in parallel
//! batches; children are bucketed by shard and each shard inserts its bucket on one thread, so no
//! locks are needed. Steps always move pos forward, so children never land in the layer being read.

use crate::problem::{Goal, Problem};
use crate::sweep::{Ents, Key, Score, Sweep, DOMINO, SLOTS};
use crate::verifier;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

const SHARDS: usize = 64;
const SHARD_BITS: u32 = 6;
const NO_PARENT: u32 = u32::MAX;
const FRONTS_PER_BATCH: usize = 1 << 12; // bounds the memory held by not-yet-inserted children
/// WASM linear memory caps at 4 GiB and an allocation failure there kills a worker thread, which
/// hangs the solve; stop cleanly with headroom instead (checked after every batch).
const WASM_MEMORY_BUDGET: u64 = 3 << 30;
pub const OUT_OF_MEMORY: &str =
    "内存不足：这个实例超出了浏览器可用的内存（约 4GB），可以减少物品数量或改用命令行版本";

pub struct Layout {
    /// item_cells[i] = board cells of item i, or None if unused.
    pub item_cells: Vec<Option<Vec<usize>>>,
    /// 1x2 pieces as pairs of board cells.
    pub dominoes: Vec<[usize; 2]>,
}

pub struct Solution {
    /// [gather, expand, insert] seconds summed over rounds (diagnostics)
    pub phases: [f64; 3],
    /// Total of the maximized attribute (or empty cells).
    pub value: i64,
    pub layout: Layout,
    pub rounds: u32,
    pub tracked_items: u32,
    pub nodes: u64,
}

pub enum Outcome {
    Solved(Solution),
    /// No layout places every required item and meets every target (proven).
    Infeasible,
}

/// Search-tree node kept to rebuild the layout: parent id and the step taken at pos.
#[derive(Clone, Copy)]
struct ArenaNode {
    parent: u32,
    pos: u8,
    code: u32,
}

/// Pareto front of one key. Most fronts hold a single entry; keeping it inline avoids a heap
/// allocation per key, and the WASM allocator is a global lock that would serialize the threads.
type Front = SmallVec<[Entry; 1]>;

/// Partial layout waiting in a Pareto front.
struct Entry {
    id: u32,
    acc: Score,
    used: u64,
    ents: Ents,
}

struct ChildRec {
    layer: usize,
    key: Key,
    entry: Entry,
    parent: u32,
    step_pos: u8,
    code: u32,
}

/// Children produced by one worker, plus their indices grouped by destination shard.
struct WorkerOut {
    children: Vec<ChildRec>,
    by_shard: Vec<Vec<u32>>,
}

#[derive(Default)]
struct Shard {
    arena: Vec<ArenaNode>,
    layers: Vec<FxHashMap<Key, Front>>,
    inserted: u64,
}

/// Per-solve settings shared by every round.
struct Ctx<'a> {
    sweep: &'a Sweep,
    /// Target of each slot (slot 0 is maximized: no target).
    targets: [Option<i32>; SLOTS],
    /// Slot cap: the target, where no value is negative so the total can only grow.
    caps: [i32; SLOTS],
}

/// Solves exactly. Fails only when the browser memory budget is exceeded.
///
/// Targets first get a cheap pass without them: if that optimum happens to meet every target it is
/// optimal for the constrained problem too (dropping constraints can only raise the optimum).
/// Comparing two scores per state is much slower, so this pays off whenever targets are loose.
pub fn solve(
    p: &Problem,
    on_progress: &mut dyn FnMut(u32, usize, usize),
) -> Result<Outcome, String> {
    let mut rounds_before = 0;
    let mut nodes_before = 0;
    if !p.goal.targets.is_empty() {
        let relaxed = Problem {
            goal: Goal {
                maximize: p.goal.maximize,
                targets: Vec::new(),
            },
            ..p.clone()
        };
        match solve_exact(&relaxed, 0, on_progress)? {
            // only the required items can make the untargeted problem infeasible
            Outcome::Infeasible => return Ok(Outcome::Infeasible),
            Outcome::Solved(sol) => {
                let score = verifier::evaluate(p, &sol.layout)?;
                if score.feasible(p, &sol.layout) {
                    return Ok(Outcome::Solved(sol));
                }
                rounds_before = sol.rounds;
                nodes_before = sol.nodes;
            }
        }
    }
    Ok(match solve_exact(p, rounds_before, on_progress)? {
        Outcome::Solved(mut sol) => {
            sol.rounds += rounds_before;
            sol.nodes += nodes_before;
            Outcome::Solved(sol)
        }
        Outcome::Infeasible => Outcome::Infeasible,
    })
}

/// Rounds are numbered from `round_base + 1` in progress reports.
fn solve_exact(
    p: &Problem,
    round_base: u32,
    on_progress: &mut dyn FnMut(u32, usize, usize),
) -> Result<Outcome, String> {
    let mut slots = vec![p.goal.maximize];
    slots.extend(p.goal.targets.iter().map(|&(a, _)| Some(a)));
    let sweep = Sweep::new(p, &slots);
    let mut targets = [None; SLOTS];
    let mut caps = [i32::MAX; SLOTS];
    for (j, &(attr, t)) in p.goal.targets.iter().enumerate() {
        let t = t.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        targets[j + 1] = Some(t);
        if p.items.iter().all(|it| it.attrs[attr] >= 0) {
            caps[j + 1] = t.max(0);
        }
    }
    let ctx = Ctx {
        sweep: &sweep,
        targets,
        caps,
    };
    let mut tracked = sweep.required;
    let mut total_nodes = 0;
    let mut phases = [0.0; 3];
    for round in 1.. {
        let Some((acc, path, nodes, ph)) = run_round(&ctx, tracked, &mut |pos| {
            on_progress(round_base + round, pos, sweep.n)
        })?
        else {
            return Ok(Outcome::Infeasible);
        };
        total_nodes += nodes;
        for k in 0..3 {
            phases[k] += ph[k];
        }
        let over = over_used(&sweep, &path);
        if over == 0 {
            return Ok(Outcome::Solved(Solution {
                phases,
                value: acc as i64,
                layout: to_layout(&sweep, &path, p.items.len()),
                rounds: round,
                tracked_items: tracked.count_ones(),
                nodes: total_nodes,
            }));
        }
        // Track the over-used classes and everything else this path used: those are the items the
        // next relaxations would otherwise over-use one at a time (fewer, slightly larger rounds).
        tracked |= over | used_classes(&sweep, &path);
    }
    unreachable!()
}

/// Shard of a key. The Fx hash is only 32 bits wide on wasm32, so mix all of it into the top bits
/// (golden-ratio multiply) instead of shifting: a plain `>> 32` sent every key to shard 0 in the
/// browser. Maps index buckets by the low hash bits, so shards stay uncorrelated with buckets.
fn shard_of(key: &[u8]) -> usize {
    let mut h = FxHasher::default();
    key.hash(&mut h);
    (h.finish().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - SHARD_BITS)) as usize
}

/// Result of one round: best maximized total, best path as (pos, code) steps, inserted nodes,
/// phase seconds. None: no complete layout meets the targets and required items.
type RoundResult = Option<(i32, Vec<(usize, u32)>, u64, [f64; 3])>;

fn run_round(
    ctx: &Ctx,
    tracked: u64,
    on_layer: &mut dyn FnMut(usize),
) -> Result<RoundResult, String> {
    let sweep = ctx.sweep;
    let mut ph = [0.0; 3];
    let mut shards: Vec<Shard> = (0..SHARDS)
        .map(|_| Shard {
            layers: (0..=sweep.n).map(|_| FxHashMap::default()).collect(),
            ..Default::default()
        })
        .collect();
    let root = Sweep::pos_of(&sweep.root_key);
    shards[shard_of(&sweep.root_key)].insert(
        ctx,
        &ChildRec {
            layer: root,
            key: sweep.root_key.clone(),
            entry: Entry {
                id: 0,
                acc: [0; SLOTS],
                used: 0,
                ents: Ents::new(),
            },
            parent: NO_PARENT,
            step_pos: 0,
            code: 0,
        },
        true,
    );

    for pos in 0..sweep.n {
        let t0 = now();
        let fronts: Vec<(Key, Front, u32)> = shards
            .iter_mut()
            .enumerate()
            .flat_map(|(s, sh)| {
                std::mem::take(&mut sh.layers[pos])
                    .into_iter()
                    .map(move |(k, f)| (k, f, s as u32))
            })
            .collect();
        ph[0] += now() - t0;
        for batch in fronts.chunks(FRONTS_PER_BATCH) {
            let t1 = now();
            let children = expand_batch(ctx, tracked, pos, batch);
            let t2 = now();
            ph[1] += t2 - t1;
            shards.par_iter_mut().enumerate().for_each(|(s, sh)| {
                for out in &children {
                    for &i in &out.by_shard[s] {
                        sh.insert(ctx, &out.children[i as usize], false);
                    }
                }
            });
            ph[2] += now() - t2;
            drop(children);
            if over_memory_budget() {
                return Err(OUT_OF_MEMORY.to_string());
            }
        }
        on_layer(pos);
    }

    let meets = |e: &Entry| {
        e.used & sweep.required == sweep.required
            && (0..SLOTS).all(|j| ctx.targets[j].is_none_or(|t| e.acc[j] >= t))
    };
    let finals: Vec<(&Entry, usize)> = shards
        .iter()
        .enumerate()
        .flat_map(|(s, sh)| sh.layers[sweep.n].values().flatten().map(move |e| (e, s)))
        .filter(|(e, _)| meets(e))
        .collect();
    let Some(best) = finals.iter().map(|(e, _)| e.acc[0]).max() else {
        return Ok(None);
    };
    let nodes = shards.iter().map(|s| s.inserted).sum();
    // Among tied best layouts prefer one that over-uses nothing: saves a needless extra round.
    let mut pick = None;
    for (e, s) in finals.iter().filter(|(e, _)| e.acc[0] == best) {
        let path = path_of(&shards, (*s as u32) << (32 - SHARD_BITS) | e.id);
        let feasible = over_used(sweep, &path) == 0;
        if pick.is_none() || feasible {
            pick = Some(path);
        }
        if feasible {
            break;
        }
    }
    Ok(Some((best, pick.unwrap(), nodes, ph)))
}

/// Expands a batch of fronts in parallel, in a few contiguous chunks per thread. Children are
/// indexed by destination shard as they are produced, so no sequential pass is needed.
fn expand_batch(
    ctx: &Ctx,
    tracked: u64,
    pos: usize,
    batch: &[(Key, Front, u32)],
) -> Vec<WorkerOut> {
    let sweep = ctx.sweep;
    let chunks = (rayon::current_num_threads() * 4).max(1);
    let chunk_len = batch.len().div_ceil(chunks).max(1);
    batch
        .par_chunks(chunk_len)
        .map(|chunk| {
            let mut out = WorkerOut {
                children: Vec::new(),
                by_shard: (0..SHARDS).map(|_| Vec::new()).collect(),
            };
            let mut sc = sweep.scratch();
            for (key, front, shard) in chunk {
                for e in front {
                    let parent = shard << (32 - SHARD_BITS) | e.id;
                    sweep.expand(key, &e.ents, e.used, tracked, &mut sc, &mut |c| {
                        let s = shard_of(&c.key);
                        push(&mut out.by_shard[s], out.children.len() as u32);
                        push(
                            &mut out.children,
                            ChildRec {
                                layer: Sweep::pos_of(&c.key),
                                entry: Entry {
                                    id: 0,
                                    acc: std::array::from_fn(|j| {
                                        (e.acc[j] + c.gain[j]).min(ctx.caps[j])
                                    }),
                                    used: c.used,
                                    ents: c.ents,
                                },
                                key: c.key,
                                parent,
                                step_pos: pos as u8,
                                code: c.code,
                            },
                        );
                    });
                }
            }
            out
        })
        .collect()
}

/// Vec::push that grows by allocate-copy-free instead of realloc. The WASM allocator holds its
/// global lock for a whole realloc, copy included, which serialized the worker threads; this way
/// the copy happens outside the lock.
fn push<T>(v: &mut Vec<T>, x: T) {
    if v.len() == v.capacity() {
        let mut bigger = Vec::with_capacity((v.capacity() * 2).max(256));
        bigger.append(v);
        *v = bigger;
    }
    v.push(x);
}

impl Shard {
    /// Adds a node unless dominated; removes nodes it dominates. Fronts are sorted by the maximized
    /// slot descending: a dominator never scores lower there, so only the part with acc[0] >= the
    /// node's can dominate it, and only the part with acc[0] <= the node's can be dominated by it.
    fn insert(&mut self, ctx: &Ctx, rec: &ChildRec, root: bool) {
        let layer = &mut self.layers[rec.layer];
        if !layer.contains_key(&rec.key) {
            layer.insert(rec.key.clone(), Front::new());
        }
        let front = layer.get_mut(&rec.key).expect("front just ensured");
        let mut e = Entry {
            id: 0,
            acc: rec.entry.acc,
            used: rec.entry.used,
            ents: rec.entry.ents.clone(),
        };
        let mut at = 0;
        while at < front.len() && front[at].acc[0] >= e.acc[0] {
            if dominates(ctx, &front[at], &e) {
                return;
            }
            at += 1;
        }
        let mut lo = at;
        while lo > 0 && front[lo - 1].acc[0] == e.acc[0] {
            lo -= 1; // equal acc may also be dominated by the new node
        }
        let mut i = front.len();
        while i > lo {
            i -= 1;
            if dominates(ctx, &e, &front[i]) {
                front.remove(i);
                if i < at {
                    at -= 1;
                }
            }
        }
        assert!(
            self.arena.len() < 1 << (32 - SHARD_BITS),
            "search too large"
        );
        e.id = self.arena.len() as u32;
        self.arena.push(ArenaNode {
            parent: if root { NO_PARENT } else { rec.parent },
            pos: rec.step_pos,
            code: rec.code,
        });
        front.insert(at, e);
        self.inserted += 1;
    }
}

/// A dominates B: every completion of B, applied to A, scores at least as much in the maximized
/// slot and meets every target B meets.
fn dominates(ctx: &Ctx, a: &Entry, b: &Entry) -> bool {
    let sweep = ctx.sweep;
    if a.used & !b.used != 0 || (a.used ^ b.used) & sweep.required != 0 {
        return false;
    }
    // per slot: what A is ahead now, minus what B's open entities may still gain over A's
    let mut slack: [i64; SLOTS] = std::array::from_fn(|j| a.acc[j] as i64 - b.acc[j] as i64);
    // a capped slot already meets its target and can only grow: nothing to compare there
    let settled: [bool; SLOTS] = std::array::from_fn(|j| a.acc[j] >= ctx.caps[j]);
    if (0..SLOTS).any(|j| !settled[j] && slack[j] < 0) {
        return false;
    }
    for (x, y) in a.ents.iter().zip(&b.ents) {
        if x == y {
            continue;
        }
        let Some(pen) = sweep.penalty(x, y) else {
            return false;
        };
        for j in 0..SLOTS {
            slack[j] -= pen[j] as i64;
            if !settled[j] && slack[j] < 0 {
                return false;
            }
        }
    }
    true
}

fn path_of(shards: &[Shard], mut id: u32) -> Vec<(usize, u32)> {
    let mut path = Vec::new();
    loop {
        let node = shards[(id >> (32 - SHARD_BITS)) as usize].arena
            [(id & ((1 << (32 - SHARD_BITS)) - 1)) as usize];
        if node.parent == NO_PARENT {
            break;
        }
        path.push((node.pos as usize, node.code));
        id = node.parent;
    }
    path.reverse();
    path
}

/// Item bits of every class used on this path.
fn used_classes(sweep: &Sweep, path: &[(usize, u32)]) -> u64 {
    path.iter()
        .filter_map(|&(pos, code)| sweep.step_cells(pos, code))
        .filter(|(cls, _)| *cls != DOMINO)
        .fold(0, |m, (cls, _)| m | sweep.classes[cls as usize].mask)
}

/// Item bits of every class used more often than it has members on this path.
fn over_used(sweep: &Sweep, path: &[(usize, u32)]) -> u64 {
    let mut count = vec![0usize; sweep.classes.len()];
    for &(pos, code) in path {
        if let Some((cls, _)) = sweep.step_cells(pos, code) {
            if cls != DOMINO {
                count[cls as usize] += 1;
            }
        }
    }
    sweep
        .classes
        .iter()
        .zip(count)
        .filter(|(c, k)| *k > c.members.len())
        .fold(0, |m, (c, _)| m | c.mask)
}

fn to_layout(sweep: &Sweep, path: &[(usize, u32)], item_count: usize) -> Layout {
    let mut item_cells = vec![None; item_count];
    let mut next_member = vec![0usize; sweep.classes.len()];
    let mut dominoes = Vec::new();
    for &(pos, code) in path {
        let Some((cls, cells)) = sweep.step_cells(pos, code) else {
            continue;
        };
        let mut cells: Vec<usize> = cells.into_iter().map(|s| sweep.cell_of_scan(s)).collect();
        cells.sort_unstable();
        if cls == DOMINO {
            dominoes.push([cells[0], cells[1]]);
        } else {
            let c = cls as usize;
            item_cells[sweep.classes[c].members[next_member[c]]] = Some(cells);
            next_member[c] += 1;
        }
    }
    Layout {
        item_cells,
        dominoes,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn now() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn now() -> f64 {
    js_sys::Date::now() / 1000.0
}

#[cfg(target_arch = "wasm32")]
fn over_memory_budget() -> bool {
    // linear memory never shrinks, so its size is the peak so far
    core::arch::wasm32::memory_size::<0>() as u64 * 65536 > WASM_MEMORY_BUDGET
}

#[cfg(not(target_arch = "wasm32"))]
fn over_memory_budget() -> bool {
    let _ = WASM_MEMORY_BUDGET;
    false
}
