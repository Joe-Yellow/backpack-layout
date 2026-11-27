//! Fast mode: beam search over the same sweep as the exact solver.
//!
//! Each layer (the states at one undecided cell) keeps only its `width` most promising states;
//! everything else is dropped, so the run takes a fixed amount of work. Every item may be placed at
//! most once from the start (no relaxation rounds). Scores use the exact rules, so the layout found
//! is scored exactly; it just isn't proven optimal and may miss layouts meeting tight targets.
//!
//! Promise of a state: the maximized total plus an optimistic outlook for the free cells left
//! (see Outlook), plus weighted progress towards each target, capped at the target. Without the
//! outlook, states that spent their best items early or put off a costly required item looked
//! best and then ran out of room. With targets a couple of passes run (TARGET_PASSES) and the best
//! layout meeting every target wins.

use crate::bits::{Bits, Wide};
use crate::exact::{to_layout, Solution};
use crate::problem::Problem;
use crate::sweep::{pos_of, Ents, Key, Score, Sweep, SLOTS};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Work per layer, in (state, anchored placement) pairs: sets the beam width. Measured on 13-20
/// items: width 5000 is within ~1% of 20000 at a quarter of the time; 2000 starts to lose.
const WORK_PER_LAYER: usize = 250_000;
const MIN_WIDTH: usize = 500;
const MAX_WIDTH: usize = 5_000;
/// Passes run when there are targets: (weight of target progress, whether that progress includes
/// the outlook). Counting only what is already reached pushes harder towards the targets and finds
/// better totals; adding the outlook steers clear of dead ends when targets are tight (it found
/// the only layouts on a 31-cell board where quality could reach 108 of 100). Two passes measured
/// within ~2% of four on 10 instances at half the time.
const TARGET_PASSES: [(f64, bool); 2] = [(1.0, false), (8.0, true)];
/// States expanded per parallel batch: bounds the children held before pruning.
const BATCH: usize = 1024;

struct Cand {
    key: Key,
    ents: Ents,
    used: Wide,
    acc: Score,
    free: u32,
    parent: u32,
    step: u32,
}

/// Best layout meeting every target and required item, or None if the beam found none.
/// Reports (round, pos, cells) after each layer, the round being the weight being tried.
pub fn solve(p: &Problem, on_progress: &mut dyn FnMut(u32, usize, usize)) -> Option<Solution> {
    let mut slots = vec![p.goal.maximize];
    slots.extend(p.goal.targets.iter().map(|&(a, _)| Some(a)));
    let sweep: Sweep<Wide> = Sweep::new(p, &slots);
    let mut targets = [None; SLOTS];
    for (j, &(_, t)) in p.goal.targets.iter().enumerate() {
        targets[j + 1] = Some(t.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    }
    // a typical item is worth about 1 in every slot
    let scale: [f64; SLOTS] = std::array::from_fn(|j| match slots.get(j) {
        Some(Some(a)) => p
            .items
            .iter()
            .map(|it| it.attrs[*a].abs())
            .max()
            .unwrap_or(0)
            .max(1) as f64,
        Some(None) => 4.0, // empty cells: about one item's area
        None => 1.0,
    });
    let width = (WORK_PER_LAYER / sweep.placements_per_cell().max(1)).clamp(MIN_WIDTH, MAX_WIDTH);
    let outlook = Outlook::new(p, &slots);
    let passes: &[(f64, bool)] = if p.goal.targets.is_empty() {
        &[(0.0, false)]
    } else {
        &TARGET_PASSES
    };
    let mut best: Option<Solution> = None;
    for (round, &(w, ahead_targets)) in passes.iter().enumerate() {
        let promise = |c: &Cand, _pos: usize| {
            let mut s = (c.acc[0] as f64 + outlook.rest(0, c.used, c.free)) / scale[0];
            for j in 1..SLOTS {
                if let Some(t) = targets[j] {
                    let ahead = if ahead_targets {
                        outlook.rest(j, c.used, c.free)
                    } else {
                        0.0
                    };
                    s += w * (c.acc[j] as f64 + ahead).min(t as f64) / scale[j];
                }
            }
            s
        };
        let found = run(&sweep, width, &promise, &targets, &mut |pos| {
            on_progress(round as u32 + 1, pos, sweep.n)
        });
        if let Some((value, path)) = found {
            if best.as_ref().is_none_or(|b| value as i64 > b.value) {
                best = Some(Solution {
                    phases: [0.0; 3],
                    value: value as i64,
                    layout: to_layout(&sweep, &path, p.items.len()),
                    rounds: round as u32 + 1,
                    tracked: 0,
                    nodes: 0,
                });
            }
        }
    }
    best
}

/// Optimistic value of the rest of the layout, per slot: the free cells filled with the unused
/// items of the best value per cell (the last one in part), unplaced required items first whatever
/// their value. Bonuses and pieces are left out: it ranks states, it bounds nothing.
struct Outlook {
    /// Per slot: (item, value, cells) by value per cell, best first.
    by_density: Vec<Vec<(usize, f64, f64)>>,
    required: Vec<usize>,
    vals: Vec<Score>,
    area: Vec<f64>,
    /// Slot 0 counts empty cells.
    empty_goal: bool,
}

impl Outlook {
    fn new(p: &Problem, slots: &[Option<usize>]) -> Outlook {
        let vals: Vec<Score> = p
            .items
            .iter()
            .map(|it| {
                std::array::from_fn(|j| slots.get(j).copied().flatten().map_or(0, |a| it.attrs[a]))
            })
            .collect();
        let area: Vec<f64> = p.items.iter().map(|it| it.shape.len() as f64).collect();
        let by_density = (0..SLOTS)
            .map(|j| {
                let mut v: Vec<(usize, f64, f64)> = (0..p.items.len())
                    .filter(|&i| !p.items[i].required && vals[i][j] > 0)
                    .map(|i| (i, vals[i][j] as f64, area[i]))
                    .collect();
                v.sort_by(|a, b| (b.1 / b.2).total_cmp(&(a.1 / a.2)));
                v
            })
            .collect();
        Outlook {
            by_density,
            required: (0..p.items.len())
                .filter(|&i| p.items[i].required)
                .collect(),
            vals,
            area,
            empty_goal: slots.first() == Some(&None),
        }
    }

    fn rest(&self, j: usize, used: Wide, free: u32) -> f64 {
        let mut cells = free as f64;
        let mut value = 0.0;
        for &i in &self.required {
            if (used & Wide::bit(i)).is_zero() {
                cells -= self.area[i];
                value += self.vals[i][j] as f64;
            }
        }
        if j == 0 && self.empty_goal {
            return cells.max(0.0); // at best every other cell stays empty
        }
        for &(i, v, a) in &self.by_density[j] {
            if cells <= 0.0 {
                break;
            }
            if (used & Wide::bit(i)).is_zero() {
                value += v * cells.min(a) / a;
                cells -= a;
            }
        }
        value
    }
}

/// One beam pass: the best final total meeting the targets and its path of (pos, code) steps.
fn run(
    sweep: &Sweep<Wide>,
    width: usize,
    promise: &(dyn Fn(&Cand, usize) -> f64 + Sync),
    targets: &[Option<i32>; SLOTS],
    on_layer: &mut dyn FnMut(usize),
) -> Option<(i32, Vec<(usize, u32)>)> {
    let all = (0..sweep.classes.len()).fold(Wide::default(), |m, c| m | sweep.classes[c].mask);
    let mut arena: Vec<(u32, u32)> = Vec::new(); // (parent, pos << 24 | code)
    let mut layers: Vec<Vec<Cand>> = (0..=sweep.n).map(|_| Vec::new()).collect();
    layers[pos_of(&sweep.root_key)].push(Cand {
        key: sweep.root_key.clone(),
        ents: Ents::new(),
        used: Wide::default(),
        acc: [0; SLOTS],
        free: sweep.n as u32, // only feeds the outlook, which the root never needs
        parent: u32::MAX,
        step: u32::MAX,
    });
    for pos in 0..sweep.n {
        let states = prune(std::mem::take(&mut layers[pos]), width, |c| promise(c, pos));
        let ids: Vec<u32> = states
            .iter()
            .map(|c| {
                arena.push((c.parent, c.step));
                arena.len() as u32 - 1
            })
            .collect();
        for (chunk, chunk_ids) in states.chunks(BATCH).zip(ids.chunks(BATCH)) {
            let children: Vec<Vec<Cand>> = chunk
                .par_iter()
                .zip(chunk_ids)
                .map_init(
                    || sweep.scratch(),
                    |sc, (c, &id)| {
                        let mut out = Vec::new();
                        sweep.expand(&c.key, &c.ents, c.used, all, sc, &mut |ch| {
                            out.push(Cand {
                                acc: std::array::from_fn(|j| c.acc[j] + ch.gain[j]),
                                free: ch.free,
                                key: ch.key,
                                ents: ch.ents,
                                used: ch.used,
                                parent: id,
                                step: (pos as u32) << 24 | ch.code,
                            });
                        });
                        out
                    },
                )
                .collect();
            for ch in children.into_iter().flatten() {
                let layer = pos_of(&ch.key);
                layers[layer].push(ch);
                if layer < sweep.n && layers[layer].len() >= 4 * width {
                    let full = std::mem::take(&mut layers[layer]);
                    layers[layer] = prune(full, 2 * width, |c| promise(c, layer));
                }
            }
        }
        on_layer(pos);
    }
    let finals = std::mem::take(&mut layers[sweep.n]);
    let best = finals
        .iter()
        .filter(|c| {
            (sweep.required & !c.used).is_zero()
                && (0..SLOTS).all(|j| targets[j].is_none_or(|t| c.acc[j] >= t))
        })
        .max_by_key(|c| c.acc[0])?;
    let mut path = vec![best.step];
    let mut id = best.parent;
    while id != u32::MAX {
        let (parent, step) = arena[id as usize];
        if step != u32::MAX {
            path.push(step);
        }
        id = parent;
    }
    path.reverse();
    Some((
        best.acc[0],
        path.into_iter()
            .map(|s| ((s >> 24) as usize, s & 0xFF_FFFF))
            .collect(),
    ))
}

/// Keeps the `width` most promising states, one per (key, open entities, used items): two such
/// states have the same futures, so only the better one is worth a slot in the beam.
fn prune(cands: Vec<Cand>, width: usize, promise: impl Fn(&Cand) -> f64) -> Vec<Cand> {
    let mut scored: Vec<(f64, Cand)> = cands.into_iter().map(|c| (promise(&c), c)).collect();
    let mut seen: FxHashMap<(&Key, &Ents, Wide), usize> = FxHashMap::default();
    let mut keep = vec![false; scored.len()];
    for (i, (s, c)) in scored.iter().enumerate() {
        match seen.entry((&c.key, &c.ents, c.used)) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(i);
                keep[i] = true;
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                if *s > scored[*o.get()].0 {
                    keep[*o.get()] = false;
                    keep[i] = true;
                    o.insert(i);
                }
            }
        }
    }
    drop(seen);
    let mut i = 0;
    scored.retain(|_| {
        i += 1;
        keep[i - 1]
    });
    if scored.len() > width {
        scored.select_nth_unstable_by(width, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(width);
    }
    scored.into_iter().map(|(_, c)| c).collect()
}
