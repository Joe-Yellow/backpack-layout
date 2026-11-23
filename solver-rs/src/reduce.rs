//! Drops items before solving.
//!
//! Exact: an item no layout needs. Items of one shape with the same border and adjacency settings
//! compete for the same placements, and at most `cap` = free cells / area of them fit. If at least
//! `cap` others are at least as good in every attribute the goal uses, an optimal layout using the
//! item can swap it for one of them left unused, losing nothing. Ties are broken by index, so
//! "dominates" is a strict order and a removed item always keeps `cap` dominators that stay (take a
//! maximal removed dominator: all of its own dominators dominate the item and stay).
//!
//! Greedy (fast mode only): beyond `limit`, keep required items, then the best value per cell.

use crate::geometry::shape_key;
use crate::problem::Problem;
use std::collections::HashMap;

/// The reduced problem and, per kept item, its index in `p` (ascending).
pub fn reduce(p: &Problem, limit: usize) -> (Problem, Vec<usize>) {
    let attrs: Vec<usize> = p
        .goal
        .maximize
        .into_iter()
        .chain(p.goal.targets.iter().map(|&(a, _)| a))
        .collect();
    let free = (p.width * p.height) as usize - p.blocked.len();
    let mut groups: HashMap<_, Vec<usize>> = HashMap::new();
    for (i, it) in p.items.iter().enumerate() {
        groups
            .entry((shape_key(&it.shape), it.border, it.adj_bonus))
            .or_default()
            .push(i);
    }
    let mut keep = vec![true; p.items.len()];
    for members in groups.values() {
        let cap = free / p.items[members[0]].shape.len();
        for &i in members {
            if p.items[i].required {
                continue;
            }
            let a = &p.items[i].attrs;
            let dominators = members
                .iter()
                .filter(|&&j| {
                    let b = &p.items[j].attrs;
                    j != i
                        && attrs.iter().all(|&k| b[k] >= a[k])
                        && (attrs.iter().any(|&k| b[k] > a[k]) || j < i)
                })
                .count();
            keep[i] = dominators < cap;
        }
    }
    let mut kept: Vec<usize> = (0..p.items.len()).filter(|&i| keep[i]).collect();
    if kept.len() > limit {
        kept = greedy_cut(p, &attrs, kept, limit);
    }
    let reduced = Problem {
        items: kept.iter().map(|&i| p.items[i].clone()).collect(),
        ..p.clone()
    };
    (reduced, kept)
}

/// Keeps the required items and then the highest value per cell, each attribute scaled by its
/// largest magnitude so they weigh alike.
// ponytail: ignores border/adjacency bonuses and how shapes pack; fine for cutting obvious losers
fn greedy_cut(p: &Problem, attrs: &[usize], kept: Vec<usize>, limit: usize) -> Vec<usize> {
    let scale: Vec<f64> = (0..3)
        .map(|k| {
            let m = kept.iter().map(|&i| p.items[i].attrs[k].abs()).max();
            m.unwrap_or(0).max(1) as f64
        })
        .collect();
    let score = |i: usize| {
        let it = &p.items[i];
        let v: f64 = attrs.iter().map(|&k| it.attrs[k] as f64 / scale[k]).sum();
        v / it.shape.len() as f64
    };
    let mut order = kept;
    order.sort_by(|&x, &y| {
        let (rx, ry) = (p.items[x].required, p.items[y].required);
        ry.cmp(&rx).then(score(y).total_cmp(&score(x)))
    });
    order.truncate(limit.max(order.iter().filter(|&&i| p.items[i].required).count()));
    order.sort_unstable();
    order
}
