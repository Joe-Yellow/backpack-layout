//! Scores a layout straight from the rules, independent of the solver.

use crate::exact::Layout;
use crate::geometry::{neighbours, orientations, touches};
use crate::problem::Problem;
use std::collections::HashSet;

pub struct Score {
    /// Totals per attribute (performance, quality, efficiency).
    pub totals: [i64; 3],
    pub item_values: Vec<[i64; 3]>,
    pub domino_values: Vec<[i64; 3]>,
    /// Non-blocked cells covered by neither an item nor a piece.
    pub empty_cells: i64,
}

impl Score {
    /// Every required item is placed and every target is met.
    pub fn feasible(&self, p: &Problem, l: &Layout) -> bool {
        let required_placed = p
            .items
            .iter()
            .zip(&l.item_cells)
            .all(|(it, cells)| !it.required || cells.is_some());
        required_placed && p.goal.targets.iter().all(|&(a, t)| self.totals[a] >= t)
    }

    /// The maximized attribute's total, or the empty cells when none is maximized.
    pub fn objective(&self, p: &Problem) -> i64 {
        p.goal.maximize.map_or(self.empty_cells, |a| self.totals[a])
    }
}

const FREE: i64 = -1;
const BLOCKED: i64 = -2;

/// Errors if the layout breaks any placement rule. Percentages round toward zero (Rust `/`),
/// for negative values too: an item's border and adjacency bonuses are merged before rounding,
/// and a piece takes 20% of the summed bases of its neighbours.
pub fn evaluate(p: &Problem, l: &Layout) -> Result<Score, String> {
    let (w, h) = (p.width as usize, p.height as usize);
    let n = p.items.len();
    let mut owner = vec![FREE; w * h];
    for c in &p.blocked {
        owner[c[1] as usize * w + c[0] as usize] = BLOCKED;
    }
    let mut claim = |c: usize, who: i64| -> Result<(), String> {
        match owner.get(c) {
            None => Err(format!("cell out of board: {c}")),
            Some(&FREE) => {
                owner[c] = who;
                Ok(())
            }
            Some(_) => Err(format!("cell {c} blocked or already occupied")),
        }
    };
    for (i, cells) in l.item_cells.iter().enumerate() {
        let Some(cells) = cells else { continue };
        let pts: Vec<[i32; 2]> = cells
            .iter()
            .map(|&c| [(c % w) as i32, (c / w) as i32])
            .collect();
        let placed = &orientations(&pts)[0];
        if !orientations(&p.items[i].shape).contains(placed) {
            return Err(format!("{}: cells do not match its shape", p.items[i].name));
        }
        for &c in cells {
            claim(c, i as i64)?;
        }
    }
    for (j, &[a, b]) in l.dominoes.iter().enumerate() {
        let (a, b) = (a.min(b), a.max(b));
        let horizontal = b == a + 1 && a / w == b / w;
        if !horizontal && b != a + w {
            return Err(format!("domino not 1x2: {a},{b}"));
        }
        claim(a, (n + j) as i64)?;
        claim(b, (n + j) as i64)?;
    }

    let mut adjacent_dominoes = vec![0i64; n];
    let domino_values: Vec<[i64; 3]> = l
        .dominoes
        .iter()
        .map(|cells| {
            let items: HashSet<usize> = cells
                .iter()
                .flat_map(|&c| neighbours(c, w, h))
                .filter(|&nb| owner[nb] >= 0 && (owner[nb] as usize) < n)
                .map(|nb| owner[nb] as usize)
                .collect();
            for &i in &items {
                adjacent_dominoes[i] += 1;
            }
            std::array::from_fn(|a| {
                let sum: i64 = items.iter().map(|&i| p.items[i].attrs[a] as i64).sum();
                sum / 5 // 20% of the merged sum, toward zero
            })
        })
        .collect();
    let item_values: Vec<[i64; 3]> = p
        .items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let Some(cells) = &l.item_cells[i] else {
                return [0; 3];
            };
            let t = touches(it.border, cells, w, h) as i64;
            let k = if it.adj_bonus {
                adjacent_dominoes[i]
            } else {
                0
            };
            std::array::from_fn(|a| {
                let v = it.attrs[a] as i64;
                v + v * (2 * t + k) / 10 // merged, then toward zero
            })
        })
        .collect();
    let totals = std::array::from_fn(|a| {
        item_values.iter().map(|v| v[a]).sum::<i64>()
            + domino_values.iter().map(|v| v[a]).sum::<i64>()
    });
    let empty_cells = owner.iter().filter(|&&o| o == FREE).count() as i64;
    Ok(Score {
        totals,
        item_values,
        domino_values,
        empty_cells,
    })
}
