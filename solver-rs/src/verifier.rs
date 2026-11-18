//! Scores a layout straight from the rules, independent of the solver.

use crate::exact::Layout;
use crate::geometry::{neighbours, orientations, touches};
use crate::problem::Problem;
use std::collections::HashSet;

pub struct Score {
    pub total: i64,
    pub item_values: Vec<i64>,
    pub domino_values: Vec<i64>,
}

const FREE: i64 = -1;
const BLOCKED: i64 = -2;

/// Errors if the layout breaks any placement rule.
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
    let domino_values: Vec<i64> = l
        .dominoes
        .iter()
        .map(|cells| {
            let items: HashSet<usize> = cells
                .iter()
                .flat_map(|&c| neighbours(c, w, h))
                .filter(|&nb| owner[nb] >= 0 && (owner[nb] as usize) < n)
                .map(|nb| owner[nb] as usize)
                .collect();
            let sum: i64 = items.iter().map(|&i| p.items[i].value as i64).sum();
            for &i in &items {
                adjacent_dominoes[i] += 1;
            }
            sum / 5 // floor(20% of the merged sum)
        })
        .collect();
    let item_values: Vec<i64> = p
        .items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let Some(cells) = &l.item_cells[i] else {
                return 0;
            };
            let t = touches(it.border, cells, w, h) as i64;
            let k = if it.adj_bonus {
                adjacent_dominoes[i]
            } else {
                0
            };
            let v = it.value as i64;
            v + v * (2 * t + k) / 10 // merged, then floored
        })
        .collect();
    let total = item_values.iter().sum::<i64>() + domino_values.iter().sum::<i64>();
    Ok(Score {
        total,
        item_values,
        domino_values,
    })
}
