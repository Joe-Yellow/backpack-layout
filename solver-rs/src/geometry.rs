//! Shape orientations and placement enumeration. Board cells are indexed y * width + x.

use crate::problem::{Border, Cell, Problem};

/// cells sorted ascending; touch = how many of the item's border sides it touches.
pub struct Placement {
    pub cells: Vec<usize>,
    pub touch: u8,
}

/// Distinct normalized orientations among the 8 rotations/mirrors; points sorted by (y, x).
pub fn orientations(shape: &[Cell]) -> Vec<Vec<Cell>> {
    let mut out: Vec<Vec<Cell>> = Vec::new();
    for t in 0..8 {
        let mut pts: Vec<Cell> = shape
            .iter()
            .map(|&[x0, y0]| {
                let (mut x, mut y) = (x0, y0);
                for _ in 0..t % 4 {
                    (x, y) = (-y, x);
                }
                if t >= 4 {
                    x = -x;
                }
                [x, y]
            })
            .collect();
        let min_x = pts.iter().map(|p| p[0]).min().unwrap();
        let min_y = pts.iter().map(|p| p[1]).min().unwrap();
        for p in &mut pts {
            p[0] -= min_x;
            p[1] -= min_y;
        }
        pts.sort_by_key(|p| (p[1], p[0]));
        if !out.contains(&pts) {
            out.push(pts);
        }
    }
    out
}

/// Orientation-independent shape identity.
pub fn shape_key(shape: &[Cell]) -> Vec<Cell> {
    orientations(shape).into_iter().min().unwrap()
}

/// Number of the given sides the cells touch (each touched side earns its own bonus).
pub fn touches(border: &[Border], cells: &[usize], w: usize, h: usize) -> u8 {
    let on = |b: Border| {
        cells.iter().any(|&c| {
            let (x, y) = (c % w, c / w);
            match b {
                Border::Top => y == 0,
                Border::Bottom => y == h - 1,
                Border::Left => x == 0,
                Border::Right => x == w - 1,
            }
        })
    };
    border.iter().filter(|&&b| on(b)).count() as u8
}

/// All legal placements of one item: inside the board, not on blocked cells.
pub fn placements(p: &Problem, item: usize) -> Vec<Placement> {
    let (w, h) = (p.width, p.height);
    let blocked = p.blocked_mask();
    let it = &p.items[item];
    let mut out = Vec::new();
    for o in orientations(&it.shape) {
        let ow = o.iter().map(|q| q[0]).max().unwrap() + 1;
        let oh = o.iter().map(|q| q[1]).max().unwrap() + 1;
        for dy in 0..=(h - oh) {
            for dx in 0..=(w - ow) {
                let cells: Vec<usize> = o
                    .iter()
                    .map(|&[x, y]| ((y + dy) * w + x + dx) as usize)
                    .collect();
                if cells.iter().any(|&c| blocked[c]) {
                    continue;
                }
                let touch = touches(&it.border, &cells, w as usize, h as usize);
                out.push(Placement { cells, touch });
            }
        }
    }
    out
}

/// 4-neighbours of a cell inside the board.
pub fn neighbours(c: usize, w: usize, h: usize) -> Vec<usize> {
    let (x, y) = (c % w, c / w);
    let mut out = Vec::with_capacity(4);
    if x > 0 {
        out.push(c - 1);
    }
    if x + 1 < w {
        out.push(c + 1);
    }
    if y > 0 {
        out.push(c - w);
    }
    if y + 1 < h {
        out.push(c + w);
    }
    out
}
