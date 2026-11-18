//! Brute-force cross-check and rule cases for the exact solver.

use backpack_solver::exact::{self, Layout};
use backpack_solver::geometry::placements;
use backpack_solver::problem::{self, Problem};
use backpack_solver::verifier::evaluate;

fn solve_value(p: &Problem) -> i64 {
    let sol = exact::solve(p, &mut |_, _, _| {}).expect("solves");
    let score = evaluate(p, &sol.layout).expect("solver produced an illegal layout");
    assert_eq!(score.total, sol.value, "solver value vs verifier score");
    sol.value
}

fn parse(json: &str) -> Problem {
    problem::parse(json).expect("valid test instance")
}

#[test]
fn piece_reward_sums_bases_before_flooring() {
    // Row of 4: A(3) [piece] B(4) -> piece gets floor(7/5)=1; per-item flooring would give 0.
    let p = parse(
        r#"{"width":4,"height":1,"items":[
        {"name":"A","value":3,"shape":[[0,0]]},{"name":"B","value":4,"shape":[[0,0]]}]}"#,
    );
    assert_eq!(solve_value(&p), 3 + 4 + 1);
}

#[test]
fn item_bonus_merges_before_flooring() {
    // 1-cell item (15, adj) in the centre of 3x3 with 4 pieces around it:
    // item = 15 + floor(15*4/10) = 21 (separate flooring: 19), each piece floor(15/5) = 3.
    let p = parse(
        r#"{"width":3,"height":3,"items":[{"name":"A","value":15,"shape":[[0,0]],"adjBonus":true}]}"#,
    );
    assert_eq!(solve_value(&p), 21 + 4 * 3);
}

#[test]
fn border_bonus_merges_with_adjacency_bonus() {
    // 1x3 board, item(5, top, adj) at an end + one piece: 5 + floor(5*(2+1)/10) = 6, piece floor(5/5) = 1.
    let p = parse(
        r#"{"width":3,"height":1,"items":[
        {"name":"A","value":5,"shape":[[0,0]],"border":"top","adjBonus":true}]}"#,
    );
    assert_eq!(solve_value(&p), 6 + 1);
}

#[test]
fn blocked_cells_are_respected() {
    let p = parse(
        r#"{"width":3,"height":1,"blocked":[[1,0]],"items":[{"name":"A","value":10,"shape":[[0,0],[1,0]]}]}"#,
    );
    assert_eq!(solve_value(&p), 0);
}

#[test]
fn built_in_type_matches_explicit_shape() {
    let by_type = parse(r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"type":"U"}]}"#);
    let by_shape = parse(
        r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"shape":[[0,0],[1,0],[2,0],[0,1],[2,1]]}]}"#,
    );
    assert_eq!(solve_value(&by_type), solve_value(&by_shape));
}

#[test]
fn rejects_bad_input() {
    for bad in [
        r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"type":"Q"}]}"#,
        r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"type":"O","shape":[[0,0]]}]}"#,
        r#"{"width":3,"height":2,"blocked":[[0,6]],"items":[]}"#,
        r#"{"width":3,"height":2,"items":[{"name":"A","value":-1,"type":"O"}]}"#,
        r#"{"width":3,"#,
    ] {
        assert!(problem::parse(bad).is_err(), "should reject {bad}");
    }
}

#[test]
fn verifier_rejects_overlap_and_wrong_shape() {
    let p =
        parse(r#"{"width":3,"height":1,"items":[{"name":"A","value":1,"shape":[[0,0],[1,0]]}]}"#);
    let overlap = Layout {
        item_cells: vec![Some(vec![0, 1])],
        dominoes: vec![[1, 2]],
    };
    assert!(evaluate(&p, &overlap).is_err());
    let wrong = Layout {
        item_cells: vec![Some(vec![0, 2])],
        dominoes: vec![],
    };
    assert!(evaluate(&p, &wrong).is_err());
}

#[test]
fn matches_brute_force_on_random_small_boards() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for t in 0..150 {
        let json = random_instance(&mut rng);
        let p = parse(&json);
        assert_eq!(solve_value(&p), brute_force(&p), "instance {t}: {json}");
    }
}

/// xorshift64*: deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_instance(rng: &mut Rng) -> String {
    let (w, h) = (2 + rng.below(3), 2 + rng.below(3));
    let blocked: Vec<String> = (0..w * h)
        .filter(|_| rng.below(7) == 0)
        .map(|c| format!("[{},{}]", c % w, c / w))
        .collect();
    let borders = ["", "", "\"top\"", "\"bottom\"", "\"left\"", "\"right\""];
    let mut items: Vec<(String, u64, String, bool)> = Vec::new();
    for i in 0..1 + rng.below(4) {
        let (shape, value, border, adj) = if i > 0 && rng.below(3) == 0 {
            let src = items[rng.below(items.len() as u64) as usize].clone();
            if rng.below(2) == 0 {
                (src.0, src.1, src.2, src.3)
            } else {
                (
                    src.0,
                    1 + rng.below(40),
                    borders[rng.below(6) as usize].to_string(),
                    rng.below(2) == 0,
                )
            }
        } else {
            let size = 1 + rng.below(4) as usize;
            let shape = random_polyomino(rng, size);
            (
                shape,
                1 + rng.below(40),
                borders[rng.below(6) as usize].to_string(),
                rng.below(2) == 0,
            )
        };
        items.push((shape, value, border, adj));
    }
    let items: Vec<String> = items
        .iter()
        .enumerate()
        .map(|(i, (shape, value, border, adj))| {
            let border = if border.is_empty() { String::new() } else { format!(",\"border\":{border}") };
            format!("{{\"name\":\"I{i}\",\"value\":{value},\"shape\":{shape}{border},\"adjBonus\":{adj}}}")
        })
        .collect();
    format!(
        "{{\"width\":{w},\"height\":{h},\"blocked\":[{}],\"items\":[{}]}}",
        blocked.join(","),
        items.join(",")
    )
}

fn random_polyomino(rng: &mut Rng, size: usize) -> String {
    let mut cells = vec![(0i32, 0i32)];
    while cells.len() < size {
        let (x, y) = cells[rng.below(cells.len() as u64) as usize];
        let (dx, dy) = [(1, 0), (-1, 0), (0, 1), (0, -1)][rng.below(4) as usize];
        if !cells.contains(&(x + dx, y + dy)) {
            cells.push((x + dx, y + dy));
        }
    }
    let parts: Vec<String> = cells.iter().map(|(x, y)| format!("[{x},{y}]")).collect();
    format!("[{}]", parts.join(","))
}

/// Exhaustive: scan cells in order; each free cell is left empty, starts a piece, or anchors an item.
fn brute_force(p: &Problem) -> i64 {
    let (w, n) = (p.width as usize, (p.width * p.height) as usize);
    let pls: Vec<_> = (0..p.items.len()).map(|i| placements(p, i)).collect();
    let mut occ = p.blocked_mask();
    let mut layout = Layout {
        item_cells: vec![None; p.items.len()],
        dominoes: Vec::new(),
    };
    dfs(p, &pls, w, n, 0, &mut occ, &mut layout)
}

fn dfs(
    p: &Problem,
    pls: &[Vec<backpack_solver::geometry::Placement>],
    w: usize,
    n: usize,
    mut c: usize,
    occ: &mut Vec<bool>,
    layout: &mut Layout,
) -> i64 {
    while c < n && occ[c] {
        c += 1;
    }
    if c == n {
        return evaluate(p, layout)
            .expect("brute force builds legal layouts")
            .total;
    }
    let mut best = dfs(p, pls, w, n, c + 1, occ, layout); // leave empty
    for other in [c + 1, c + w] {
        let ok = other < n && !occ[other] && (other == c + w || c % w + 1 < w);
        if !ok {
            continue;
        }
        occ[c] = true;
        occ[other] = true;
        layout.dominoes.push([c, other]);
        best = best.max(dfs(p, pls, w, n, c + 1, occ, layout));
        layout.dominoes.pop();
        occ[c] = false;
        occ[other] = false;
    }
    for i in 0..pls.len() {
        if layout.item_cells[i].is_some() {
            continue;
        }
        for pl in &pls[i] {
            if pl.cells[0] != c || pl.cells.iter().any(|&k| occ[k]) {
                continue;
            }
            for &k in &pl.cells {
                occ[k] = true;
            }
            layout.item_cells[i] = Some(pl.cells.clone());
            best = best.max(dfs(p, pls, w, n, c + 1, occ, layout));
            layout.item_cells[i] = None;
            for &k in &pl.cells {
                occ[k] = false;
            }
        }
    }
    best
}
