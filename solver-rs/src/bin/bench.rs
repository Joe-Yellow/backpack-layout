//! Native benchmark: solves the given instance files and prints value, rounds, nodes, the time spent
//! in each DP phase [gather, expand, insert] and the total. Every result is checked by the verifier.
//! Thread count: RAYON_NUM_THREADS (defaults to all cores). BENCH_LOG=1 prints round start times.
//! FAST=1 runs the fast mode (item reduction + beam search) through solve_json instead.

use backpack_solver::{exact, problem, verifier};
use std::time::Instant;

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: bench <instance.json>...");
        std::process::exit(2);
    }
    for f in files {
        if let Err(e) = run(&f) {
            eprintln!("{f}: {e}");
            std::process::exit(1);
        }
    }
}

fn run(f: &str) -> Result<(), String> {
    let json = std::fs::read_to_string(f).map_err(|e| e.to_string())?;
    if std::env::var_os("FAST").is_some() {
        let mut v: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        v["mode"] = "fast".into();
        let t0 = Instant::now();
        let out = backpack_solver::solve_json(&v.to_string(), &mut |_, _, _| {})?;
        println!(
            "{f}: fast feasible={} totals={:?} items={} weights_round={} time={:.2}s",
            out.feasible,
            out.totals,
            out.items_considered,
            out.rounds,
            t0.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    let p = problem::parse(&json)?;
    let t0 = Instant::now();
    let verbose = std::env::var_os("BENCH_LOG").is_some();
    let mut last_round = 0;
    let mut progress = |round: u32, _pos: usize, _cells: usize| {
        if verbose && round != last_round {
            eprintln!(
                "  round {round} started at {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            last_round = round;
        }
    };
    let sol = match exact::solve(&p, &mut progress)? {
        exact::Outcome::Solved(sol) => sol,
        exact::Outcome::Infeasible => {
            println!("{f}: infeasible time={:.1}s", t0.elapsed().as_secs_f64());
            return Ok(());
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    let score = verifier::evaluate(&p, &sol.layout)?;
    if !score.feasible(&p, &sol.layout) || score.objective(&p) != sol.value {
        return Err(format!(
            "solver value {} != verifier score {} (or targets unmet)",
            sol.value,
            score.objective(&p)
        ));
    }
    println!(
        "{f}: optimum={} totals={:?} rounds={} nodes={} phases={:.1?} time={secs:.1}s",
        sol.value, score.totals, sol.rounds, sol.nodes, sol.phases
    );
    Ok(())
}
