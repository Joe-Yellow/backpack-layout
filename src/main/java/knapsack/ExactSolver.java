package knapsack;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Exact solver with lazily added "each item at most once" constraints.
 *
 * <p>Each round runs a forward DP over the Sweep in which only the tracked items are limited to one use;
 * all other classes may be reused. That is a relaxation, so its optimum is an upper bound on the true
 * optimum. If the best relaxed layout happens to use every class within its count, it is feasible and
 * therefore optimal. Otherwise the over-used classes become tracked and the round repeats. Tracking
 * every item gives the original problem, so this terminates with a proven optimum.
 *
 * <p>Within one layer, partial layouts with the same Sweep.shapeKey form a Pareto front. A dominates B
 * if A used a subset of B's tracked items, each open item of A has value >= and the bonus flag whenever
 * B's has it, and acc(A) - acc(B) >= #entities where A's remainder is smaller: every completion of B is
 * open to A, future rewards are monotone in those values, and a smaller remainder costs at most 1 point
 * per entity (an entity's whole future reward is floor((rem + added) / q)).
 *
 * <p>Each layer is expanded in parallel; a step always moves pos forward, so children land in later
 * layers, and each front is updated atomically under its map bin lock.
 */
final class ExactSolver {

    record Result(long value, Verifier.Layout layout, int rounds, int trackedItems, long nodes) {}

    /** Partial layout; state is released once expanded, the rest is kept to rebuild the layout. */
    private static final class Node {
        Sweep.State state;
        final long acc;
        final Node parent;
        final Sweep.Step step;

        Node(Sweep.State state, long acc, Node parent, Sweep.Step step) {
            this.state = state;
            this.acc = acc;
            this.parent = parent;
            this.step = step;
        }
    }

    static final boolean LOG = Boolean.getBoolean("exact.log");

    private final Sweep sweep;
    private final long tracked;
    private final List<Map<Sweep.Key, List<Node>>> layers = new ArrayList<>();
    private final LongAdder nodes = new LongAdder();

    private ExactSolver(Sweep sweep, long tracked) {
        this.sweep = sweep;
        this.tracked = tracked;
        for (int i = 0; i <= sweep.n; i++) layers.add(new ConcurrentHashMap<>());
    }

    static Result solve(Problem p) {
        Sweep sweep = new Sweep(p);
        long tracked = 0;
        long totalNodes = 0;
        for (int round = 1; ; round++) {
            ExactSolver ex = new ExactSolver(sweep, tracked);
            List<Sweep.Step> path = ex.run();
            totalNodes += ex.nodes.sum();
            long acc = path.stream().mapToLong(Sweep.Step::gain).sum();
            long overUsed = overUsedClasses(sweep, path);
            if (LOG) {
                System.err.printf("round %d: tracked=%d relaxed optimum=%d nodes=%d over-used mask=%x%n",
                        round, Long.bitCount(tracked), acc, ex.nodes.sum(), overUsed);
            }
            if (overUsed == 0) {
                return new Result(acc, sweep.layout(path, p.items().size()), round, Long.bitCount(tracked), totalNodes);
            }
            tracked |= overUsed;
        }
    }

    /** Item bits of every class used more often than it has members on this path. */
    private static long overUsedClasses(Sweep sweep, List<Sweep.Step> path) {
        int[] count = new int[sweep.classes.size()];
        for (Sweep.Step st : path) {
            if (st.cells() != null && st.cls() != Sweep.DOMINO) count[st.cls()]++;
        }
        long mask = 0;
        for (int c = 0; c < count.length; c++) {
            if (count[c] > sweep.classes.get(c).members().length) mask |= sweep.classes.get(c).mask();
        }
        return mask;
    }

    /**
     * Runs the forward DP; returns the steps of one best complete layout, root first. Among tied best
     * layouts, prefers one that over-uses nothing, which saves a needless extra round.
     */
    private List<Sweep.Step> run() {
        insert(new Node(sweep.root(), 0, null, null));
        for (int pos = 0; pos < sweep.n; pos++) expandLayer(pos);
        long bestAcc = Long.MIN_VALUE;
        for (List<Node> front : layers.get(sweep.n).values()) {
            for (Node nd : front) bestAcc = Math.max(bestAcc, nd.acc);
        }
        List<Sweep.Step> pick = null;
        for (List<Node> front : layers.get(sweep.n).values()) {
            for (Node nd : front) {
                if (nd.acc != bestAcc) continue;
                List<Sweep.Step> path = new ArrayList<>();
                for (Node at = nd; at.step != null; at = at.parent) path.add(0, at.step);
                if (pick == null || overUsedClasses(sweep, path) == 0) pick = path;
                if (overUsedClasses(sweep, pick) == 0) return pick;
            }
        }
        return pick;
    }

    private void expandLayer(int pos) {
        Map<Sweep.Key, List<Node>> layer = layers.get(pos);
        layer.values().parallelStream().forEach(front -> {
            for (Node nd : front) {
                for (Sweep.Step st : sweep.steps(nd.state, tracked)) {
                    insert(new Node(st.next(), nd.acc + st.gain(), nd, st));
                }
                nd.state = null;
            }
        });
        layer.clear();
    }

    /**
     * Adds a node unless dominated; removes nodes it dominates. Atomic per key. Fronts are kept sorted by
     * acc descending: a dominator never has lower acc (slack must stay >= 0), so only the part with
     * acc >= nd.acc can dominate nd, and only the part with acc <= nd.acc can be dominated by it.
     */
    private void insert(Node nd) {
        layers.get(nd.state.pos()).compute(sweep.shapeKey(nd.state), (k, front) -> {
            List<Node> f = front == null ? new ArrayList<>() : front;
            int at = 0;
            while (at < f.size() && f.get(at).acc >= nd.acc) {
                if (dominates(f.get(at), nd)) return f;
                at++;
            }
            int lo = at;
            while (lo > 0 && f.get(lo - 1).acc == nd.acc) lo--; // equal acc may also be dominated by nd
            for (int i = f.size() - 1; i >= lo; i--) {
                if (dominates(nd, f.get(i))) {
                    f.remove(i);
                    if (i < at) at--;
                }
            }
            f.add(at, nd);
            nodes.increment();
            return f;
        });
    }

    private static boolean dominates(Node a, Node b) {
        if ((a.state.used() & ~b.state.used()) != 0) return false;
        long slack = a.acc - b.acc;
        Sweep.Ent[] ea = a.state.ents();
        Sweep.Ent[] eb = b.state.ents();
        for (int i = 0; i < ea.length && slack >= 0; i++) {
            Sweep.Ent x = ea[i];
            Sweep.Ent y = eb[i];
            if (!x.isDomino() && (x.value() < y.value() || (y.adjBonus() && !x.adjBonus()))) return false;
            if (x.rem() < y.rem() && (x.isDomino() || y.adjBonus())) slack--;
        }
        return slack >= 0;
    }
}
