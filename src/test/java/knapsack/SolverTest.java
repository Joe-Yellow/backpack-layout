package knapsack;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;

class SolverTest {

    private static Problem.Item item(String name, int value, String border, boolean adj, int[]... shape) {
        return new Problem.Item(name, value, List.of(shape), border, adj);
    }

    private static Problem problem(int w, int h, List<int[]> blocked, Problem.Item... items) {
        return new Problem(w, h, blocked, List.of(items)).validated();
    }

    private static long solveOptimal(Problem p) {
        ExactSolver.Result r = ExactSolver.solve(p);
        assertEquals(r.value(), Verifier.evaluate(p, r.layout()).total(), "solver vs verifier");
        return r.value();
    }

    @Test
    void orientations_deduplicateSymmetricShapes() {
        int[] o = {0, 0};
        assertEquals(1, Geometry.orientations(List.of(o)).size());
        assertEquals(2, Geometry.orientations(List.of(o, new int[] {1, 0})).size());
        assertEquals(4, Geometry.orientations(List.of(o, new int[] {1, 0}, new int[] {1, 1})).size()); // L3
        assertEquals(4, Geometry.orientations(List.of(o, new int[] {1, 0}, new int[] {2, 0}, new int[] {1, 1})).size()); // T
        assertEquals(4, Geometry.orientations(List.of(o, new int[] {1, 0}, new int[] {1, 1}, new int[] {2, 1})).size()); // S
        assertEquals(8, Geometry.orientations(List.of(o, new int[] {0, 1}, new int[] {0, 2}, new int[] {1, 2})).size()); // L4
        assertEquals(1, Geometry.orientations(List.of(o, new int[] {1, 0}, new int[] {0, 1}, new int[] {1, 1})).size()); // O
    }

    @Test
    void dominoReward_sumsBasesBeforeFlooring() {
        // Row of 4: A(3) [domino] B(4) -> domino gets floor(7/5)=1; per-item flooring would give 0.
        Problem p = problem(4, 1, List.of(),
                item("A", 3, null, false, new int[] {0, 0}),
                item("B", 4, null, false, new int[] {0, 0}));
        assertEquals(3 + 4 + 1, solveOptimal(p));
    }

    @Test
    void itemBonus_mergesBeforeFlooring() {
        // 1-cell item (15, adj) in the centre of 3x3, 4 dominoes around it:
        // item = 15 + floor(15*4/10) = 21 (separate flooring: 19), each domino floor(15/5)=3.
        Problem p = problem(3, 3, List.of(), item("A", 15, null, true, new int[] {0, 0}));
        assertEquals(21 + 4 * 3, solveOptimal(p));
    }

    @Test
    void borderBonus_mergedWithAdjacencyBonus() {
        // 1x3 board, item(5, top, adj) at an end + one domino: 5 + floor(5*(2+1)/10)=6, domino floor(5/5)=1.
        Problem p = problem(3, 1, List.of(), item("A", 5, "top", true, new int[] {0, 0}));
        assertEquals(6 + 1, solveOptimal(p));
    }

    @Test
    void blockedCellsAreRespected() {
        // 2x1 item can't fit on 3x1 board with middle cell blocked.
        Problem p = problem(3, 1, List.of(new int[] {1, 0}), item("A", 10, null, false, new int[] {0, 0}, new int[] {1, 0}));
        assertEquals(0, solveOptimal(p));
    }

    @Test
    void itemType_resolvesBuiltInShape() {
        Problem byType = new Problem(3, 2, null, List.of(new Problem.Item("A", 9, null, null, false, "U"))).validated();
        Problem byShape = problem(3, 2, List.of(),
                item("A", 9, null, false, new int[] {0, 0}, new int[] {1, 0}, new int[] {2, 0}, new int[] {0, 1}, new int[] {2, 1}));
        assertEquals(solveOptimal(byShape), solveOptimal(byType));
        assertThrows(IllegalArgumentException.class, () -> new Problem(3, 2, null,
                List.of(new Problem.Item("A", 9, null, null, false, "Nope"))).validated());
        assertThrows(IllegalArgumentException.class, () -> new Problem(3, 2, null,
                List.of(new Problem.Item("A", 9, List.<int[]>of(new int[] {0, 0}), null, false, "O"))).validated());
    }

    @Test
    void verifier_rejectsOverlapAndWrongShape() {
        Problem p = problem(3, 1, List.of(), item("A", 1, null, false, new int[] {0, 0}, new int[] {1, 0}));
        assertThrows(IllegalArgumentException.class, () -> Verifier.evaluate(p,
                new Verifier.Layout(new int[][] {{0, 1}}, List.<int[]>of(new int[] {1, 2}))));
        assertThrows(IllegalArgumentException.class, () -> Verifier.evaluate(p,
                new Verifier.Layout(new int[][] {{0, 2}}, List.of())));
    }

    @Test
    void solverMatchesBruteForceOnRandomSmallBoards() {
        Random rnd = new Random(42);
        for (int t = 0; t < 150; t++) {
            Problem p = randomProblem(rnd);
            assertEquals(bruteForce(p), solveOptimal(p), "instance " + t);
        }
    }

    // ---- helpers ----

    private static Problem randomProblem(Random rnd) {
        int w = 2 + rnd.nextInt(3);
        int h = 2 + rnd.nextInt(3);
        List<int[]> blocked = new ArrayList<>();
        for (int c = 0; c < w * h; c++) {
            if (rnd.nextInt(7) == 0) blocked.add(new int[] {c % w, c / w});
        }
        String[] borders = {null, null, "top", "bottom", "left", "right"};
        List<Problem.Item> items = new ArrayList<>();
        int count = 1 + rnd.nextInt(4);
        for (int i = 0; i < count; i++) {
            if (i > 0 && rnd.nextInt(3) == 0) { // same shape as an earlier item (exact copy half the time)
                Problem.Item src = items.get(rnd.nextInt(items.size()));
                items.add(rnd.nextBoolean() ? new Problem.Item("I" + i, src.value(), src.shape(), src.border(), src.adjBonus())
                        : new Problem.Item("I" + i, 1 + rnd.nextInt(40), src.shape(),
                                borders[rnd.nextInt(borders.length)], rnd.nextBoolean()));
                continue;
            }
            items.add(new Problem.Item("I" + i, 1 + rnd.nextInt(40), randomPolyomino(rnd, 1 + rnd.nextInt(4)),
                    borders[rnd.nextInt(borders.length)], rnd.nextBoolean()));
        }
        return new Problem(w, h, blocked, items).validated();
    }

    private static List<int[]> randomPolyomino(Random rnd, int size) {
        List<int[]> cells = new ArrayList<>(List.of(new int[] {0, 0}));
        Set<String> seen = new HashSet<>(Set.of("0,0"));
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        while (cells.size() < size) {
            int[] base = cells.get(rnd.nextInt(cells.size()));
            int[] dir = dirs[rnd.nextInt(4)];
            int[] next = {base[0] + dir[0], base[1] + dir[1]};
            if (seen.add(next[0] + "," + next[1])) cells.add(next);
        }
        return cells;
    }

    /** Exhaustive: scan cells in order; each free cell is left empty, starts a domino, or anchors an item. */
    private static long bruteForce(Problem p) {
        int n = p.items().size();
        List<List<Geometry.Placement>> pl = new ArrayList<>();
        for (int i = 0; i < n; i++) pl.add(Geometry.placements(p, i));
        boolean[] occ = p.blockedMask();
        return dfs(p, pl, 0, occ, new int[n][], new ArrayList<>());
    }

    private static long dfs(Problem p, List<List<Geometry.Placement>> pl, int c, boolean[] occ,
            int[][] itemCells, List<int[]> doms) {
        int w = p.width();
        while (c < occ.length && occ[c]) c++;
        if (c == occ.length) return Verifier.evaluate(p, new Verifier.Layout(itemCells.clone(), List.copyOf(doms))).total();

        long best = dfs(p, pl, c + 1, occ, itemCells, doms); // leave empty
        int[][] partners = {{c, c + 1}, {c, c + w}};
        for (int[] dm : partners) {
            boolean ok = dm[1] < occ.length && !occ[dm[1]] && (dm[1] == c + w || c % w + 1 < w);
            if (!ok) continue;
            occ[c] = occ[dm[1]] = true;
            doms.add(dm);
            best = Math.max(best, dfs(p, pl, c + 1, occ, itemCells, doms));
            doms.remove(doms.size() - 1);
            occ[c] = occ[dm[1]] = false;
        }
        for (int i = 0; i < itemCells.length; i++) {
            if (itemCells[i] != null) continue;
            for (Geometry.Placement g : pl.get(i)) {
                if (g.cells()[0] != c || Arrays.stream(g.cells()).anyMatch(k -> occ[k])) continue;
                for (int k : g.cells()) occ[k] = true;
                itemCells[i] = g.cells();
                best = Math.max(best, dfs(p, pl, c + 1, occ, itemCells, doms));
                itemCells[i] = null;
                for (int k : g.cells()) occ[k] = false;
            }
        }
        return best;
    }
}
