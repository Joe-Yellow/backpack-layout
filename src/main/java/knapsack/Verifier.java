package knapsack;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Scores a layout straight from the rules, independent of the solver.
 * Used to double-check the solver and as the oracle for brute-force tests.
 */
final class Verifier {

    /** itemCells[i] = cells of item i, or null if unused. dominoes = {a, b} cell pairs. */
    record Layout(int[][] itemCells, List<int[]> dominoes) {}

    record Score(long total, long[] itemValues, long[] dominoValues) {}

    private static final int FREE = -1;
    private static final int BLOCKED = -2;

    private Verifier() {}

    /** Throws IllegalArgumentException if the layout breaks any placement rule. */
    static Score evaluate(Problem p, Layout l) {
        int w = p.width();
        int h = p.height();
        int n = p.items().size();
        int[] owner = new int[w * h];
        Arrays.fill(owner, FREE);
        for (int[] b : p.blocked()) owner[b[1] * w + b[0]] = BLOCKED;

        for (int i = 0; i < n; i++) {
            int[] cells = l.itemCells()[i];
            if (cells == null) continue;
            checkShape(p.items().get(i), cells, w);
            for (int c : cells) claim(owner, c, i);
        }
        for (int j = 0; j < l.dominoes().size(); j++) {
            int[] d = l.dominoes().get(j);
            int a = Math.min(d[0], d[1]);
            int b = Math.max(d[0], d[1]);
            boolean horizontal = b == a + 1 && a / w == b / w;
            if (!horizontal && b != a + w) throw new IllegalArgumentException("domino not 1x2: " + Arrays.toString(d));
            claim(owner, a, n + j);
            claim(owner, b, n + j);
        }

        long[] dominoValues = new long[l.dominoes().size()];
        int[] adjacentDominoes = new int[n];
        for (int j = 0; j < dominoValues.length; j++) {
            Set<Integer> items = new HashSet<>();
            for (int c : l.dominoes().get(j)) {
                for (int nb : Geometry.neighbours(c, w, h)) {
                    if (owner[nb] >= 0 && owner[nb] < n) items.add(owner[nb]);
                }
            }
            long baseSum = 0;
            for (int i : items) {
                baseSum += p.items().get(i).value();
                adjacentDominoes[i]++;
            }
            dominoValues[j] = baseSum / 5; // floor(20% of merged sum)
        }

        long[] itemValues = new long[n];
        for (int i = 0; i < n; i++) {
            if (l.itemCells()[i] == null) continue;
            Problem.Item it = p.items().get(i);
            int touch = Geometry.touches(it.border(), l.itemCells()[i], w, h) ? 1 : 0;
            int k = it.adjBonus() ? adjacentDominoes[i] : 0;
            itemValues[i] = it.value() + (long) it.value() * (2L * touch + k) / 10; // merged, then floored
        }
        long total = Arrays.stream(itemValues).sum() + Arrays.stream(dominoValues).sum();
        return new Score(total, itemValues, dominoValues);
    }

    private static void claim(int[] owner, int c, int who) {
        if (c < 0 || c >= owner.length) throw new IllegalArgumentException("cell out of board: " + c);
        if (owner[c] != FREE) throw new IllegalArgumentException("cell " + c + " blocked or already occupied");
        owner[c] = who;
    }

    private static void checkShape(Problem.Item it, int[] cells, int w) {
        List<int[]> pts = Arrays.stream(cells).mapToObj(c -> new int[] {c % w, c / w}).toList();
        if (pts.stream().map(q -> q[0] + "," + q[1]).distinct().count() != pts.size()) {
            throw new IllegalArgumentException(it.name() + ": duplicate cells");
        }
        int[][] placed = Geometry.orientations(pts).get(0); // t=0 is identity, just normalized
        for (int[][] o : Geometry.orientations(it.shape())) {
            if (Arrays.deepEquals(o, placed)) return;
        }
        throw new IllegalArgumentException(it.name() + ": cells do not match its shape");
    }
}
