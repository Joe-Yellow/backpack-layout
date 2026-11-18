package knapsack;

import java.nio.file.Path;
import java.util.Arrays;

public final class Main {

    // ponytail: 36 labels; more items print '*' (see the breakdown list for exact cells)
    private static final String ITEM_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    private Main() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            Server.start(Integer.getInteger("port", 8080));
            return;
        }
        if (args.length != 1) {
            System.err.println("usage: (no args: web UI on -Dport=8080) | <input.json>");
            System.exit(2);
        }
        Problem p = Problem.load(Path.of(args[0]));

        long start = System.nanoTime();
        ExactSolver.Result r = ExactSolver.solve(p);
        double seconds = (System.nanoTime() - start) / 1e9;

        Verifier.Score s = Verifier.evaluate(p, r.layout());
        if (s.total() != r.value()) {
            System.err.printf("BUG: solver value %d != verifier score %d%n", r.value(), s.total());
            System.exit(3);
        }
        printLayout(p, r.layout());
        printBreakdown(p, r.layout(), s);
        System.out.printf("%nTOTAL = %d  (proven optimal; %.1fs, %d rounds, %d items tracked, %d DP nodes)%n",
                s.total(), seconds, r.rounds(), r.trackedItems(), r.nodes());
    }

    private static void printLayout(Problem p, Verifier.Layout l) {
        int w = p.width();
        char[] grid = new char[w * p.height()];
        Arrays.fill(grid, '.');
        for (int[] b : p.blocked()) grid[b[1] * w + b[0]] = '#';
        for (int i = 0; i < l.itemCells().length; i++) {
            if (l.itemCells()[i] == null) continue;
            char label = i < ITEM_LABELS.length() ? ITEM_LABELS.charAt(i) : '*';
            for (int c : l.itemCells()[i]) grid[c] = label;
        }
        for (int j = 0; j < l.dominoes().size(); j++) {
            for (int c : l.dominoes().get(j)) grid[c] = (char) ('a' + j % 26);
        }
        System.out.println("\nlayout ('#' blocked, '.' empty, A.. items, a.. 1x2 pieces):");
        for (int y = 0; y < p.height(); y++) System.out.println("  " + new String(grid, y * w, w));
    }

    private static void printBreakdown(Problem p, Verifier.Layout l, Verifier.Score s) {
        int w = p.width();
        System.out.println("\nitems:");
        for (int i = 0; i < l.itemCells().length; i++) {
            Problem.Item it = p.items().get(i);
            String label = i < ITEM_LABELS.length() ? String.valueOf(ITEM_LABELS.charAt(i)) : "*";
            if (l.itemCells()[i] == null) {
                System.out.printf("  %s %-12s unused%n", label, it.name());
            } else {
                System.out.printf("  %s %-12s base=%d value=%d%n", label, it.name(), it.value(), s.itemValues()[i]);
            }
        }
        System.out.println("1x2 pieces:");
        for (int j = 0; j < l.dominoes().size(); j++) {
            int[] dm = l.dominoes().get(j);
            System.out.printf("  %c (%d,%d)-(%d,%d) value=%d%n", (char) ('a' + j % 26),
                    dm[0] % w, dm[0] / w, dm[1] % w, dm[1] / w, s.dominoValues()[j]);
        }
    }
}
