package unipd.se;

import unipd.se.model.QueryDoc;

import java.util.List;
import java.util.Map;

/**
 * Evaluator for retrieval results.
 * Computes Recall@k, MRR, and nDCG@10 for a set of queries.
 */
public final class Evaluator {

    private Evaluator() { }

    /**
     * Evaluate the retrieval results.
     *
     * @param results a map from query pubkey to ranked list of retrieved paper pubkeys
     * @param queries the list of gold standard queries
     */
    public static void evaluate(Map<String, List<String>> results, List<QueryDoc> queries) {
        int total = queries.size();
        int hitAt1 = 0, hitAt5 = 0, hitAt10 = 0, hitAt100 = 0;
        double mrr = 0.0, ndcgAt10 = 0.0;

        for (QueryDoc q : queries) {
            List<String> ranked = results.get(q.index);  // key = query index
            String gold = q.pubkey;                      // relevant paper

            int rank = -1;
            if (ranked != null) {
                for (int i = 0; i < ranked.size(); i++) {
                    if (gold.equals(ranked.get(i))) {
                        rank = i + 1;
                        break;
                    }
                }
            }

            if (rank == 1) hitAt1++;
            if (rank > 0 && rank <= 5) hitAt5++;
            if (rank > 0 && rank <= 10) {
                hitAt10++;
                ndcgAt10 += 1.0 / log2(rank + 1); // simple DCG for single relevant
            }
            if (rank > 0 && rank <= 100) hitAt100++;
            if (rank > 0) mrr += 1.0 / rank;
        }

        double denom = total == 0 ? 1.0 : total;
        System.out.println("Queries: " + total);
        System.out.printf("Recall@1: %.4f%n", hitAt1 / denom);
        System.out.printf("Recall@5: %.4f%n", hitAt5 / denom);
        System.out.printf("Recall@10: %.4f%n", hitAt10 / denom);
        System.out.printf("Recall@100: %.4f%n", hitAt100 / denom);
        System.out.printf("MRR: %.4f%n", mrr / denom);
        System.out.printf("nDCG@10: %.4f%n", ndcgAt10 / denom);
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }
}
