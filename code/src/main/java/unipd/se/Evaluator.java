package unipd.se;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.QueryDoc;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Evaluator for IR results.
 * Computes:
 *   - Recall@k        (k = 1, 5, 10, 100)
 *   - Precision@k     (k = 1, 5, 10)
 *   - F1@k            (k = 1)
 *   - MRR
 *   - MAP
 *   - nDCG@k          (k = 5, 10, 100)
 *   - Per-query stats (min, max, median for MRR and nDCG@10)
 *
 * Supports QueryDoc and subclasses (e.g., ExpandedQueryDoc).
 */
public final class Evaluator {

    private Evaluator() {}

    /**
     * Evaluate queries (works for both QueryDoc and subclasses).
     */
    public static void evaluate(
            Map<String, List<String>> results,
            List<? extends QueryDoc> queries,
            ObjectNode config,
            String outputFilePath
    ) throws IOException {

        int total = queries.size();

        // Recall@k
        int hitAt1 = 0, hitAt5 = 0, hitAt10 = 0, hitAt100 = 0;

        // Precision@k accumulators
        double precAt1 = 0.0, precAt5 = 0.0, precAt10 = 0.0;

        // Classic metrics
        double mrr      = 0.0;
        double map      = 0.0;

        // nDCG@k
        double ndcgAt5   = 0.0;
        double ndcgAt10  = 0.0;
        double ndcgAt100 = 0.0;

        // Per-query collections for stats
        List<Double> perQueryMrr    = new ArrayList<>(total);
        List<Double> perQueryNdcg10 = new ArrayList<>(total);

        for (QueryDoc q : queries) {

            String qid = q.index;
            Set<String> goldSet = new HashSet<>();
            if (q.pubkey != null) {
                goldSet.add(q.pubkey);
            }

            List<String> ranked = results.getOrDefault(qid, Collections.emptyList());
            int relCount = goldSet.size(); // number of known relevant docs

            // -------------------------------------------------------
            // Find first relevant document rank (1-based, -1 if none)
            // -------------------------------------------------------
            int rank = -1;
            for (int i = 0; i < ranked.size(); i++) {
                if (goldSet.contains(ranked.get(i))) {
                    rank = i + 1;
                    break;
                }
            }

            // -------------------------------------------------------
            // Recall@k  (binary: was the relevant doc found in top-k?)
            // -------------------------------------------------------
            if (rank == 1)                      hitAt1++;
            if (rank > 0 && rank <= 5)          hitAt5++;
            if (rank > 0 && rank <= 10)         hitAt10++;
            if (rank > 0 && rank <= 100)        hitAt100++;

            // -------------------------------------------------------
            // Precision@k = (# relevant in top-k) / k
            // -------------------------------------------------------
            precAt1  += precisionAtK(ranked, goldSet, 1);
            precAt5  += precisionAtK(ranked, goldSet, 5);
            precAt10 += precisionAtK(ranked, goldSet, 10);

            // -------------------------------------------------------
            // MRR = measures the average position of the first relevant result in a list of search results.
            // A higher MRR indicates that relevant items are found closer to the top of the list,
            // -------------------------------------------------------
            double qMrr = (rank > 0) ? 1.0 / rank : 0.0;
            mrr += qMrr;
            perQueryMrr.add(qMrr);

            // -------------------------------------------------------
            // nDCG@k = evaluate the effectiveness of search retrieval systems by measuring how well the results
            // are ranked in terms of relevance. It compares the actual ranking of results to an ideal ranking,
            // giving higher scores to results that are more relevant and appear earlier in the list,
            // thus helping to assess the quality of search algorithms
            // -------------------------------------------------------
            double qNdcg5   = ndcgAtK(ranked, goldSet, relCount, 5);
            double qNdcg10  = ndcgAtK(ranked, goldSet, relCount, 10);
            double qNdcg100 = ndcgAtK(ranked, goldSet, relCount, 100);

            ndcgAt5   += qNdcg5;
            ndcgAt10  += qNdcg10;
            ndcgAt100 += qNdcg100;

            perQueryNdcg10.add(qNdcg10);

            // -------------------------------------------------------
            // MAP  =  average precision over all retrieved positions
            //         divided by the total number of relevant docs
            //         (not just the ones founddef)
            // -------------------------------------------------------
            double avgPrecision = 0.0;
            int hitCount = 0;
            for (int i = 0; i < ranked.size(); i++) {
                if (goldSet.contains(ranked.get(i))) {
                    hitCount++;
                    avgPrecision += hitCount / (double) (i + 1);
                }
            }
            // Denominator is relCount (total relevant), not hitCount
            if (relCount > 0) {
                map += avgPrecision / relCount;
            }
        }

        double denom = total == 0 ? 1.0 : total;

        // -------------------------------------------------------
        // F1@1: it evaluates the balance between precision and recall for these top results,
        // helping to assess the effectiveness of the search algorithm in retrieving relevant documents
        // -------------------------------------------------------
        double avgRecall1 = hitAt1 / denom;
        double avgPrec1   = precAt1 / denom;
        double f1At1 = (avgPrec1 + avgRecall1 > 0)
                ? 2.0 * avgPrec1 * avgRecall1 / (avgPrec1 + avgRecall1)
                : 0.0;

        // -------------------------------------------------------
        // Per-query stats
        // -------------------------------------------------------
        double medianMrr    = median(perQueryMrr);
        double medianNdcg10 = median(perQueryNdcg10);
        double minMrr       = perQueryMrr.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxMrr       = perQueryMrr.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        double minNdcg10    = perQueryNdcg10.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxNdcg10    = perQueryNdcg10.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);

        // -------------------------------------------------------
        // Console output
        // -------------------------------------------------------
        System.out.println("=== Evaluation Results ===");
        System.out.println("Queries:      " + total);
        System.out.println();
        System.out.printf("Recall@1:   %.4f%n", hitAt1   / denom);
        System.out.printf("Recall@5:   %.4f%n", hitAt5   / denom);
        System.out.printf("Recall@10:  %.4f%n", hitAt10  / denom);
        System.out.printf("Recall@100: %.4f%n", hitAt100 / denom);
        System.out.printf("P@1:        %.4f%n", precAt1  / denom);
        System.out.printf("P@5:        %.4f%n", precAt5  / denom);
        System.out.printf("P@10:       %.4f%n", precAt10 / denom);
        System.out.printf("F1@1:      %.4f%n", f1At1);
        System.out.printf("MRR:        %.4f  (min=%.4f, median=%.4f, max=%.4f)%n",
                mrr / denom, minMrr, medianMrr, maxMrr);
        System.out.printf("MAP:        %.4f%n", map / denom);
        System.out.printf("nDCG@5:     %.4f%n", ndcgAt5   / denom);
        System.out.printf("nDCG@10:    %.4f  (min=%.4f, median=%.4f, max=%.4f)%n",
                ndcgAt10 / denom, minNdcg10, medianNdcg10, maxNdcg10);
        System.out.printf("nDCG@100:   %.4f%n", ndcgAt100 / denom);

        // -------------------------------------------------------
        // JSON output
        // -------------------------------------------------------
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode root = mapper.createObjectNode();
        root.set("config", config);

        ObjectNode metrics = mapper.createObjectNode();
        metrics.put("queries",     total);

        // Recall
        metrics.put("recall@1",    hitAt1   / denom);
        metrics.put("recall@5",    hitAt5   / denom);
        metrics.put("recall@10",   hitAt10  / denom);
        metrics.put("recall@100",  hitAt100 / denom);

        // Precision
        metrics.put("precision@1",    precAt1  / denom);
        metrics.put("precision@5",    precAt5  / denom);
        metrics.put("precision@10",   precAt10 / denom);

        // F1
        metrics.put("f1@1", f1At1);

        // Classic
        metrics.put("mrr",  mrr / denom);
        metrics.put("map",  map / denom);

        // nDCG
        metrics.put("ndcg@5",   ndcgAt5   / denom);
        metrics.put("ndcg@10",  ndcgAt10  / denom);
        metrics.put("ndcg@100", ndcgAt100 / denom);

        // Per-query stats
        ObjectNode stats = mapper.createObjectNode();

        ObjectNode mrrStats = mapper.createObjectNode();
        mrrStats.put("min",    minMrr);
        mrrStats.put("median", medianMrr);
        mrrStats.put("max",    maxMrr);
        stats.set("mrr", mrrStats);

        ObjectNode ndcg10Stats = mapper.createObjectNode();
        ndcg10Stats.put("min",    minNdcg10);
        ndcg10Stats.put("median", medianNdcg10);
        ndcg10Stats.put("max",    maxNdcg10);
        stats.set("ndcg@10", ndcg10Stats);

        metrics.set("per_query_stats", stats);
        root.set("metrics", metrics);

        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputFilePath), root);
        System.out.println("\nEvaluation saved to: " + outputFilePath);
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /**
     * Precision@k = (# relevant in top-k) / k
     */
    private static double precisionAtK(List<String> ranked, Set<String> goldSet, int k) {
        if (ranked.isEmpty() || k == 0) return 0.0;
        int limit = Math.min(k, ranked.size());
        int hits = 0;
        for (int i = 0; i < limit; i++) {
            if (goldSet.contains(ranked.get(i))) hits++;
        }
        return hits / (double) k;
    }

    /**
     * nDCG@k using binary relevance.
     *
     * @param relCount total number of known relevant documents (for IDCG)
     */
    private static double ndcgAtK(List<String> ranked, Set<String> goldSet, int relCount, int k) {
        int limit = Math.min(k, ranked.size());
        double dcg  = 0.0;
        double idcg = 0.0;

        for (int i = 0; i < limit; i++) {
            if (goldSet.contains(ranked.get(i))) {
                dcg += 1.0 / log2(i + 2);  // log2(rank + 1), rank is 1-based
            }
        }

        int idealHits = Math.min(relCount, k);
        for (int i = 0; i < idealHits; i++) {
            idcg += 1.0 / log2(i + 2);
        }

        return (idcg > 0) ? dcg / idcg : 0.0;
    }

    /**
     * Median of a list (does not modify the original).
     */
    private static double median(List<Double> values) {
        if (values.isEmpty()) return 0.0;
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int n = sorted.size();
        if (n % 2 == 1) return sorted.get(n / 2);
        return (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }
}