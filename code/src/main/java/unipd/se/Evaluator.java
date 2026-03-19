package unipd.se;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.QueryDoc;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Evaluator for IR results.
 * Computes Recall@k, MAP, MRR, and nDCG@10.
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
        int hitAt1 = 0, hitAt5 = 0, hitAt10 = 0, hitAt100 = 0;
        double mrr = 0.0;
        double ndcgAt10 = 0.0;
        double map = 0.0;

        for (QueryDoc q : queries) {

            String qid = q.index;
            Set<String> goldSet = new HashSet<>();

            if (q.pubkey != null) {
                goldSet.add(q.pubkey);
            }

            List<String> ranked = results.getOrDefault(qid, Collections.emptyList());

            int rank = -1;

            // Find first relevant document
            for (int i = 0; i < ranked.size(); i++) {
                if (goldSet.contains(ranked.get(i))) {
                    rank = i + 1;
                    break;
                }
            }

            // Recall@k
            if (rank == 1) hitAt1++;
            if (rank > 0 && rank <= 5) hitAt5++;
            if (rank > 0 && rank <= 10) hitAt10++;
            if (rank > 0 && rank <= 100) hitAt100++;

            // MRR
            if (rank > 0) mrr += 1.0 / rank;

            // nDCG@10
            double dcg = 0.0;
            double idcg = 0.0;
            int k = Math.min(10, ranked.size());
            int relCount = goldSet.size();

            for (int i = 0; i < k; i++) {
                if (goldSet.contains(ranked.get(i))) {
                    dcg += 1.0 / log2(i + 2);
                }
            }

            for (int i = 0; i < Math.min(relCount, 10); i++) {
                idcg += 1.0 / log2(i + 2);
            }

            if (idcg > 0) {
                ndcgAt10 += dcg / idcg;
            }

            // MAP
            double avgPrecision = 0.0;
            int hitCount = 0;

            for (int i = 0; i < ranked.size(); i++) {
                if (goldSet.contains(ranked.get(i))) {
                    hitCount++;
                    avgPrecision += hitCount / (double) (i + 1);
                }
            }

            if (hitCount > 0) {
                map += avgPrecision / hitCount;
            }
        }

        double denom = total == 0 ? 1.0 : total;

        // Print
        System.out.println("Queries: " + total);
        System.out.printf("Recall@1: %.4f%n", hitAt1 / denom);
        System.out.printf("Recall@5: %.4f%n", hitAt5 / denom);
        System.out.printf("Recall@10: %.4f%n", hitAt10 / denom);
        System.out.printf("Recall@100: %.4f%n", hitAt100 / denom);
        System.out.printf("MRR: %.4f%n", mrr / denom);
        System.out.printf("MAP: %.4f%n", map / denom);
        System.out.printf("nDCG@10: %.4f%n", ndcgAt10 / denom);

        // Save JSON
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode root = mapper.createObjectNode();
        root.set("config", config);

        ObjectNode metrics = mapper.createObjectNode();
        metrics.put("queries", total);
        metrics.put("recall@1", hitAt1 / denom);
        metrics.put("recall@5", hitAt5 / denom);
        metrics.put("recall@10", hitAt10 / denom);
        metrics.put("recall@100", hitAt100 / denom);
        metrics.put("mrr", mrr / denom);
        metrics.put("map", map / denom);
        metrics.put("ndcg@10", ndcgAt10 / denom);

        root.set("metrics", metrics);

        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputFilePath), root);
        System.out.println("Evaluation saved to: " + outputFilePath);
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }
}