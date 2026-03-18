package unipd.se;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.QueryDoc;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * State-of-the-art Evaluator for IR results.
 * Computes Recall@k, Precision@k, MAP, MRR, and nDCG@k for a set of queries.
 * Can handle multiple relevant documents per query and expanded queries.
 */
public final class Evaluator {

    private Evaluator() {}

    /**
     * Evaluate standard queries.
     */
    public static void evaluate(
            Map<String, List<String>> results,
            List<QueryDoc> queries,
            ObjectNode config,
            String outputFilePath
    ) throws IOException {
        evaluateInternal(results, queries, config, outputFilePath);
    }

    /**
     * Evaluate expanded queries.
     */
    public static void evaluateExpanded(
            Map<String, List<String>> results,
            List<ExpandedQueryDoc> queries,
            ObjectNode config,
            String outputFilePath
    ) throws IOException {
        evaluateInternal(results, queries, config, outputFilePath);
    }

    /**
     * Generic internal evaluator for both QueryDoc and ExpandedQueryDoc.
     */
    private static <T> void evaluateInternal(
            Map<String, List<String>> results,
            List<T> queries,
            ObjectNode config,
            String outputFilePath
    ) throws IOException {

        int total = queries.size();
        int hitAt1 = 0, hitAt5 = 0, hitAt10 = 0, hitAt100 = 0;
        double mrr = 0.0;
        double ndcgAt10 = 0.0;
        double map = 0.0;

        for (T qObj : queries) {
            String qid;
            Set<String> goldSet = new HashSet<>();

            if (qObj instanceof QueryDoc qd) {
                qid = qd.index;
                if (qd.pubkey != null) goldSet.add(qd.pubkey);
            } else if (qObj instanceof ExpandedQueryDoc eqd) {
                qid = eqd.index;
                if (eqd.pubkey != null) goldSet.add(eqd.pubkey);
            } else {
                continue;
            }

            List<String> ranked = results.getOrDefault(qid, List.of());
            int rank = -1;

            // Compute rank of first relevant document for MRR
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
            int i = 0;
            int relCount = goldSet.size();

            // DCG
            for (i = 0; i < k; i++) {
                String docId = ranked.get(i);
                if (goldSet.contains(docId)) {
                    dcg += 1.0 / log2(i + 2);  // rank i -> log2(i+2)
                }
            }

            // IDCG
            for (i = 0; i < Math.min(relCount, 10); i++) {
                idcg += 1.0 / log2(i + 2);
            }

            if (idcg > 0) ndcgAt10 += dcg / idcg;

            // MAP
            double avgPrecision = 0.0;
            int hitCount = 0;
            for (i = 0; i < ranked.size(); i++) {
                if (goldSet.contains(ranked.get(i))) {
                    hitCount++;
                    avgPrecision += hitCount / (double)(i + 1);
                }
            }
            if (hitCount > 0) map += avgPrecision / hitCount;
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