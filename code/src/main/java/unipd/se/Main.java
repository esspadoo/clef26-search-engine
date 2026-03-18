package unipd.se;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.Paper;
import unipd.se.model.QueryDoc;
import org.apache.lucene.store.Directory;

import java.util.List;
import java.util.Map;

/**
 * Entry point for the Information Retrieval pipeline.
 * <p>
 * This class performs the following steps:
 * <ol>
 *     <li>Load papers and queries from JSON files.</li>
 *     <li>Build a Lucene index of the papers.</li>
 *     <li>Search the index for each query.</li>
 *     <li>Print the top-matching paper IDs for each query.</li>
 * </ol>
 * </p>
 */
public class Main {

    /**
     * Main method to run the IR pipeline.
     *
     * @param args optional command-line arguments:
     *             args[0] = path to papers JSON (default: data/collection_data.json)
     *             args[1] = path to queries JSON (default: data/en_dev.json)
     */
    public static void main(String[] args) {

        String papersPath = args.length > 0 ? args[0] : "code/data/collection_data.json";
        String queriesPath = args.length > 1 ? args[1] : "code/data/en_train.json";

        try {
            // 1. Load data
            List<Paper> papers = DataLoader.loadPapers(papersPath);
            List<QueryDoc> queries = DataLoader.loadQueries(queriesPath);

            // 2. Build index (persistent)
            Directory index = Indexer.buildIndex(papers);

            // 3. Search
            Map<String, List<String>> results = Searcher.search(index, queries);

            // 4. Print results
            for (Map.Entry<String, List<String>> entry : results.entrySet()) {
                System.out.println("Query: " + entry.getKey());
                System.out.println("Top docs: " + entry.getValue());
                System.out.println();
            }

            // Build configuration info
            ObjectMapper mapper = new ObjectMapper();
            ObjectNode config = mapper.createObjectNode();
            config.put("analyzer", "StandardAnalyzer");
            config.put("query_parser", "SimpleQueryParser");
            config.put("top_n", 50);
            config.put("title_boost", 2.0);
            config.put("similarity", "BM25");
            config.putPOJO("fields", new String[]{"title","abstract"});

            // Evaluate and save to JSON
            Evaluator.evaluate(results, queries, config, "results/evaluation_results.json");

        } catch (Exception e) {
            System.err.println("Error running IR pipeline: " + e.getMessage());
            e.printStackTrace();
        }
    }
}