package unipd.se;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.Paper;
import unipd.se.model.QueryDoc;
import org.apache.lucene.store.Directory;

import java.io.File;
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
        System.out.println("Starting retrieval");
        String papersPath = args.length > 0 ? args[0] : "code/data/collection_data.json";
        String queriesPath = args.length > 1 ? args[1] : "code/data/expanded_queries.json";

        try {
            // 1. Load data
            List<Paper> papers = DataLoader.loadPapers(papersPath);
            //List<QueryDoc> queries = DataLoader.loadQueries(queriesPath, QueryDoc[].class);
            List<ExpandedQueryDoc> queries = DataLoader.loadQueries(queriesPath, ExpandedQueryDoc[].class);

            // 2. Build index (persistent)
            Directory index = Indexer.buildIndex(papers);

            // 3. Search
            Map<String, List<String>> results = Searcher.search(index, queries, 1.0f, 100);

            // 4. Print results
            for (Map.Entry<String, List<String>> entry : results.entrySet()) {
                System.out.println("Query: " + entry.getKey());
                System.out.println("Top docs: " + entry.getValue());
                System.out.println();
            }

            // Build configuration info
            ObjectMapper mapper = new ObjectMapper();
            ObjectNode config = mapper.createObjectNode();
            config.put("analyzer", "MyCustomAnalyzer");
            config.put("query_parser", "SBERT");
            config.put("top_n", 100);
            config.put("title_boost", 1.0);
            config.put("similarity", "BM25");
            config.putPOJO("fields", new String[]{"title","abstract"});

            // Evaluate and save to JSON
            String basePath = "results/evaluation_results";
            String extension = ".json";

            File file = new File(basePath + extension);
            int counter = 1;

            // Keep incrementing until we find a filename that doesn't exist
            while (file.exists()) {
                file = new File(basePath + "_" + counter + extension);
                counter++;
            }

            Evaluator.evaluate(results, queries, config, file.getPath());

        } catch (Exception e) {
            System.err.println("Error running IR pipeline: " + e.getMessage());
            e.printStackTrace();
        }
    }
}