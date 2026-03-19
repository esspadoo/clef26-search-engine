package unipd.se;

import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.QueryDoc;
import unipd.se.model.SearchQuery;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.*;

/**
 * Utility class for performing searches on a Lucene index of papers.
 * <p>
 * Each query is searched against the index, and the top N matching
 * papers are returned for each query.
 * </p>
 */
public class Searcher {

    /** Shared custom analyzer for parsing queries. */
    private static final MyEnglishAnalyzer ANALYZER = new MyEnglishAnalyzer();

    /**
     * Searches the Lucene index for each standard query and returns the top results.
     *
     * @param dir the Lucene {@link Directory} containing the index
     * @param queries the list of {@link QueryDoc} objects to search for
     * @return a map from query index to a list of top-matching paper pubkeys
     * @throws IOException if an I/O error occurs reading the index
     * @throws ParseException if a query cannot be parsed
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<QueryDoc> queries
    ) throws IOException, ParseException {
        return searchInternal(dir, queries, 2.0f);
    }

    /**
     * Searches the Lucene index for each expanded query and returns the top results.
     *
     * @param dir the Lucene {@link Directory} containing the index
     * @param queries the list of {@link ExpandedQueryDoc} objects to search for
     * @return a map from query index to a list of top-matching paper pubkeys
     * @throws IOException if an I/O error occurs reading the index
     * @throws ParseException if a query cannot be parsed
     */
    public static Map<String, List<String>> searchExpanded(
            Directory dir,
            List<ExpandedQueryDoc> queries
    ) throws IOException, ParseException {
        return searchInternal(dir, queries, 3.0f);
    }

    /**
     * Internal generic search method shared by both standard and expanded queries.
     *
     * @param dir the Lucene {@link Directory} containing the index
     * @param queries the list of queries implementing {@link SearchQuery}
     * @param titleBoost boost applied to the title field
     * @param <T> query type implementing {@link SearchQuery}
     * @return a map from query index to a list of top-matching paper pubkeys
     * @throws IOException if an I/O error occurs reading the index
     * @throws ParseException if a query cannot be parsed
     */
    private static <T extends SearchQuery> Map<String, List<String>> searchInternal(
            Directory dir,
            List<T> queries,
            float titleBoost
    ) throws IOException, ParseException {

        Map<String, List<String>> results = new HashMap<>();

        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            searcher.setSimilarity(new BM25Similarity());

            Map<String, Float> fields = new HashMap<>();
            fields.put("title", titleBoost);
            fields.put("abstract", 1.0f);

            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);

            for (T q : queries) {
                Query query = parser.parse(QueryParser.escape(q.getSearchText()));
                TopDocs topDocs = searcher.search(query, 100);

                List<String> topIds = new ArrayList<>();
                for (ScoreDoc sd : topDocs.scoreDocs) {
                    Document doc = searcher.doc(sd.doc);
                    topIds.add(doc.get("pubkey"));
                }

                results.put(q.getIndex(), topIds);
            }
        }

        return results;
    }
}