package unipd.se;

import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.QueryDoc;
import unipd.se.model.ExpandedQueryDoc;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

/**
 * Utility class for performing searches on a Lucene index of papers.
 * <p>
 * Supports both {@link QueryDoc} and its subclass {@link ExpandedQueryDoc}.
 * Uses BM25 similarity and a weighted multi-field query (title + abstract).
 * </p>
 * <p>
 * When an {@link ExpandedQueryDoc} is provided, a {@link BooleanQuery} is
 * built with two SHOULD clauses:
 * </p>
 * <ul>
 *   <li>The <b>original</b> query text, boosted at 2.0 — primary signal.</li>
 *   <li>The <b>expansion-only</b> terms, boosted at 1.0 — secondary signal.</li>
 * </ul>
 * <p>
 * This approach works transparently with both the old expansion format
 * (separate {@code expansion} field) and the new format (concatenated
 * {@code expanded} field), thanks to {@link ExpandedQueryDoc#getExpansionOnly()}.
 * </p>
 */
public class SearcherV3 {

    /** Shared custom analyzer for parsing queries. */
    private static final MyEnglishAnalyzer_TEX ANALYZER = new MyEnglishAnalyzer_TEX();

    /** Regex for extracting venue filter from query text. */
    private static final Pattern VENUE_PATTERN =
            Pattern.compile("\\bvenue:\\s*([^\\s]+)");

    /** Regex for extracting author filter from query text. */
    private static final Pattern AUTHOR_PATTERN =
            Pattern.compile("\\bauthor:\\s*([^\\s]+(?:\\s+[^\\s]+)*)");

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Search the index with a configurable title boost.
     * <p>
     * Queries are processed in parallel on a dedicated {@link ForkJoinPool}.
     * {@link IndexSearcher} is thread-safe for concurrent reads (guaranteed
     * by Lucene).
     * </p>
     *
     * @param dir        the Lucene index directory
     * @param queries    list of queries ({@link QueryDoc} or subclasses)
     * @param titleBoost boost applied to the title field
     * @param topK       number of top documents to retrieve per query
     * @return map from query index → ranked list of pubkeys
     * @throws IOException if the index cannot be opened or if the search fails
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {

        // Field weights (shared, immutable after construction)
        Map<String, Float> fields = new HashMap<>();
        fields.put("title",    titleBoost);
        fields.put("abstract", 1.0f);

        // IndexReader and IndexSearcher are opened once and shared across threads.
        // DirectoryReader is thread-safe for concurrent reads.
        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            int cores = Runtime.getRuntime().availableProcessors();

            try (ForkJoinPool pool = new ForkJoinPool(cores)) {
                return pool.submit(() ->
                        queries.parallelStream().collect(Collectors.toMap(
                                q -> q.index,
                                q -> searchSingle(q, searcher, fields, topK),
                                (a, b) -> a,
                                LinkedHashMap::new   // preserve insertion order
                        ))
                ).get();

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Search interrupted", e);
            } catch (ExecutionException e) {
                throw new IOException("Search execution failed", e.getCause());
            }
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Execute a single query against the searcher and return the ranked pubkeys.
     * <p>
     * A new {@link SimpleQueryParser} is created per invocation because the
     * parser is not thread-safe.
     * </p>
     *
     * @param q        the query document
     * @param searcher the shared index searcher
     * @param fields   field-to-boost map
     * @param topK     number of top documents to retrieve
     * @return ranked list of pubkeys, empty list on failure
     */
    private static List<String> searchSingle(
            QueryDoc q,
            IndexSearcher searcher,
            Map<String, Float> fields,
            int topK
    ) {
        String text = q.getSearchText();
        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }

        try {
            // SimpleQueryParser is not thread-safe: create a local instance per call
            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);

            Query query;
            Map<String, String> filters = new HashMap<>();

            if (q instanceof ExpandedQueryDoc eq) {
                query = buildExpandedQuery(eq, parser, filters);
            } else {
                query = buildSimpleQuery(text, parser, filters);
            }

            // Apply field filters (venue, author) as MUST clauses if present
            query = applyFilters(query, filters);

            TopDocs topDocs = searcher.search(query, topK);

            List<String> topIds = new ArrayList<>(topDocs.scoreDocs.length);
            for (ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(sd.doc);
                topIds.add(doc.get("pubkey"));
            }
            return topIds;

        } catch (IOException e) {
            System.err.println("Search failed for query " + q.index + ": " + e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Builds a {@link BooleanQuery} for an {@link ExpandedQueryDoc}.
     * <p>
     * Two SHOULD clauses are created:
     * </p>
     * <ul>
     *   <li>Original query text — boost 2.0 (primary signal)</li>
     *   <li>Expansion-only terms — boost 1.0 (secondary signal)</li>
     * </ul>
     * <p>
     * If either part is missing, the available part is used alone without
     * boost wrapping. The expansion-only terms are obtained via
     * {@link ExpandedQueryDoc#getExpansionOnly()}, which handles both the old
     * format ({@code expansion} field) and the new format ({@code expanded}
     * field containing the concatenation).
     * </p>
     *
     * @param eq      the expanded query document
     * @param parser  the query parser (already local to this thread)
     * @param filters mutable map populated with any venue/author filters found
     *                in the original text
     * @return the built Lucene query
     */
    private static Query buildExpandedQuery(
            ExpandedQueryDoc eq,
            SimpleQueryParser parser,
            Map<String, String> filters
    ) {
        String origSearch    = eq.getOriginal();
        String expansionOnly = eq.getExpansionOnly();

        // Estrai filtri dall'original
        if (origSearch != null && !origSearch.isBlank()) {
            origSearch = extractFilters(origSearch, filters);
        }

        // 🔥 CONCATENAZIONE
        String combined = "";

        if (origSearch != null && !origSearch.isBlank()) {
            combined += origSearch;
        }

        if (expansionOnly != null && !expansionOnly.isBlank()) {
            combined += (combined.isEmpty() ? "" : " ") + expansionOnly;
        }

        // fallback di sicurezza
        if (combined.isBlank()) {
            combined = eq.getSearchText();
        }

        return parser.parse(combined);
    }



    /**
     * Builds a simple query for a plain {@link QueryDoc}.
     *
     * @param text    raw search text from the query
     * @param parser  the query parser
     * @param filters mutable map populated with any venue/author filters found
     * @return the built Lucene query
     */
    private static Query buildSimpleQuery(
            String text,
            SimpleQueryParser parser,
            Map<String, String> filters
    ) {
        String searchText = extractFilters(text, filters);
        return parser.parse(searchText);
    }

    /**
     * Extracts {@code venue:X} and {@code author:X} tokens from the query
     * text, stores their values in {@code filters}, and returns the cleaned
     * text with those tokens removed.
     *
     * @param text    input text possibly containing filter tokens
     * @param filters mutable map to populate with extracted filter values
     * @return the input text with filter tokens removed
     */
    private static String extractFilters(String text, Map<String, String> filters) {
        Matcher venueM = VENUE_PATTERN.matcher(text);
        if (venueM.find()) {
            filters.put("venue", venueM.group(1));
            text = text.replaceFirst("\\bvenue:\\s*[^\\s]+", "").trim();
        }

        Matcher authorM = AUTHOR_PATTERN.matcher(text);
        if (authorM.find()) {
            filters.put("authors", authorM.group(1));
            text = text.replaceFirst("\\bauthor:\\s*[^\\s]+(?:\\s+[^\\s]+)*", "").trim();
        }

        return text;
    }

    /**
     * Wraps the given query in a {@link BooleanQuery} with MUST clauses for
     * any field filters present. If the filter map is empty, the original
     * query is returned unchanged.
     *
     * @param query   the base query
     * @param filters map of field name → required term value
     * @return the original query, or a filtered {@link BooleanQuery}
     */
    private static Query applyFilters(Query query, Map<String, String> filters) {
        if (filters.isEmpty()) {
            return query;
        }
        BooleanQuery.Builder bq = new BooleanQuery.Builder();
        bq.add(query, BooleanClause.Occur.MUST);
        for (Map.Entry<String, String> f : filters.entrySet()) {
            bq.add(new TermQuery(new Term(f.getKey(), f.getValue())),
                    BooleanClause.Occur.MUST);
        }
        return bq.build();
    }
}