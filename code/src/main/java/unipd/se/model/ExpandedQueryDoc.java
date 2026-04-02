package unipd.se.model;

/**
 * Represents a query enriched with additional information generated during
 * the query expansion phase.
 * <p>
 * This class extends {@link QueryDoc} by storing the original query text,
 * extracted keywords, and the expanded query text used for retrieval.
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class ExpandedQueryDoc extends QueryDoc {

    /**
     * Original text of the query before expansion.
     */
    public String original;

    /**
     * Keywords extracted from the original query.
     */
    public String keywords;
    /**
     * Expanded version of the query used to improve retrieval.
     */
    public String expanded;

    /**
     * Returns the original, non-expanded query text.
     *
     * @return the original query text
     */
    public String getOriginal() {
        return original;
    }

    /**
     * Returns the expanded query text.
     *
     * @return the expanded query text
     */
    public String getExpanded() {
        return expanded;
    }

    /**
     * Returns the text that must be used by the search engine.
     * <p>
     * In this implementation, the search text corresponds to the
     * {@code expanded} query.
     * </p>
     *
     * @return the expanded query text used for retrieval
     */
    @Override
    public String getSearchText() {
        return expanded;
    }
}