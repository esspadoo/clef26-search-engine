package unipd.se.model;

/**
 * Represents a basic query loaded from the dataset.
 * A query contains an identifier, its text and the identifier of the
 * relevant scientific paper.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class QueryDoc implements SearchQuery {
    public String index;
    public String text;
    public String pubkey;

    /**
     * Returns the identifier of the query.
     *
     * @return the query identifier
     */
    @Override
    public String getIndex() {
        return index;
    }

    /**
     * Returns the identifier of the relevant paper.
     *
     * @return the relevant paper identifier
     */
    @Override
    public String getPubkey() {
        return pubkey;
    }

    /**
     * Returns the original text of the query.
     *
     * @return the query text
     */
    @Override
    public String getSearchText() {
        return text;
    }
}