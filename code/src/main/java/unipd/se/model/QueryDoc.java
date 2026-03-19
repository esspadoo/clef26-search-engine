package unipd.se.model;

public class QueryDoc implements SearchQuery {
    public String index;
    public String text;
    public String pubkey;

    public String getIndex() {
        return index;
    }

    public String getPubkey() {
        return pubkey;
    }

    @Override
    public String getSearchText() {
        return text;
    }
}