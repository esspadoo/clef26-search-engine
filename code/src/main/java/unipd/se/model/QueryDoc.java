package unipd.se.model;

public class QueryDoc {
    public String index;
    public String text;
    public String pubkey;

    public String getIndex() {
        return index;
    }

    public String getText() {
        return text;
    }

    public String getPubkey() {
        return pubkey;
    }

    @Override
    public String getSearchText() {
        return text;
    }
}