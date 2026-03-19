
package unipd.se.model;

public class ExpandedQueryDoc implements SearchQuery {
    public String index;
    public String original;
    public String keywords;
    public String expanded;
    public String pubkey;

    public String getIndex() {
        return index;
    }

    public String getOriginal() {
        return original;
    }

    public String getNormalized() {
        return normalized;
    }

    public String getExpanded() {
        return expanded;
    }

    public String getPubkey() {
        return pubkey;
    }

    @Override
    public String getSearchText() {
        return expanded;
    }
}
