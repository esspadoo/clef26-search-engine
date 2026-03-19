
package unipd.se.model;

public class ExpandedQueryDoc extends QueryDoc {

    public String original;
    public String keywords;
    public String expanded;

    public String getOriginal() {
        return original;
    }

    public String getExpanded() {
        return expanded;
    }

    @Override
    public String getSearchText() {
        return expanded;
    }
}
