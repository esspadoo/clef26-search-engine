package unipd.se.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Paper {
    public String pubkey;
    public String title;
    @JsonProperty("abstract")
    public String abstractText;
    public String venue;
    public String authors;
}
