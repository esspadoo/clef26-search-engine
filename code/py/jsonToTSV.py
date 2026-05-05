import json

with open("collection_data.json") as f:
    data = json.load(f)

with open("corpus.tsv", "w", encoding="utf-8") as out:
    # header
    out.write("pubkey\ttitle\tabstract\tvenue\tauthors\n")

    for item in data:
        row = [
            str(item.get("pubkey", "")),
            item.get("title", "").replace("\t", " ").replace("\n", " "),
            item.get("abstract", "").replace("\t", " ").replace("\n", " "),
            item.get("venue", "").replace("\t", " ").replace("\n", " "),
            item.get("authors", "").replace("\t", " ").replace("\n", " "),
        ]
        out.write("\t".join(row) + "\n")