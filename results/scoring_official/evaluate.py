#script da eseguire per usare lo scorer ufficiale!

#serve generare un token di accesso hugging face con il vostro account che usate per accedere alla repo di huggin face di clef
#poi prima di eseguire il programma fare nel terminale: 
#export HF_TOKEN="hf_XXXXXXX"

#nella stessa cartella devono esserci scorer.py e il file reranked_results.json (o quello che si vuole, basta cambiare il percorso 
#qui sotto)


import json
from scorer import scorer

with open("reranked_results.json", "r") as f:
    reranker_results = json.load(f)

top5_preds = [
    [int(x) for x in reranker_results[str(i)][:5]]
    for i in sorted(int(k) for k in reranker_results.keys())
]

score = scorer(top5_preds, lang="en", split="train")
print(f"MRR@5: {score:.4f}")
