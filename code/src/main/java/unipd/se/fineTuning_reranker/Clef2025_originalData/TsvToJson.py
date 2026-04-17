import csv
import json

data = []
with open('subtask4b_query_tweets_dev.tsv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        # Converti post_id e cord_uid in interi
        data.append({
            'index': int(row['post_id']),
            'text': row['tweet_text'],
            'pubkey': (row['cord_uid'])
        })

# Scrivi il JSON (senza ensure_ascii=False per mantenere i caratteri originali)
with open('subtask4b_query_tweets_dev.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=2, ensure_ascii=False)