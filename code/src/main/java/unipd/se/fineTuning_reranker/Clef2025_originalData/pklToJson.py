import pandas as pd
import json

# Carica il pickle
df = pd.read_pickle("/home/fabio/WorkInProgress/SearchEngines/Homework_SE/seupd2526-retrix/code/src/main/java/unipd/se/fineTuning_reranker/Clef2025_originalData/subtask4b_collection_data.pkl")

df_renamed = df.rename(columns={
    'cord_uid': 'pubkey', # rinominazione dei campi: 'old_name1': 'new_name1',
})

# Salva come JSON con i nuovi nomi
df_renamed.to_json('collection_data2025.json', orient='records', indent=4)

