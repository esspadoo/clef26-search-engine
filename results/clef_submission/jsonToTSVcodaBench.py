import json

# Load the JSON file to convert to TSV format for submission to codaBench
with open('reranked_results_nemotronFTAarsen20252026_Lora_topk1000_onbiEncoderTopk1000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create the TSV file
with open('en_predictions.tsv', 'w', encoding='utf-8') as f:
    # Write the header
    f.write('index\tpreds\n')
    
    # For each document (key = post index, value = list of predictions)
    for index, predictions in data.items():
        # Take only the first 5 values
        top5 = predictions[:5]
        
        # Convert the list to string in the format [pred1, pred2, ...]
        preds_string = '[' + ', '.join(top5) + ']'
        
        # Write the row
        f.write(f'{index}\t{preds_string}\n')