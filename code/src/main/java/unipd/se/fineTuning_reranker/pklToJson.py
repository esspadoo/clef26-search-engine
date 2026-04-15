import pickle
import json

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    else:
        return obj

input_file = "input.pkl"
output_file = "output.json"

with open(input_file, "rb") as f:
    data = pickle.load(f)

data = make_json_serializable(data)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Conversione completata!")