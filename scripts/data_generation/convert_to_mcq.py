import json
import random
from pathlib import Path

def load_wikidata_labels():
    cache_file = "data/wikidata_labels.json"
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)

GLOBAL_LABELS = load_wikidata_labels()

def get_entity_labels(uris, global_labels):
    ids = []
    if isinstance(uris, list):
        for uri in uris:
            if isinstance(uri, str): ids.append(uri.split("/")[-1])
            elif isinstance(uri, list) and uri and isinstance(uri[0], str): ids.append(uri[0].split("/")[-1])
    elif isinstance(uris, str):
        ids.append(uris.split("/")[-1])
    
    labels = []
    for eid in ids:
        val = global_labels.get(eid, {})
        label = val.get("label", eid) if isinstance(val, dict) else str(eid)
        labels.append(label)
    return ", ".join(labels) if labels else "Unknown"

def get_random_distractor(global_labels, exclude_texts):
    all_keys = list(global_labels.keys())
    for _ in range(100):
        k = random.choice(all_keys)
        val = global_labels.get(k, {})
        label = val.get("label", k) if isinstance(val, dict) else str(k)
        if label not in exclude_texts:
            return label
    return "Random Distractor"

def process_single_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        
    if not isinstance(data, dict): return
    
    if "clean" in data and "broken" in data:
        # Multi-variant file (perturbed)
        gt_text = get_entity_labels(data["clean"].get("answers", []), GLOBAL_LABELS)
        
        swapping_text = None
        if "swapping" in data and "swapped_answers" in data["swapping"]:
            swapping_text = get_entity_labels(data["swapping"]["swapped_answers"], GLOBAL_LABELS)
        elif "swapping" in data and "answers" in data["swapping"]:
            # Fallback if swapped answers replaced the original answers key
            swapping_text = get_entity_labels(data["swapping"]["answers"], GLOBAL_LABELS)
            
        if not swapping_text or swapping_text == gt_text:
            swapping_text = get_random_distractor(GLOBAL_LABELS, [gt_text])
            
        no_context_text = "Insufficient information to answer"
        distractor_text = get_random_distractor(GLOBAL_LABELS, [gt_text, swapping_text, no_context_text])
        
        options_list = [gt_text, swapping_text, no_context_text, distractor_text]
        random.shuffle(options_list)
        
        letters = ["A", "B", "C", "D"]
        options_dict = {letters[i]: options_list[i] for i in range(4)}
        options_text = "\n\nOptions:\n" + "\n".join([f"{k}. {v}" for k, v in options_dict.items()])
        
        # Assign correctness
        for variant, content in data.items():
            if not isinstance(content, dict): continue
            
            content["mcq_options_text"] = options_text
            content["mcq_options_dict"] = options_dict
            
            if variant in ["clean", "topological", "type_matching"]:
                correct_val = gt_text
            elif variant == "broken":
                correct_val = no_context_text
            elif variant == "swapping":
                correct_val = swapping_text
            else:
                correct_val = gt_text # default
                
            for k, v in options_dict.items():
                if v == correct_val:
                    content["mcq_correct_letter"] = k
                    break
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    elif "answers" in data:
        # Single clean file
        gt_text = get_entity_labels(data["answers"], GLOBAL_LABELS)
        swapping_text = get_random_distractor(GLOBAL_LABELS, [gt_text])
        no_context_text = "Insufficient information to answer"
        distractor_text = get_random_distractor(GLOBAL_LABELS, [gt_text, swapping_text, no_context_text])
        
        options_list = [gt_text, swapping_text, no_context_text, distractor_text]
        random.shuffle(options_list)
        
        letters = ["A", "B", "C", "D"]
        options_dict = {letters[i]: options_list[i] for i in range(4)}
        options_text = "\n\nOptions:\n" + "\n".join([f"{k}. {v}" for k, v in options_dict.items()])
        
        data["mcq_options_text"] = options_text
        data["mcq_options_dict"] = options_dict
        
        for k, v in options_dict.items():
            if v == gt_text:  # Default single file considers GT as correct
                data["mcq_correct_letter"] = k
                break
                
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

def process_dir(dir_name):
    path = Path(f"data/{dir_name}")
    if not path.exists(): return
    for file_path in path.glob("*.json"):
        process_single_file(file_path)
    print(f"Processed {dir_name}")

if __name__ == '__main__':
    process_dir("clean_subgraphs")
    process_dir("perturbed_subgraphs")
