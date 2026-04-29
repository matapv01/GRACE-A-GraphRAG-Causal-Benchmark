import json
import random
import re
from pathlib import Path

def load_wikidata_labels():
    cache_file = "data/wikidata_labels.json"
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)

GLOBAL_LABELS = load_wikidata_labels()


CURRENT_BENCHMARK_YEAR = 2026
ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:Z)?)?(?!\d)")
DMY_SLASH_RE = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")
DMY_DASH_RE = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)")
YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20[0-1]\d|202[0-5])(?!\d)")


def load_lcquad_questions_map():
    p = Path("data/lcquad_test.json")
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    qmap = {}
    for row in rows:
        uid = str(row.get("uid", ""))
        text = (row.get("question") or row.get("NNQT_question") or "").strip()
        if uid:
            qmap[uid] = text
    return qmap


LCQUAD_QUESTION_MAP = load_lcquad_questions_map()


def has_past_timestamp(text: str) -> bool:
    if not text:
        return False

    years = []
    spans = []
    for pat, yg in ((ISO_DATE_RE, 1), (DMY_SLASH_RE, 3), (DMY_DASH_RE, 3)):
        for m in pat.finditer(text):
            y = int(m.group(yg))
            if y < CURRENT_BENCHMARK_YEAR:
                years.append(y)
                spans.append(m.span())

    def in_spans(i):
        return any(a <= i < b for a, b in spans)

    for m in YEAR_RE.finditer(text):
        if in_spans(m.start()):
            continue
        y = int(m.group(1))
        if y < CURRENT_BENCHMARK_YEAR:
            years.append(y)

    return len(years) > 0

def get_entity_labels(uris, global_labels):
    if isinstance(uris, list):
        # If it's a single entity (swapping), return just its label
        if len(uris) == 1:
            eid = uris[0].split("/")[-1]
            val = global_labels.get(eid, {})
            label = val.get("label", eid) if isinstance(val, dict) else str(eid)
            return label
        else:
            ids = []
            for uri in uris:
                if isinstance(uri, str): ids.append(uri.split("/")[-1])
                elif isinstance(uri, list) and uri and isinstance(uri[0], str): ids.append(uri[0].split("/")[-1])
            labels = []
            for eid in ids:
                val = global_labels.get(eid, {})
                label = val.get("label", eid) if isinstance(val, dict) else str(eid)
                labels.append(label)
            return ", ".join(labels) if labels else "Unknown"
    elif isinstance(uris, str):
        eid = uris.split("/")[-1]
        val = global_labels.get(eid, {})
        label = val.get("label", eid) if isinstance(val, dict) else str(eid)
        return label
    return "Unknown"

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
        qid = str(data.get("clean", {}).get("question_id", file_path.stem))
        question_text = LCQUAD_QUESTION_MAP.get(qid, "")
        is_timestamp_group = has_past_timestamp(question_text)
        data["question_time_group"] = "timestamp" if is_timestamp_group else "non_timestamp"

        # Nếu là nhóm timestamp, đồng bộ answers cho tất cả variants
        if is_timestamp_group:
            ref_answers = data["clean"].get("answers", [])
            for v in data.values():
                if isinstance(v, dict):
                    v["answers"] = list(ref_answers)

        gt_text = get_entity_labels(data["clean"].get("answers", []), GLOBAL_LABELS)
        swapping_text = get_entity_labels(data["swapping"].get("answers", []), GLOBAL_LABELS)
        no_context_text = "Insufficient information to answer"
        opts = [gt_text, swapping_text, no_context_text]
        opts_final = []
        seen = set()
        for o in opts:
            if o not in seen:
                opts_final.append(o)
                seen.add(o)
        while len(opts_final) < 4:
            distractor = get_random_distractor(GLOBAL_LABELS, opts_final)
            if distractor not in opts_final:
                opts_final.append(distractor)
        random.shuffle(opts_final)
        letters = ["A", "B", "C", "D"]
        options_dict = {letters[i]: opts_final[i] for i in range(4)}
        options_text = "\n\nOptions:\n" + "\n".join([f"{k}. {v}" for k, v in options_dict.items()])

        for variant, content in data.items():
            if not isinstance(content, dict): continue
            content["mcq_options_text"] = options_text
            content["mcq_options_dict"] = options_dict
            content["mcq_scoring_group"] = "timestamp" if is_timestamp_group else "default"

            if is_timestamp_group:
                correct_val = gt_text
            else:
                if variant in ["clean", "topological", "type_matching"]:
                    correct_val = gt_text
                elif variant == "broken":
                    correct_val = no_context_text
                elif variant == "swapping":
                    correct_val = swapping_text
                else:
                    correct_val = gt_text

            found = False
            for k, v in options_dict.items():
                if v == correct_val:
                    content["mcq_correct_letter"] = k
                    found = True
                    break
            if not found:
                content["mcq_correct_letter"] = "None"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    elif "answers" in data:
        # Single clean file
        qid = str(data.get("question_id", file_path.stem))
        question_text = LCQUAD_QUESTION_MAP.get(qid, "")
        is_timestamp_group = has_past_timestamp(question_text)
        data["question_time_group"] = "timestamp" if is_timestamp_group else "non_timestamp"

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
    process_dir("test_clean_subgraphs")
    process_dir("test_perturbed_subgraphs")
