# Centralized Prompts and Instructions for Benchmark

BENCHMARK_MULTIPLE_CHOICE_INSTRUCTION = (
    "\n\n--- CRITICAL INSTRUCTION FOR BENCHMARK ---"
    "\nThis is a multiple-choice question. You must output ONLY the exact letter corresponding to the correct answer (e.g., 'A', 'B', 'C', or 'D')."
    "\nDo NOT include any explanations, full sentences, or markdown formatting. Output a single character."
    "\nCRITICAL: If the provided knowledge context is entirely isolated (e.g. contains 'Q_ISOLATED', 'Q_DUMMY', or random isolated numbers with NO logical connection), or if you logically CANNOT find the answer based strictly on the text, you MUST output EXACTLY: 'None'"
)

import random

def generate_abcd_options(ground_truth_uris, global_labels):
    """
    Sinh đáp án trắc nghiệm bằng cách lấy label của Entities từ ground_truths.
    Sau đó tạo thêm 3 option giả từ DB (global_labels).
    Trả về: text (chuỗi ABCD đính kèm vào câu hỏi), correct_letter (A, B, C, hoặc D)
    """
    gt_ids = []
    if isinstance(ground_truth_uris, list):
        for uri in ground_truth_uris:
            if isinstance(uri, str):
                gt_ids.append(uri.split("/")[-1])
            elif isinstance(uri, list) and uri and isinstance(uri[0], str):
                gt_ids.append(uri[0].split("/")[-1])
    elif isinstance(ground_truth_uris, str):
        gt_ids.append(ground_truth_uris.split("/")[-1])
            
    gt_labels = []
    for eid in gt_ids:
        val = global_labels.get(eid, {})
        label = val.get("label", eid) if isinstance(val, dict) else str(eid)
        gt_labels.append(label)
        
    num_entities = max(1, len(gt_labels))
    all_keys = list(global_labels.keys())
    
    options = [gt_labels]
    
    # Retry mechanism to avoid infinite loop if DB is too small
    attempts = 0
    while len(options) < 4 and attempts < 100:
        attempts += 1
        random_keys = random.sample(all_keys, num_entities)
        fake_labels = []
        for k in random_keys:
            val = global_labels.get(k, {})
            label = val.get("label", k) if isinstance(val, dict) else str(k)
            fake_labels.append(label)
            
        if fake_labels not in options:
            options.append(fake_labels)
            
    # Pad if db was too small
    while len(options) < 4:
        options.append([f"Unknown Entity {len(options)}"])
            
    random.shuffle(options)
    
    options_text = "\n\nOptions:"
    letters = ["A", "B", "C", "D"]
    correct_letter = "None"
    
    for i, opt in enumerate(options):
        text_val = ", ".join(opt)
        options_text += f"\n{letters[i]}. {text_val}"
        if opt == gt_labels:
            correct_letter = letters[i]
            
    return options_text, correct_letter
