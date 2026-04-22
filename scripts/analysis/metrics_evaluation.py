import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict

def evaluate_metrics(results_file):
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print(f"File {results_file} not found.")
        return

    # metrics per variant
    y_true_all = defaultdict(list)
    y_pred_all = defaultdict(list)

    for item in all_results:
        variants = item.get("variants", {})
        for variant_name, variant_data in variants.items():
            true_letter = variant_data.get("mcq_correct_letter", "None")
            pred_letter = variant_data.get("predicted_letter", "None")
            
            # Map unrecognized back to something neutral so it gets penalized
            if true_letter == "None": continue
            if pred_letter not in ["A", "B", "C", "D"]:
                pred_letter = "UNK" # will be counted as wrong
                
            y_true_all[variant_name].append(true_letter)
            y_pred_all[variant_name].append(pred_letter)

    print(f"=== MULTI-CLASS MCQ METRICS FOR {results_file} ===")
    for variant_name in y_true_all.keys():
        trues = y_true_all[variant_name]
        preds = y_pred_all[variant_name]
        
        acc = accuracy_score(trues, preds)
        
        # Macro F1 accounts for class imbalances evenly
        # include UNK in labels basically ensures we get penalized 
        labels_present = list(set(trues + preds))
        f1_m = f1_score(trues, preds, average='macro', labels=labels_present)
        
        print(f"\n[{variant_name.upper()}] (N={len(trues)} samples)")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {f1_m:.4f}")
        
    print("\nNote: Metrics are dynamically computed over A, B, C, D Multi-Class Classification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ABCD multiple choice benchmark results.")
    parser.add_argument("--results_file", type=str, default="data/lightrag/benchmark_results.json", help="Path to the JSON results file.")
    args = parser.parse_args()
    evaluate_metrics(args.results_file)
