import os
import json
import glob
import shutil

def find_and_move_outliers(base_dir="data", threshold=1000, dest_dir="data/outliers_large_subgraphs"):
    # Find directories starting with 'test_' (e.g., test_clean_subgraphs, test_perturbed_subgraphs)
    sub_dirs = [d for d in os.listdir(base_dir) if d.startswith("test_") and os.path.isdir(os.path.join(base_dir, d))]
    
    outlier_ids = set()
    outlier_details = {}

    print("=" * 60)
    print(" STEP 1: SCANNING FOR OUTLIERS".center(60))
    print("=" * 60)
    print(f"Filtering Condition (Threshold): > {threshold} Nodes or Edges")
    
    # Collect all violating IDs across any sub-directory/variant
    for sub_dir in sub_dirs:
        dir_path = os.path.join(base_dir, sub_dir)
        files = glob.glob(os.path.join(dir_path, "*.json"))
        
        for filepath in files:
            q_id = os.path.basename(filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    
                    for variant, content in data.items():
                        if not isinstance(content, dict):
                            continue
                        triples = content.get("triples", [])
                        num_edges = len(triples)
                        
                        unique_nodes = set()
                        for t in triples:
                            s = t.get("subject")
                            o = t.get("object")
                            if s: unique_nodes.add(s)
                            if o: unique_nodes.add(o)
                            
                        num_nodes = len(unique_nodes)
                        
                        if num_nodes > threshold or num_edges > threshold:
                            outlier_ids.add(q_id)
                            if q_id not in outlier_details:
                                outlier_details[q_id] = []
                            outlier_details[q_id].append(f"{sub_dir}/{variant} ({num_nodes} nodes, {num_edges} edges)")
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")

    # Count total distinct questions (Q_IDs)
    all_q_ids = set()
    for sub_dir in sub_dirs:
         for f in os.listdir(os.path.join(base_dir, sub_dir)):
             if f.endswith(".json"):
                 all_q_ids.add(f)
                 
    total_files = len(all_q_ids)
    total_outliers = len(outlier_ids)
    
    if total_files == 0:
        print("No data found to check.")
        return

    percent_removed = (total_outliers / total_files) * 100
    
    print(f"Total original questions (Q_ID):          {total_files}")
    print(f"Total outlier questions detected:         {total_outliers}")
    print(f"Percentage of data isolated (filtered):   {percent_removed:.2f}%\n")

    if total_outliers == 0:
        print("Excellent! No files exceeded the threshold, the dataset is completely clean.")
        return

    print("=" * 60)
    print(" STEP 2: MOVING OUTLIER DATA".center(60))
    print("=" * 60)
    print(f"Target isolation directory: {dest_dir}\n")

    moved_count = 0
    for q_id in sorted(outlier_ids):
        # Move across ALL sub_dirs to maintain parallel dataset integrity
        success_move = False
        for sub_dir in sub_dirs:
            src_path = os.path.join(base_dir, sub_dir, q_id)
            if os.path.exists(src_path):
                target_dir = os.path.join(dest_dir, sub_dir)
                os.makedirs(target_dir, exist_ok=True)
                dest_path = os.path.join(target_dir, q_id)
                
                try:
                    shutil.move(src_path, dest_path)
                    success_move = True
                except Exception as e:
                    print(f"Error moving {src_path}: {e}")
        
        if success_move:
            moved_count += 1
            reasons_str = " | ".join(outlier_details[q_id][:2])
            if len(outlier_details[q_id]) > 2:
                reasons_str += "..."
            print(f"[Moved] {q_id} => due to: {reasons_str}")

    print("-" * 60)
    print(f"Successfully isolated {moved_count}/{total_outliers} outlier IDs!")
    print(f"Datasets in 'data/test_*' are now fully safe for GraphRAG Benchmark evaluation.")

if __name__ == "__main__":
    find_and_move_outliers(base_dir="data", threshold=1000, dest_dir="data/outliers_large_subgraphs")
