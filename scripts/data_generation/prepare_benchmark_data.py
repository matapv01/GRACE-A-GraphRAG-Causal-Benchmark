import subprocess
import sys
import os

def run_script(script_path, args=[]):
    print(f"\n--- Running: {script_path} ---")
    try:
        subprocess.run(["uv", "run", "python", script_path] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        sys.exit(1)

def main():
    # 1. Fetch Wikidata Labels (The base metadata)
    # This creates data/wikidata_labels.json
    if not os.path.exists("data/wikidata_labels.json"):
        run_script("scripts/data_generation/fetch_wikidata_labels.py")
    else:
        print("data/wikidata_labels.json already exists, skipping fetch.")

    # 2. Extract Clean Subgraphs (Base datasets)
    # This creates data/test_clean_subgraphs/
    run_script("scripts/data_generation/extract_clean_subgraphs.py")

    # 3. Generate Perturbations (Causal scenarios)
    # This creates data/test_perturbed_subgraphs/
    run_script("scripts/data_generation/generate_perturbations.py")

    # 4. Convert to MCQ (The final step for the current benchmark)
    # This adds ABCD options to the JSON files
    run_script("scripts/data_generation/convert_to_mcq.py")

    print("\n[SUCCESS] Causal Benchmark Data Preparation complete!")
    print("Clean datasets: data/test_clean_subgraphs/")
    print("Perturbed datasets: data/test_perturbed_subgraphs/")

if __name__ == "__main__":
    main()
