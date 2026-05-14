import subprocess
import sys
import os
import argparse

def run_script(script_path, args=[]):
    print(f"\n--- Running: {script_path} ---")
    try:
        subprocess.run(["uv", "run", "python", script_path] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the generation of benchmark data.")
    parser.add_argument("--retry_errors", action="store_true", help="Retry failed extractions and perturbations.")
    args = parser.parse_args()

    retry_flag = ["--retry_errors"] if args.retry_errors else []

    # 1. Extract Clean Subgraphs (Base datasets)
    # This creates data/test_clean_subgraphs/
    run_script("scripts/data_generation/extract_clean_subgraphs.py", retry_flag)

    # 2. Generate Perturbations (Causal scenarios)
    # This creates data/test_perturbed_subgraphs/
    run_script("scripts/data_generation/generate_perturbations.py", retry_flag)

    # 3. Fetch Wikidata Labels (The base metadata)
    # We run this after perturbations so we have the IDs to fetch.
    if not os.path.exists("data/wikidata_labels.json"):
        run_script("scripts/data_generation/fetch_wikidata_labels.py")
    else:
        print("data/wikidata_labels.json already exists, skipping fetch.")

    # 4. Convert to MCQ (The final step for the current benchmark)
    # This adds ABCD options to the JSON files
    run_script("scripts/data_generation/convert_to_mcq.py")

    print("\n[SUCCESS] Causal Benchmark Data Preparation complete!")
    print("Clean datasets: data/test_clean_subgraphs/")
    print("Perturbed datasets: data/test_perturbed_subgraphs/")

if __name__ == "__main__":
    main()
