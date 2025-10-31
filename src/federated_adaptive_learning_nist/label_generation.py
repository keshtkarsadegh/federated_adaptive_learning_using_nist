import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

from src.federated_adaptive_learning_nist.split_hash_by_digits import split_hashes_by_digit

DIGIT_NAMES = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]

def process_digit(args):
    digit, digit_log_path, by_write_lines = args
    hash_set = set()

    # Load hashes for this digit
    with open(digit_log_path, "r") as f:
        for line in f:
            h = line.strip()
            if h:
                hash_set.add(h)

    label_map = {}
    for line in by_write_lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        hash_val, full_path = parts
        if hash_val not in hash_set:
            continue  # hash doesn't belong to this digit

        path_parts = full_path.strip().split("/")
        if len(path_parts) < 6:
            continue  # malformed path

        penultimate_folder = path_parts[-2]
        if not penultimate_folder.startswith("d"):
            continue  # skip if not a digit writer folder

        # Extract path starting from 'by_write'
        if "by_write" in path_parts:
            idx = path_parts.index("by_write")
            relative_path = "/".join(path_parts[idx:])
            label_map[relative_path] = digit

    return label_map

def generate_labels_parallel(by_write_log, digit_hash_logs_dir, output_json_path, num_workers=None):
    # Step 1: Load all by_write lines into memory
    with open(by_write_log, "r") as f:
        by_write_lines = f.readlines()

    # Step 2: Prepare parallel tasks — one per digit
    tasks = []
    for digit, name in enumerate(DIGIT_NAMES):
        log_file = Path(digit_hash_logs_dir) / f"{name}.log"
        if log_file.exists():
            tasks.append((digit, log_file, by_write_lines))
        else:
            print(f"⚠️ Skipping missing digit log: {log_file}")

    # Step 3: Run digit processing in parallel
    with Pool(processes=num_workers or min(cpu_count(), 10)) as pool:
        results = pool.map(process_digit, tasks)

    # Step 4: Merge all results into one dictionary
    combined_map = {}
    for label_map in results:
        combined_map.update(label_map)

    # Step 5: Write the final label map to JSON`
    with open(output_json_path, "w") as f:
        json.dump(combined_map, f, indent=2)

    print(f"✅ Wrote {len(combined_map)} labeled entries to {output_json_path}")



if __name__=="__main__":
    script_dir = Path(__file__).resolve().parent
    PROJECT_ROOT = script_dir.parent.parent
    target_dir = str(PROJECT_ROOT / "data")
    DIGITS_HASHES = f"{target_dir}/by_write/digits_hashes"
    JSON_DIGIT_LABELS = f"{target_dir}/by_write/digits_labels.json"
    BY_CLASS_LOG =f"{target_dir}/by_class/by_class_md5.log"
    BY_WRITE_LOG = f"{target_dir}/by_write/by_write_md5.log"

    split_hashes_by_digit(BY_CLASS_LOG,DIGITS_HASHES)
    # 🔧 Run it (entry point)
    generate_labels_parallel(
        by_write_log=BY_WRITE_LOG,
        digit_hash_logs_dir=DIGITS_HASHES,
        output_json_path=JSON_DIGIT_LABELS,
        num_workers=50
    )
