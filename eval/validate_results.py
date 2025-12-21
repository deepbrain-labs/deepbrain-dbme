import json, sys
from jsonschema import validate, ValidationError

try:
    schema = json.load(open('eval/schema/result_schema.json'))
except FileNotFoundError:
    print("ERROR: Schema file not found at 'eval/schema/result_schema.json'.")
    sys.exit(1)

def validate_file(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Results file not found: {path}. Skipping.")
        return []
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {path}.")
        return [(i, "Invalid JSON format")]

    bad = []
    for i, r in enumerate(data):
        try:
            validate(r, schema)
        except ValidationError as e:
            # Shorten the error message for readability
            error_message = f"Record {i}: {e.message} in field '{'.'.join(str(p) for p in e.path)}'"
            bad.append(error_message)
    return bad

if __name__ == "__main__":
    files_to_check = [
        'results/dbme_retention.json',
        'results/baseline_kv.json',
        'results/baseline_retrieval.json',
        'results/c2_consolidation.json',
        'results/c3_forgetting.json'
    ]

    total_errors = 0
    for f in files_to_check:
        print(f"--- Validating {f} ---")
        bad_records = validate_file(f)
        if bad_records:
            print(f"Found {len(bad_records)} invalid records in {f}:")
            for error in bad_records[:5]:  # Print first 5 errors
                print(f"  - {error}")
            if len(bad_records) > 5:
                print(f"  ... and {len(bad_records) - 5} more.")
            total_errors += len(bad_records)
        else:
            print("OK")
    
    if total_errors > 0:
        print(f"\nValidation failed with a total of {total_errors} errors.")
        sys.exit(1)
    else:
        print("\nAll result files are valid.")