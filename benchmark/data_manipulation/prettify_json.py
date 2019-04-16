import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description="Pretty print JSON file",
                    usage="prettify_json.py <input_file>")
    parser.add_argument('input_file', help='Input file')
    args = parser.parse_args()
    with open(args.input_file, errors='replace', encoding='utf-8-sig') as f:
        data = json.load(f)
    with open(args.input_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)