import json
import argparse


def test_iob(converted_path, true_path):
    with open(converted_path, errors='replace') as f:
        converted_iob = json.load(f)
    with open(true_path, errors='replace') as f:
        true_iob = json.load(f)

    assert(len(converted_iob) == len(true_iob))

    incorrect = []
    for c in converted_iob:
        if c['seq_labels'] != true_iob[str(c['id'])]['seq']:
            incorrect.append(c['id]'])

    return incorrect


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Testing IOB labels",
                usage="test_iob_labels.py <converted_iob_file> <true_iob_file>")
    parser.add_argument('converted_iob_file', help='File with IOB tags obtained via eval_converter script (in Wit.ai - style format')
    parser.add_argument('true_iob_file', help='File containing true IOB labeling for the same data')
    args = parser.parse_args()
    incorrect_ids = test_iob(args.converted_iob_file, args.true_iob_file)
    if not incorrect_ids:
        print("Conversion correct")
    else:
        for i in incorrect_ids:
            print(i)

