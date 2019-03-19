import argparse
import json
from seqeval.metrics import f1_score


def evaluate(validation_file, response_file):
    with open(validation_file, errors='replace') as f:
        val_data = json.load(f)
    with open(response_file, errors='replace') as f:
        resp_data = json.load(f)

    y_true = []
    y_pred = []
    for v, r in val_data, resp_data:
        if v['id'] != r['id']:
            print('Ids not matching! Something went wrong in the response file')
            return
        y_true.append(v['seq_labels'])
        y_pred.append(r['labels'])

    f1_score(y_true, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script',
        usage="evaluate.py <validation_file> <response_file>")
    parser.add_argument('validation_file', help='File containing validation set')
    parser.add_argument('response_file', help='File containing Wit.ai responses to the validation examples')
    args = parser.parse_args()
    evaluate(args.validation_file, args.reponse_file)
