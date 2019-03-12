import argparse
import json


def words_before_index(text, idx):
    """Counts the number of words in the text before idx"""
    while text[idx] != ' ':
        idx -= 1
    if idx <= 0:
        return 0
    n_words = len(text[:idx].split(' '))
    return n_words


def convert_iob_example(item):
    """Takes an example in Wit.ai format and returns the sequence labeling for the entities"""
    text = item['text']
    seq_labels = ['O'] * len(text.split(' '))
    for entity in item['entities']:
        if entity['entity'] != 'intent':
            seq_idx = words_before_index(text, entity['start'])
            entity_len = len(entity['value'].split(' '))
            seq_labels[seq_idx] = 'B-' + entity['entity']
            for w in range(1, entity_len):
                seq_labels[seq_idx + w] = 'I-' + entity['entity']
    return seq_labels


# def convert_IOB_response(item):

def convert(input_path, output_path, pred=False):
    with open(input_path, errors='replace') as f:
        data = json.load(f)
    if pred:
        print('Converting predictions')
    else:
        for d in data:
            d['seq_labels'] = convert_iob_example(d)
        with open(output_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add labels in IOB format to data',
        usage="IOB_converter.py <input_file> <output_file> [--pred]")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--pred', help='input file contains Wit.ai predictions', action='store_true')
    args = parser.parse_args()
    if args.pred:
        convert(args.input_file, args.output_file, pred=True)
    else:
        convert(args.input_file, args.output_file, pred=False)

    print('Converted results saved in {}'.format(args.output_file))

