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


def convert_iob_prediction(item):
    """Takes a prediction response in Wit.ai format and returns the sequence labeling for the entities"""
    text = item['_text']
    seq_labels = ['O'] * len(text.split(' '))
    for entity_name, entity_dict in item['entities'].items():
        if entity_name != 'intent':
            seq_idx = words_before_index(text, entity_dict[0]['_start'])
            entity_len = len(entity_dict[0]['value'].split(' '))
            seq_labels[seq_idx] = 'B-' + entity_name
            for w in range(1, entity_len):
                seq_labels[seq_idx + w] = 'I-' + entity_name
    return seq_labels


def convert(input_path, output_path, pred=False, ids=False):
    curr_id = 1
    with open(input_path, errors='replace') as f:
        data = json.load(f)
    if pred:
        for d in data:
            if ids:
                d['id'] = curr_id
                curr_id += 1
            d['seq_labels'] = convert_iob_prediction(d)

    else:
        for d in data:
            if ids:
                d['id'] = curr_id
                curr_id += 1
            d['seq_labels'] = convert_iob_example(d)

    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add labels in IOB format to data to prepare for evaluation',
        usage="eval_converter.py <input_file> <output_file> [--pred] [--ids]")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--pred', help='indicates that the input file contains Wit.ai predictions', action='store_true')
    parser.add_argument('--ids', help='add ids for examples/predictions', action='store_true')
    args = parser.parse_args()
    convert(args.input_file, args.output_file, pred=args.pred, ids=args.ids)
    print('Converted results saved in {}'.format(args.output_file))
