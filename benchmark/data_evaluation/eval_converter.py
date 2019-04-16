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
    """Takes an example in Wit.ai format and returns the converted response"""
    conv_item = dict(item)
    text = item['text']
    text = ' '.join(text.replace('\n','').split())  # remove duplicate whitespace characters
    text = text.replace('&', 'and').rstrip()
    conv_item['text'] = text
    seq_labels = ['O'] * len(text.split(' '))
    for entity in conv_item['entities']:
        if entity['entity'] != 'intent':
            seq_idx = words_before_index(text, entity['start'])
            entity_value = entity['value']
            entity_value = (' '.join(entity_value.replace('\n', '').split())).replace('&', 'and').rstrip()
            entity['value'] = entity_value
            entity_len = len(entity_value.split(' '))
            seq_labels[seq_idx] = 'B-' + entity['entity']
            for w in range(1, entity_len):
                seq_labels[seq_idx + w] = 'I-' + entity['entity']
    conv_item['seq_labels'] = seq_labels
    return conv_item


def convert_iob_prediction(item):
    """Takes a prediction response in Wit.ai format and returns the converted response"""
    conv_item = dict(item)
    text = item['_text']
    seq_labels = ['O'] * len(text.rstrip().split(' '))
    for entity_name, entity_list in item['entities'].items():
        if entity_name != 'intent':
            for ent in entity_list:
                seq_idx = words_before_index(text, ent['_start'])
                entity_value = ent['value']
                entity_text_value = text[ent['_start']:ent['_end']]
                if len(entity_value.strip().split(' ')) != len(entity_text_value.strip().split(' ')):
                    entity_len = len(entity_text_value.strip().split(' '))
                else:
                    entity_len = len(entity_value.split(' '))
                seq_labels[seq_idx] = 'B-' + entity_name
                for w in range(1, entity_len):
                    seq_labels[seq_idx + w] = 'I-' + entity_name
    conv_item['labels'] = seq_labels
    return conv_item


def convert(input_path, output_path, pred=False, ids=False):
    curr_id = 1
    with open(input_path, errors='replace') as f:
        data = json.load(f)
    conv_data = []
    if pred:
        for d in data:
            conv_d = convert_iob_prediction(d)
            if ids:
                conv_d['id'] = curr_id
                curr_id += 1
            conv_data.append(conv_d)

    else:
        for d in data:
            conv_d = convert_iob_example(d)
            if ids:
                conv_d['id'] = curr_id
                curr_id += 1
            conv_data.append(conv_d)

    with open(output_path, 'w') as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=4)


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
