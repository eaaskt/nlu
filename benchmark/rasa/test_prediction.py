from rasa_nlu.model import Interpreter
import argparse
import json


def load_interpreter(path):
    interpreter = Interpreter.load(path)
    return interpreter


def predict(interpreter, message):
    return interpreter.parse(message)


def words_before_index(text, idx):
    """Counts the number of words in the text before idx"""
    while text[idx] != ' ':
        idx -= 1
    if idx <= 0:
        return 0
    n_words = len(text[:idx].split(' '))
    return n_words


def convert_iob_prediction(item):
    """Takes a prediction response in Rasa format and returns the converted response"""
    conv_item = dict(item)
    text = item['text']
    seq_labels = ['O'] * len(text.rstrip().split(' '))
    for entity in conv_item['entities']:
        seq_idx = words_before_index(text, entity['start'])
        entity_value = text[entity['start']: entity['end']]
        entity['value'] = entity_value
        entity_len = len(entity_value.split(' '))
        seq_labels[seq_idx] = 'B-' + entity['entity']
        for w in range(1, entity_len):
            seq_labels[seq_idx + w] = 'I-' + entity['entity']
    conv_item['labels'] = seq_labels
    return conv_item


def validate(input_path, output_path, interpreter):
    with open(input_path, errors='replace') as f:
        data = json.load(f)

    val_data = []
    for sample in data:
        message = sample['text']
        resp = predict(interpreter, message)
        val_d = convert_iob_prediction(resp)
        val_d['id'] = sample['id']
        val_data.append(val_d)

    with open(output_path, 'w') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Obtain validation set predictions',
        usage="test_prediction.py <input_file> <output_file> <model>")
    parser.add_argument('input_file', help='Input file - containing validation set examples')
    parser.add_argument('output_file', help='Output file - where the predictions will be saved')
    parser.add_argument('model', help='Path where model is stored')
    args = parser.parse_args()

    interpreter = load_interpreter(args.model)
    print('Loaded interpreter!')
    validate(args.input_file, args.output_file, interpreter)
    print('Validation responses saved in {}'.format(args.output_file))
