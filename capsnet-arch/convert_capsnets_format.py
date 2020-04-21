import json
import argparse
import random


def convert_entry(itm):
    entities = itm['entities']
    intent = ''
    text = itm['text'].replace(',', '')
    slots = []
    for entity in entities:
        if entity['entity'] == 'intent':
            intent = entity['value']

            # For zero-shot CapsNet
            # split_intent = re.sub('([a-z])([A-Z])', r'\1 \2', intent).split()
            # split_intent = [w.lower() for w in split_intent]
            # intent = ' '.join(split_intent)

    slots = itm['seq_labels']
    return intent, text, slots


def convert(input_path, output_path, shuffle=False):
    with open(input_path, errors='replace', encoding='utf-8-sig') as f:
        data = json.load(f)
    output = [convert_entry(itm) for itm in data]
    if shuffle:
        random.shuffle(output)
    intents = set()
    all_slots = set()
    with open(output_path, 'w', encoding='utf-8') as f:
        for intent, text, slots in output:
            f.write(intent)
            f.write('\t')
            f.write(' '.join(slots))
            f.write('\t')
            f.write(text)
            f.write('\n')
            intents.add(intent)
            for s in slots:
                all_slots.add(s)
    intents = list(intents)
    slots = list(all_slots)
    print(intents)
    print(slots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Convert to format needed for CapsNets',
                usage='format_converter.py <input_file> <output_file> [--shuffle]')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--shuffle', help='shuffles the dataset', action='store_true')
    args = parser.parse_args()
    convert(args.input_file, args.output_file, shuffle=args.shuffle)
    print('Converted results saved in {}'.format(args.output_file))
