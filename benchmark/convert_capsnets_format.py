import json
import argparse
import random
import re


def convert_entry(itm):
    entities = itm['entities']
    intent = ''
    text = itm['text'].replace(',', '')
    for entity in entities:
        if entity['entity'] == 'intent':
            intent = entity['value']
            split_intent = re.sub('([a-z])([A-Z])', r'\1 \2', intent).split()
            split_intent = [w.lower() for w in split_intent]
            intent = ' '.join(split_intent)
    return intent, text


def convert(input_path, output_path, shuffle=False):
    with open(input_path, errors='replace', encoding='utf-8-sig') as f:
        data = json.load(f)
    output = [convert_entry(itm) for itm in data]
    if shuffle:
        random.shuffle(output)
    intents = []
    with open(output_path, 'w') as f:
        for intent, text in output:
            f.write(intent)
            f.write('\t')
            f.write(text)
            f.write('\n')
            intents.append(intent)
    intents = list(set(intents))
    print(intents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Convert to format needed for CapsNets",
                usage="format_converter.py <input_file> <output_file> [--shuffle]")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--shuffle', help='shuffles the dataset', action='store_true')
    args = parser.parse_args()
    convert(args.input_file, args.output_file, shuffle=args.shuffle)
    print('Converted results saved in {}'.format(args.output_file))
