from enum import Enum
import json
import argparse


class Format(Enum):
    snips = 'snips'
    rasa = 'rasa'

    def __str__(self):
        return self.value


def convert_entry_rasa(itm):
    entities = itm['entities']
    entities.insert(0, {'entity': 'intent', 'value': itm['intent']})
    res = {'text': itm['text'], 'entities': entities}
    return res


def convert_entry_snips(itm, intent):
    text = ''.join([d['text'] for d in itm['data']])
    entities = []
    text = ''
    entities.append({'entity': 'intent', 'value': intent})
    for el in itm['data']:
        etxt = el['text']
        if "\ufffd" in etxt:
            print('Warning: replacing unknown char in {} [{}]'.format(etxt, text))
            etxt = etxt.replace("\ufffd","")
        if 'entity' in el:
            entities.append({'start': len(text),
                             'end': len(text)+len(etxt),
                             'value': etxt,
                             'entity': el['entity']})
        text += etxt

    res = {'text': text, 'entities': entities}
    return res


def convert(input_path, output_path, init_format):
    with open(input_path, errors='replace') as f:
        data = json.load(f)
    if init_format == Format.snips:
        assert len(data.keys()) == 1
        key = list(data.keys())[0]
        output = [convert_entry_snips(itm, key) for itm in data[key]]
        with open(output_path, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    elif init_format == Format.rasa:
        data = data['rasa_nlu_data']['common_examples']
        output = [convert_entry_rasa(itm) for itm in data]
        with open(output_path, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Convert to Wit.ai format",
                usage="format_converter.py <input_file> <output_file>")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('init_format', help='Initial format of data', type=Format, choices=list(Format))
    args = parser.parse_args()
    convert(args.input_file, args.output_file, args.init_format)
    print('Converted results saved in {}'.format(args.output_file))
