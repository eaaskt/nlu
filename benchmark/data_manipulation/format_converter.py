from enum import Enum
import json
import argparse


class InputFormat(Enum):
    snips = 'snips'
    rasa = 'rasa'

    def __str__(self):
        return self.value


class OutputFormat(Enum):
    snips = 'snips'
    rasa = 'rasa'
    wit = 'wit'

    def __str__(self):
        return self.value


def convert_entry_rasa(itm, output_format):
    if output_format == OutputFormat.wit:
        entities = itm['entities']
        entities.insert(0, {'entity': 'intent', 'value': itm['intent']})
        res = {'text': itm['text'], 'entities': entities}
    return res


def convert_entry_snips(itm, intent, output_format):
    entities = []
    text = ''
    if output_format == OutputFormat.wit:
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

    if output_format == OutputFormat.wit:
        res = {'text': text, 'entities': entities}
    elif output_format == OutputFormat.rasa:
        res = {'text': text, 'entities': entities, 'intent': intent}
    return res


def convert(input_path, output_path, init_format, output_format):
    with open(input_path, errors='replace') as f:
        data = json.load(f)
    if init_format == InputFormat.snips:
        assert len(data.keys()) == 1
        key = list(data.keys())[0]
        examples = [convert_entry_snips(itm, key, output_format) for itm in data[key]]
        if output_format == OutputFormat.rasa:
            output = dict()
            output['rasa_nlu_data'] = {}
            output['rasa_nlu_data']['common_examples'] = examples
        elif output_format == OutputFormat.wit:
            output = examples
        with open(output_path, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    elif init_format == InputFormat.rasa:
        data = data['rasa_nlu_data']['common_examples']
        output = [convert_entry_rasa(itm, output_format) for itm in data]
        with open(output_path, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Convert to Wit.ai format",
                usage="format_converter.py <input_file> <output_file> <init_format> <output_format>")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('init_format', help='Initial format of data', type=InputFormat, choices=list(InputFormat))
    parser.add_argument('output_format', help='Output format of data', type=OutputFormat, choices=list(OutputFormat))
    args = parser.parse_args()
    convert(args.input_file, args.output_file, args.init_format, args.output_format)
    print('Converted results saved in {}'.format(args.output_file))
