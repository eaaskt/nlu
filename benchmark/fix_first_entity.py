import json
import argparse


def fix_entry(itm):
    for entity in itm['entities']:
        if entity['start'] == 1:
            entity['start'] = 0
            entity['end'] -= 1
    return itm


def fix_first_entity(input_path, output_path):
    with open(input_path, errors='replace') as f:
        data = json.load(f)
    examples = data['rasa_nlu_data']['common_examples']
    examples_fixed = [fix_entry(itm) for itm in examples]
    output = dict()
    output['rasa_nlu_data'] = {}
    output['rasa_nlu_data']['common_examples'] = examples_fixed
    with open(output_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Fix first entity start and end limits",
                usage="fix_first_entity.py <input_file> <output_file>")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    args = parser.parse_args()
    fix_first_entity(args.input_file, args.output_file)
    print('Results saved in {}'.format(args.output_file))
