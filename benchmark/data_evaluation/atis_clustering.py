import argparse
import json


INTENTS_TO_REPLACE = {
    'aircraft+flight+flight_no' : 'aircraft',
    'airfare+flight': 'airfare',
    'airfare+flight_time': 'airfare',
    'airline+flight_no': 'airline',
    'flight+airfare': 'flight',
    'flight+airline': 'flight',
    'flight_no+airline': 'flight_no',
    'ground_service+ground_fare': 'ground_service'
}


def cluster(input_path, output_path):
    with open(input_path, errors='replace') as f:
        data = json.load(f)
    conv_data = []
    for d in data:
        # Skip over 'flight' intents
        for entity in d['entities']:
            if entity['entity'] == 'intent' and entity['value'] != 'flight':
                if entity['value'] in INTENTS_TO_REPLACE.keys():
                    entity['value'] = INTENTS_TO_REPLACE[entity['value']]
                conv_data.append(d)

    with open(output_path, 'w') as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove flight intents and cluster the group intents',
        usage="atis_clustering.py <input_file> <output_file>")
    parser.add_argument('input_file', help='Input file in wit + iob labels format')
    parser.add_argument('output_file', help='Output file')
    args = parser.parse_args()
    cluster(args.input_file, args.output_file)
    print('Clustered results saved in {}'.format(args.output_file))

