import json
import argparse


def extract(input_file, output_file):
    with open(input_file, "r") as fp:
        data = json.load(fp)

    entity_set = set()
    for sample in data:
        for i in range(1, len(sample.get("entities"))):
            entity_obj = sample.get("entities")[i]
            entity_set.add(entity_obj.get("entity"))

    entity_list = list(entity_set)

    with open(output_file, "w") as rf:
        rf.write("[\n")
        for i in range(0, len(entity_list) - 1):
            rf.write('  \"' + entity_list[i] + '\"' + ",\n")
        rf.write('  \"' + entity_list[len(entity_list) - 1] + '\"\n')
        rf.write("]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Extract entities from sample Wit.ai format",
                usage="entity_extractor.py <input_file> <output_file>")
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    args = parser.parse_args()
    extract(args.input_file, args.output_file)
    print('Extraction results saved in {}'.format(args.output_file))
