import argparse
import json


def comparator(train_file, validate_file,true_test_file, output_file):
    with open(train_file) as inf:
        test_data = json.loads(inf.read())

    with open(validate_file) as inf:
        validate_data = json.loads(inf.read())

    with open(true_test_file) as inf:
        true_test = json.loads(inf.read())

    true_test_dict = dict()
    for sample in true_test:
        true_test_dict[int(sample["id"])] = sample

    train_dict = dict()
    for sample in test_data:
        train_dict[int(sample["id"])] = sample

    valid_dict = dict()
    for sample in validate_data:
        valid_dict[int(sample["id"])] = sample

    for train_sample in test_data:
        valid_sample = valid_dict[int(train_sample["id"])]
        tr_test_sample = true_test_dict[int(train_sample["id"])]
        tr_te_intent = None
        for entity in tr_test_sample['entities']:
            if entity['entity'] == 'intent':
                tr_te_intent = entity['value']
                break

        tr_intent = None
        for entity in train_sample['entities']:
            if entity['entity'] == 'intent':
                tr_intent = entity['value']
                break
        v_intent = None
        for entity in valid_sample['entities']:
            if entity == 'intent':
                v_intent = valid_sample['entities'][entity][0]['value']
                break

        if tr_intent == 'flight' and v_intent != 'flight':
            print(train_sample["text"] + " --> " + v_intent + "(" + tr_te_intent + ")")
    # with open(output_file, "w") as ff:
    #     json.dump(misclassed_list, ff)
    #
    # with open(output_file, "w") as of:
    #     json.dump(response, of, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Get results from Wit.ai for validation set",
        usage="post_entities.py <config_file> <input_file> <failed_samples_file> <app_name>")
    parser.add_argument('test_file', help='test file')
    parser.add_argument('valid_file', help='validation file')
    parser.add_argument('true_test_file', help='true_test file')
    parser.add_argument('output_file', help='output file')
    args = parser.parse_args()
    comparator(train_file=args.test_file, validate_file=args.valid_file,true_test_file=args.true_test_file,
               output_file=args.output_file)
