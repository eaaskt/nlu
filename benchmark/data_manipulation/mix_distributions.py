"""
Script that randomly selects n examples from each intent in the test set and moves them to the training set.

The train set will maintain its original size, while the test set will have n examples less for each intent.
Those examples could be manually replaced via Chatito.
"""

import json
import os

N_EXAMPLES = 1
BASE_DIR = '../data/romanian_dataset/home_assistant/diacritics/scenario_'
BASE_DIR_OUTPUT = '../data/romanian_dataset/home_assistant/diacritics_mixed_1/scenario_'
TRAIN_FILE = 'train.json'
TEST_FILE = 'test.json'


def build_files_list(dir_path):
    # No scenario 0 because the train and test sets are already drawn from the same distribution there
    return [
        (dir_path + '1/' + TRAIN_FILE, dir_path + '1/' + TEST_FILE),
        (dir_path + '2/' + TRAIN_FILE, dir_path + '2/' + TEST_FILE),
        (dir_path + '3_1/' + TRAIN_FILE, dir_path + '3_1/' + TEST_FILE),
        (dir_path + '3_2/' + TRAIN_FILE, dir_path + '3_2/' + TEST_FILE),
        (dir_path + '3_3/' + TRAIN_FILE, dir_path + '3_3/' + TEST_FILE),
    ]


INPUT_FILES = build_files_list(BASE_DIR)
OUTPUT_FILES = build_files_list(BASE_DIR_OUTPUT)


def categorize_intents(examples_list):
    intent_examples = dict()
    for ex in examples_list:
        if ex['intent'] in intent_examples:
            intent_examples[ex['intent']].append(ex)
        else:
            intent_examples[ex['intent']] = [ex]
    return intent_examples


def remove_random(examples_list, examples_to_add=None):
    intent_examples = categorize_intents(examples_list)
    removed = dict()
    for intent, i_examples in intent_examples.items():
        rm_examples = i_examples[:N_EXAMPLES]
        removed[intent] = rm_examples
        del i_examples[:N_EXAMPLES]
    if examples_to_add is not None:
        for intent, examples in examples_to_add.items():
            intent_examples[intent] = intent_examples[intent] + examples
    dict_values = list(intent_examples.values())
    flat_examples = [x for l in dict_values for x in l]
    return flat_examples, removed


def mix_distributions(i_train, i_test, o_train, o_test):
    with open(i_test, errors='replace', encoding='utf-8') as f:
        data_i_te = json.load(f)
    with open(i_train, errors='replace', encoding='utf-8') as f:
        data_i_tr = json.load(f)
    new_examples_te, removed_examples_te = remove_random(data_i_te['rasa_nlu_data']['common_examples'])
    print(removed_examples_te)
    new_examples_tr, _ = remove_random(data_i_tr['rasa_nlu_data']['common_examples'], removed_examples_te)
    data_i_te['rasa_nlu_data']['common_examples'] = new_examples_te
    data_i_tr['rasa_nlu_data']['common_examples'] = new_examples_tr

    os.makedirs(os.path.dirname(o_train), exist_ok=True)
    os.makedirs(os.path.dirname(o_test), exist_ok=True)
    with open(o_train, 'w', encoding='utf-8') as f:
        json.dump(data_i_tr, f, ensure_ascii=False, indent=4)
    with open(o_test, 'w', encoding='utf-8') as f:
        json.dump(data_i_te, f, ensure_ascii=False, indent=4)


def main():
    for in_files, out_files in zip(INPUT_FILES, OUTPUT_FILES):
        print(in_files)
        print(out_files)
        mix_distributions(in_files[0], in_files[1], out_files[0], out_files[1])


if __name__ == '__main__':
    main()
