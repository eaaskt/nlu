"""
Script to add the missing slot to the schimbaIntensitateMuzica intent

The slot "nivel" was missing due to an error in the Chatito script corresponding to this intent (in scenarios 0,1,3)
The default format for the input file is RASA
"""

import json

FILES_TO_FIX = [
    'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/training_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/testing_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/training_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/testing_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/training_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/testing_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/training_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/testing_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/training_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/testing_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/training_dataset.json',
    'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/testing_dataset.json',
]

# These are extracted from the Chatito scripts
SLOT_VALUES = [
    'putin mai tare',
    'puțin mai tare',
    'mai tare',
    'mai sus',
    'mai incet',
    'mai jos',
    'mai încet',
]

SLOT_NAME = 'nivel'


def fix_item(itm):
    text = itm['text']
    entities = itm['entities']
    for slot_val in SLOT_VALUES:
        slot_index = text.find(slot_val)
        if slot_index != -1:
            entity = {
                'start': slot_index,
                'end': slot_index + len(slot_val),
                'value': slot_val,
                'entity': SLOT_NAME
            }
            entities.append(entity)
            break
    itm['entities'] = entities
    return itm


def fix(input_path, output_path):
    with open(input_path, errors='replace', encoding='utf-8') as f:
        data = json.load(f)
    examples = data['rasa_nlu_data']['common_examples']
    for itm in examples:
        if itm['intent'] == 'schimbaIntensitateMuzica':
            itm = fix_item(itm)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))


def check_diff(input_path, output_path):
    with open(input_path, errors='replace', encoding='utf-8') as f:
        data_in = json.load(f)
    with open(output_path, errors='replace', encoding='utf-8') as f:
        data_out = json.load(f)
    examples_in = data_in['rasa_nlu_data']['common_examples']
    examples_out = data_out['rasa_nlu_data']['common_examples']
    for ex_in, ex_out in zip(examples_in, examples_out):
        if ex_in != ex_out:
            print('---------')
            print(ex_in)
            print('||||||||')
            print(ex_out)


def main():
    for filename in FILES_TO_FIX:
        fix(filename, filename)
        print('Fixed dataset saved in ' + filename)


if __name__ == '__main__':
    main()
