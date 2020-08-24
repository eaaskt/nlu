"""
    Script that adds the cresteIntensitateMuzica and scadeIntensitateMuzica intents to the original datasets

    This script also deletes the schimbaIntensitateMuzicaIntent, as it should be replaced with the two new intents:
    cresteIntensitateMuzica and scadeIntensitateMuzica, so that the dataset remains consistent with the other types of
    intents (creste/scadeTemperatura and creste/scadeIntensitateLumina)
"""

import json

FILES_TO_MODIFY = [
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/test.json',
]

NEW_FILES = [
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/test.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/train.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/test.json',
]

VOLUME_INTENTS_FILES = [
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/train-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/test-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/train-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/test-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/train-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/test-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/train-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/test-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/train-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/test-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/train-fixed.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/test-fixed.json',
]


def update(input_path, new_volume_path, output_path):
    with open(input_path, errors='replace', encoding='utf-8') as f:
        data = json.load(f)
    examples = data['rasa_nlu_data']['common_examples']
    updated_examples = []
    for itm in examples:
        if itm['intent'] != 'cresteIntensitateMuzica' and itm['intent'] != 'scadeIntensitateMuzica':
            updated_examples.append(itm)
    with open(new_volume_path, errors='replace', encoding='utf-8') as f:
        new_intents_data = json.load(f)
    for itm in new_intents_data['rasa_nlu_data']['common_examples']:
        updated_examples.append(itm)
    data['rasa_nlu_data']['common_examples'] = updated_examples
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))


def main():
    for f1, f2, fo in zip(FILES_TO_MODIFY, VOLUME_INTENTS_FILES, NEW_FILES):
        update(f1, f2, fo)
        print('Fixed dataset saved in ' + fo)


if __name__ == '__main__':
    main()
