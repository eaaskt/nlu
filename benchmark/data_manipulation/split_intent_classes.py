"""
    Script that splits the whole dataset into 3 subdatasets, corresponding to the 3 intent classes: light, temp, media
"""

import json

CLASS_CORRESP = {'aprindeLumina': 'lumina',
                 'cresteIntensitateLumina': 'lumina',
                 'cresteIntensitateMuzica': 'media',
                 'cresteTemperatura': 'temperatura',
                 'opresteMuzica': 'media',
                 'opresteTV': 'media',
                 'pornesteTV': 'media',
                 'puneMuzica': 'media',
                 'scadeIntensitateLumina': 'lumina',
                 'scadeIntensitateMuzica': 'media',
                 'scadeTemperatura': 'temperatura',
                 'schimbaCanalTV': 'media',
                 'schimbaIntensitateMuzica': 'media',
                 'seteazaTemperatura': 'temperatura',
                 'stingeLumina': 'lumina',
                 }

FILES_TO_MODIFY = [
    '../data/romanian_dataset/home_assistant/diacritics/scenario_0/train.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_0/test.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_1/train.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_1/test.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_2/train.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_2/test.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_3_1/train.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_3_1/test.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_3_2/train.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_3_2/test.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_3_3/train.json',
    '../data/romanian_dataset/home_assistant/diacritics/scenario_3_3/test.json',
]


def categorize(input_path):
    with open(input_path, errors='replace', encoding='utf-8') as f:
        data = json.load(f)
    examples = data['rasa_nlu_data']['common_examples']
    data_light = {'rasa_nlu_data': {}}
    data_light['rasa_nlu_data']['common_examples'] = []
    data_temp = {'rasa_nlu_data': {}}
    data_temp['rasa_nlu_data']['common_examples'] = []
    data_media = {'rasa_nlu_data': {}}
    data_media['rasa_nlu_data']['common_examples'] = []

    for itm in examples:
        intent = itm['intent']
        if CLASS_CORRESP[intent] == 'lumina':
            data_light['rasa_nlu_data']['common_examples'].append(itm)
        elif CLASS_CORRESP[intent] == 'temperatura':
            data_temp['rasa_nlu_data']['common_examples'].append(itm)
        elif CLASS_CORRESP[intent] == 'media':
            data_media['rasa_nlu_data']['common_examples'].append(itm)

    fname = input_path[:-5]
    light_path = fname + '-light.json'
    temp_path = fname + '-temp.json'
    media_path = fname + '-media.json'

    with open(light_path, 'w', encoding='utf-8') as f:
        json.dump(data_light, f, ensure_ascii=False, separators=(',', ':'))
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data_temp, f, ensure_ascii=False, separators=(',', ':'))
    with open(media_path, 'w', encoding='utf-8') as f:
        json.dump(data_media, f, ensure_ascii=False, separators=(',', ':'))


def main():
    for f in FILES_TO_MODIFY:
        categorize(f)


if __name__ == '__main__':
    main()
