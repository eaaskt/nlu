"""
Automate conversion for all datasets from Wit.ai format to our capsnet format
"""

import convert_capsnets_format

### NO DIA
# DATASETS_WIT_FORMAT_TRAIN = [
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/training_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario1_synonyms/training_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/training_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/training_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/training_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/training_dataset_wit_ids.json',
# ]
#
# DATASETS_WIT_FORMAT_TEST = [
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/testing_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario1_synonyms/testing_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/testing_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/testing_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/testing_dataset_wit_ids.json',
#     '../benchmark/data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/testing_dataset_wit_ids.json',
# ]
#
# DATASETS_CAPSNET_TRAIN = [
#     '../data-capsnets/scenario0/train.txt',
#     '../data-capsnets/scenario1/train.txt',
#     '../data-capsnets/scenario2/train.txt',
#     '../data-capsnets/scenario3.1/train.txt',
#     '../data-capsnets/scenario3.2/train.txt',
#     '../data-capsnets/scenario3.3/train.txt',
# ]
#
# DATASETS_CAPSNET_TEST = [
#     '../data-capsnets/scenario0/test.txt',
#     '../data-capsnets/scenario1/test.txt',
#     '../data-capsnets/scenario2/test.txt',
#     '../data-capsnets/scenario3.1/test.txt',
#     '../data-capsnets/scenario3.2/test.txt',
#     '../data-capsnets/scenario3.3/test.txt',
# ]


#### DIA
DATASETS_WIT_FORMAT_TRAIN = [
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_0/train_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_1/train_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_2/train_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_3_1/train_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_3_2/train_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_3_3/train_wit_ids.json',

]

DATASETS_WIT_FORMAT_TEST = [
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_0/test_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_1/test_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_2/test_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_3_1/test_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_3_2/test_wit_ids.json',
    '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_3_3/test_wit_ids.json',
]

DATASETS_CAPSNET_TRAIN = [
    '../data-capsnets/diacritics/scenario0/train.txt',
    '../data-capsnets/diacritics/scenario1/train.txt',
    '../data-capsnets/diacritics/scenario2/train.txt',
    '../data-capsnets/diacritics/scenario31/train.txt',
    '../data-capsnets/diacritics/scenario32/train.txt',
    '../data-capsnets/diacritics/scenario33/train.txt',
]

DATASETS_CAPSNET_TEST = [
    '../data-capsnets/diacritics/scenario0/test.txt',
    '../data-capsnets/diacritics/scenario1/test.txt',
    '../data-capsnets/diacritics/scenario2/test.txt',
    '../data-capsnets/diacritics/scenario31/test.txt',
    '../data-capsnets/diacritics/scenario32/test.txt',
    '../data-capsnets/diacritics/scenario33/test.txt',
]


def main():
    for wit_train, capsnet_train in zip(DATASETS_WIT_FORMAT_TRAIN, DATASETS_CAPSNET_TRAIN):
        convert_capsnets_format.convert(wit_train, capsnet_train, shuffle=True)
    for wit_test, capsnet_test in zip(DATASETS_WIT_FORMAT_TEST, DATASETS_CAPSNET_TEST):
        convert_capsnets_format.convert(wit_test, capsnet_test, shuffle=False)


if __name__ == '__main__':
    main()
