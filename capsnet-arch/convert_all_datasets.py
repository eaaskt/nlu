"""
Automate conversion for all datasets from Wit.ai format to our capsnet format
"""

import convert_capsnets_format

# Regular diacritics datasets
# INPUT_DIR = '../benchmark/data/romanian_dataset/home_assistant/diacritics/scenario_'
# CAPSNETS_DIR = '../data-capsnets/diacritics/scenario'

# Mixed distribution datasets - 1 example
INPUT_DIR = '../benchmark/data/romanian_dataset/home_assistant/diacritics_mixed_1/scenario_'
CAPSNETS_DIR = '../data-capsnets/diacritics_mixed_1/scenario'

INPUT_TRAIN_FILE = 'train_wit_ids.json'
INPUT_TEST_FILE = 'test_wit_ids.json'
CAPS_TRAIN_FILE = 'train.txt'
CAPS_TEST_FILE = 'test.txt'


def build_wit_files_list(dir_path):
    train_list = [
        # dir_path + '0/' + INPUT_TRAIN_FILE,
        dir_path + '1/' + INPUT_TRAIN_FILE,
        dir_path + '2/' + INPUT_TRAIN_FILE,
        dir_path + '3_1/' + INPUT_TRAIN_FILE,
        dir_path + '3_2/' + INPUT_TRAIN_FILE,
        dir_path + '3_3/' + INPUT_TRAIN_FILE,
    ]
    test_list = [
        # dir_path + '0/' + INPUT_TEST_FILE,
        dir_path + '1/' + INPUT_TEST_FILE,
        dir_path + '2/' + INPUT_TEST_FILE,
        dir_path + '3_1/' + INPUT_TEST_FILE,
        dir_path + '3_2/' + INPUT_TEST_FILE,
        dir_path + '3_3/' + INPUT_TEST_FILE,
    ]
    return train_list, test_list


def build_caps_files_list(dir_path):
    train_list = [
        # dir_path + '0/' + CAPS_TRAIN_FILE,
        dir_path + '1/' + CAPS_TRAIN_FILE,
        dir_path + '2/' + CAPS_TRAIN_FILE,
        dir_path + '31/' + CAPS_TRAIN_FILE,
        dir_path + '32/' + CAPS_TRAIN_FILE,
        dir_path + '33/' + CAPS_TRAIN_FILE,
    ]
    test_list = [
        # dir_path + '0/' + CAPS_TEST_FILE,
        dir_path + '1/' + CAPS_TEST_FILE,
        dir_path + '2/' + CAPS_TEST_FILE,
        dir_path + '31/' + CAPS_TEST_FILE,
        dir_path + '32/' + CAPS_TEST_FILE,
        dir_path + '33/' + CAPS_TEST_FILE,
    ]
    return train_list, test_list


DATASETS_WIT_FORMAT_TRAIN, DATASETS_WIT_FORMAT_TEST = build_wit_files_list(INPUT_DIR)
DATASETS_CAPSNET_TRAIN, DATASETS_CAPSNET_TEST = build_caps_files_list(CAPSNETS_DIR)


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


def main():
    for wit_train, capsnet_train in zip(DATASETS_WIT_FORMAT_TRAIN, DATASETS_CAPSNET_TRAIN):
        convert_capsnets_format.convert(wit_train, capsnet_train, shuffle=True)
    for wit_test, capsnet_test in zip(DATASETS_WIT_FORMAT_TEST, DATASETS_CAPSNET_TEST):
        convert_capsnets_format.convert(wit_test, capsnet_test, shuffle=False)


if __name__ == '__main__':
    main()
