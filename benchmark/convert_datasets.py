"""
Script that automatically takes given input datasets through the conversion pipeline

First, convert from RASA to Wit format
Then, add ids and IOB labels to Wit format
"""

import json
import data_manipulation.format_converter as format_conv
import data_evaluation.eval_converter as eval_conv

# TODO: Add config files for these

# Regular diacritics datasets
# BASE_DIR = 'data/romanian_dataset/home_assistant/diacritics/scenario_'

# Mixed distribution datasets - 1 example
BASE_DIR = 'data/romanian_dataset/home_assistant/diacritics_mixed_1/scenario_'

INPUT_FILES = ('train.json', 'test.json')
WIT_FILES = ('train_wit.json', 'test_wit.json')
WIT_IDS_FILES = ('train_wit_ids.json', 'test_wit_ids.json')


def build_files_list(dir_path, files):
    train_file, test_file = files
    return [
        # dir_path + '0/' + train_file, dir_path + '0/' + test_file,
        dir_path + '1/' + train_file, dir_path + '1/' + test_file,
        dir_path + '2/' + train_file, dir_path + '2/' + test_file,
        dir_path + '3_1/' + train_file, dir_path + '3_1/' + test_file,
        dir_path + '3_2/' + train_file, dir_path + '3_2/' + test_file,
        dir_path + '3_3/' + train_file, dir_path + '3_3/' + test_file,
    ]


INITIAL_DATASETS = build_files_list(BASE_DIR, INPUT_FILES)
DATASETS_WIT_FORMAT = build_files_list(BASE_DIR, WIT_FILES)
DATASETS_FINAL_FORMAT = build_files_list(BASE_DIR, WIT_IDS_FILES)

### NO_DIA
# INITIAL_DATASETS = [
#     'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/training_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/testing_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/training_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/testing_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/training_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/testing_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/training_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/testing_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/training_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/testing_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/training_dataset.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/testing_dataset.json',
# ]
#
# DATASETS_WIT_FORMAT = [
#     'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/training_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/testing_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/training_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/testing_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/training_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/testing_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/training_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/testing_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/training_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/testing_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/training_dataset_wit.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/testing_dataset_wit.json',
# ]
#
# DATASETS_FINAL_FORMAT = [
#     'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/training_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario0_uniform_dist/testing_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/training_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario1_synonyms/testing_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/training_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario2_missing_slots/testing_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/training_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.1/testing_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/training_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.2/testing_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/training_dataset_wit_ids.json',
#     'data/romanian_dataset/home_assistant/version1/scenario3_imbalance/3.3/testing_dataset_wit_ids.json',
# ]


def main():
    for init_file, wit_file, wit_ids_file in zip(INITIAL_DATASETS, DATASETS_WIT_FORMAT, DATASETS_FINAL_FORMAT):
        format_conv.convert(init_file, wit_file, format_conv.InputFormat.rasa, format_conv.OutputFormat.wit)
        print('Wit dataset saved in ' + wit_file)
        eval_conv.convert(wit_file, wit_ids_file, pred=False, ids=True)
        print('Wit_ids dataset saved in ' + wit_ids_file)


if __name__ == '__main__':
    main()