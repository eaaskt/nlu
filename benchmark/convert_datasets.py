"""
Script that automatically takes given input datasets through the conversion pipeline

First, convert from RASA to Wit format
Then, add ids and IOB labels to Wit format
"""

import json
import data_manipulation.format_converter as format_conv
import data_evaluation.eval_converter as eval_conv

# TODO: Add config files for these

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

### DIA
INITIAL_DATASETS = [
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

DATASETS_WIT_FORMAT = [
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/train_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/test_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/train_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/test_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/train_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/test_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/train_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/test_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/train_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/test_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/train_wit.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/test_wit.json',
]

DATASETS_FINAL_FORMAT = [
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/train_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_0/test_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/train_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_1/test_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/train_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_2/test_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/train_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_1/test_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/train_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_2/test_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/train_wit_ids.json',
    'data/romanian_dataset/home_assistant/diacritics/scenario_3_3/test_wit_ids.json',
]


def main():
    for init_file, wit_file, wit_ids_file in zip(INITIAL_DATASETS, DATASETS_WIT_FORMAT, DATASETS_FINAL_FORMAT):
        format_conv.convert(init_file, wit_file, format_conv.InputFormat.rasa, format_conv.OutputFormat.wit)
        print('Wit dataset saved in ' + wit_file)
        eval_conv.convert(wit_file, wit_ids_file, pred=False, ids=True)
        print('Wit_ids dataset saved in ' + wit_ids_file)


if __name__ == '__main__':
    main()