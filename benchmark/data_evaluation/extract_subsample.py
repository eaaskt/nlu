import argparse
import json
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from enum import Enum


TEST_DATA_3_1 = '../data/romanian_dataset/home_assistant/take3/scenario3_imbalance/3.1/testing_dataset_wit_ids.json'
TEST_DATA_3_2 = '../data/romanian_dataset/home_assistant/take3/scenario3_imbalance/3.2/testing_dataset_wit_ids.json'
TEST_DATA_3_3 = '../data/romanian_dataset/home_assistant/take3/scenario3_imbalance/3.3/testing_dataset_wit_ids.json'
TEST_DATA = [TEST_DATA_3_1, TEST_DATA_3_2, TEST_DATA_3_3]

RESP_3_1 = '../data/romanian_dataset/home_assistant/take3/scenario3_imbalance/3.1/validation_results.json'
RESP_3_2 = '../data/romanian_dataset/home_assistant/take3/scenario3_imbalance/3.2/validation_results.json'
RESP_3_3 = '../data/romanian_dataset/home_assistant/take3/scenario3_imbalance/3.3/validation_results.json'
RESP = [RESP_3_1, RESP_3_2, RESP_3_3]


class RespFormat(Enum):
    wit = 'wit'
    rasa = 'rasa'

    def __str__(self):
        return self.value


def sample(data, percentage=0.1):
    nr_total_errors = len(data)
    nr_sample = math.ceil(percentage * nr_total_errors)
    sample = random.sample(data, k=nr_sample)
    return sample


def extract_errors(test_data, resp_data, resp_format):
    intent_errors = []
    slot_errors = []
    for t, r in zip(test_data, resp_data):
        if t['id'] != r['id']:
            print('Ids not matching! Something went wrong in the response file')
            return

        text = t['text']
        true_intent = None
        for e in t['entities']:
            if e['entity'] == 'intent':
                true_intent = e['value']
        true_slots = t['seq_labels']

        if resp_format == RespFormat.wit:
            pred_intent = r['entities']['intent'][0]['value']
            pred_slots = r['labels']
        else:
            pred_intent = r['intent']['name']
            pred_slots = r['labels']

        if true_intent != pred_intent:
            # Intent error
            intent_errors.append((text, true_intent, pred_intent))

        if true_slots != pred_slots:
            # Slots error
            slot_errors.append((text, true_slots, pred_slots))

    return intent_errors, slot_errors


def print_errors(err):
    for e in err:
        print('Text: ' + str(e[0]))
        print('TRUE: ' + str(e[1]))
        print('PRED: ' + str(e[2]))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script',
        usage='evaluate.py <evaluation_file> <validation_resp_format')
    parser.add_argument('resp_format', help='Format of response file', type=RespFormat, choices=list(RespFormat))
    args = parser.parse_args()

    intent_errors = []
    slot_errors = []
    for f1, f2 in zip(TEST_DATA, RESP):

        with open(f1, 'r') as f:
            test_data = json.load(f)
        with open(f2, 'r') as f:
            resp_data = json.load(f)

        intent_err, slot_err = extract_errors(test_data, resp_data, args.resp_format)
        intent_errors += intent_err
        slot_errors += slot_err

    print('Total # of errors (intent):' + str(len(intent_errors)))
    print('Total # of errors (slot):' + str(len(slot_errors)))
    intent_sample = sample(intent_errors)
    slot_sample = sample(slot_errors)

    print('INTENT ERRORS')
    print_errors(intent_sample)
    print()
    print('SLOT ERRORS')
    print_errors(slot_sample)
