import argparse
import json

import matplotlib.pyplot as plt

from enum import Enum


class RespFormat(Enum):
    wit = 'wit'
    rasa = 'rasa'

    def __str__(self):
        return self.value


def plot_confidence_levels(test_file_path, resp_file_path, resp_format=RespFormat.rasa):
    with open(test_file_path, 'r') as f:
        test_data = json.load(f)

    with open(resp_file_path, 'r') as f:
        resp_data = json.load(f)

    confidence_levels = []
    if resp_format == RespFormat.rasa:
        test_data = test_data['rasa_nlu_data']['common_examples']
        for t, r in zip(test_data, resp_data):
            if t['intent'] != r['intent']['name']:
                # Error,  add to list of confidence levels
                confidence_levels.append(100 * r['intent']['confidence'])

    elif resp_format == RespFormat.wit:
        for t, r in zip(test_data, resp_data):
            true_intent = ''

            for e in t['entities']:
                if e['entity'] == 'intent':
                    true_intent = e['value']
            pred_intent = r['entities']['intent'][0]['value']
            pred_intent_confidence = r['entities']['intent'][0]['confidence']

            if true_intent != pred_intent:
                # Error,  add to list of confidence levels
                confidence_levels.append(100 * pred_intent_confidence)

    plot_histogram(confidence_levels, num_bins=25)
    plt.show()


def plot_histogram(confidence_levels, num_bins=10):
    print('Here we will plot the confidence levels')
    plt.hist(confidence_levels, num_bins, edgecolor='k')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script',
        usage='evaluate.py <evaluation_file> <validation_resp_format')
    parser.add_argument('test_file', help='File containing test samples')
    parser.add_argument('resp_file', help='File containing prediction responses from the tool')
    parser.add_argument('resp_format', help='Format of response file', type=RespFormat, choices=list(RespFormat))
    args = parser.parse_args()

    plot_confidence_levels(args.test_file, args.resp_file, args.resp_format)
