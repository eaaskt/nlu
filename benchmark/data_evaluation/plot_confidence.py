import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from enum import Enum


class RespFormat(Enum):
    wit = 'wit'
    rasa = 'rasa'

    def __str__(self):
        return self.value


def plot_confidence_levels(test_file_path, resp_file_path, resp_format=RespFormat.rasa, intent=None):
    with open(test_file_path, 'r') as f:
        test_data = json.load(f)

    with open(resp_file_path, 'r') as f:
        resp_data = json.load(f)

    if intent:
        confidence_levels = dict()
        plot_title = intent + ' errors confidence distribution'
    else:
        confidence_levels = []
        plot_title = 'Rasa Intent error confidence distribution'

    if resp_format == RespFormat.rasa:
        test_data = test_data['rasa_nlu_data']['common_examples']
        for t, r in zip(test_data, resp_data):
            true_intent = t['intent']
            pred_intent = r['intent']['name']
            pred_intent_confidence = r['intent']['confidence']
            if t['intent'] != r['intent']['name']:
                if intent is not None:
                    if true_intent == intent:
                        print(t['text'])
                        print(true_intent)
                        print(pred_intent)
                        print(pred_intent_confidence)
                        print()
                        if pred_intent not in confidence_levels:
                            confidence_levels[pred_intent] = [100 * pred_intent_confidence]
                        else:
                            confidence_levels[pred_intent].append(100 * pred_intent_confidence)
                else:
                    confidence_levels.append(100 * pred_intent_confidence)

    elif resp_format == RespFormat.wit:
        for t, r in zip(test_data, resp_data):
            true_intent = ''

            for e in t['entities']:
                if e['entity'] == 'intent':
                    true_intent = e['value']
            pred_intent = r['entities']['intent'][0]['value']
            pred_intent_confidence = r['entities']['intent'][0]['confidence']

            if true_intent != pred_intent:
                if intent is not None:
                    if true_intent == intent:
                        print(true_intent)
                        print(pred_intent)
                        print(pred_intent_confidence)
                        print()
                        if pred_intent not in confidence_levels:
                            confidence_levels[pred_intent] = [100 * pred_intent_confidence]
                        else:
                            confidence_levels[pred_intent].append(100 * pred_intent_confidence)
                else:
                    confidence_levels.append(100 * pred_intent_confidence)

    plot_histogram(confidence_levels, plot_title)


def plot_histogram(confidence_levels, title=''):
    if type(confidence_levels) == list:
        bins = np.linspace(0, 100, 25)
        plt.hist(confidence_levels, bins, edgecolor='k')
    elif type(confidence_levels) == dict:
        bins = np.linspace(0, 100, 5)
        plt.hist(confidence_levels.values(), bins, edgecolor='k', label=confidence_levels.keys())
        plt.legend(loc='upper right', fontsize='small')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script',
        usage='evaluate.py <evaluation_file> <validation_resp_format')
    parser.add_argument('test_file', help='File containing test samples')
    parser.add_argument('resp_file', help='File containing prediction responses from the tool')
    parser.add_argument('resp_format', help='Format of response file', type=RespFormat, choices=list(RespFormat))
    parser.add_argument('-i', '--intent', nargs='?', help='Intent for which to plot confidence histogram')
    args = parser.parse_args()

    plot_confidence_levels(args.test_file, args.resp_file, args.resp_format, args.intent)
