import argparse
from enum import Enum
import json

import numpy as np
import matplotlib.pyplot as plt

from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix


class Dataset(Enum):
    snips = 'snips'
    atis = 'atis'

    def __str__(self):
        return self.value


def count_intents(validation_files):
    intent_count = {}
    for val_f in validation_files:
        with open(val_f, errors='replace') as f:
            val_data = json.load(f)

        for v in val_data:
            intent = None
            for entity in v['entities']:
                if entity['entity'] == 'intent':
                    intent = entity['value']
                    if intent not in intent_count:
                        intent_count[intent] = 1
                    else:
                        intent_count[intent] += 1
                    break
    return intent_count


def plot_distribution(data, labels, plot_number, plot_title, ylim_top=None, save_file=False, filename=None):
    y_pos = np.arange(len(labels))
    plt.figure(plot_number)
    plt.bar(y_pos, data, color='blue', align='center')
    plt.xticks(y_pos, labels, rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.title(plot_title)
    if ylim_top is not None:
        plt.ylim(top=ylim_top)
    if save_file and filename is not None:
        plt.savefig('pics/%s%s.png' % (filename, plot_number))


def get_slot_statistics(validation_f):
    with open(validation_f, 'r') as f:
        val_data = json.load(f)
    slot_count = {}
    for v in val_data:
        for entity in v['entities']:
            entity_value = entity['entity']
            if entity_value != 'intent':
                if entity_value not in slot_count:
                    slot_count[entity_value] = 1
                else:
                    slot_count[entity_value] += 1

    return slot_count


def split_list(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_intent(data_file):
    with open(data_file, 'r') as f:
        val_data = json.load(f)
    intent = None
    for v in val_data:
        for entity in v['entities']:
            if entity['entity'] == 'intent':
                intent = entity['value']
                break

    return intent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validation data analysis script',
        usage='analyze_data.py <evaluation_file> <dataset>')
    parser.add_argument('evaluation_file', help='File containing files to be analyzed')
    parser.add_argument('dataset', help='atis/snips', type=Dataset, choices=list(Dataset))
    parser.add_argument('--val', help='Analyze validation data', action='store_true')
    args = parser.parse_args()

    files = []
    with open(args.evaluation_file, 'r') as f:
        for line in f:
            file_line = line.replace('\n', '').split(' ')
            if len(file_line) > 1:
                print('Incorrect format for the eval files %s' % str(file_line))
                exit(1)
            files.append(file_line[0])

    if args.dataset == Dataset.atis:
        f = files[0]
        print('File: %s' % f)
        if args.val:
            intent_filename = 'atis_validation_intent_distribution'
            intent_plot_title = 'ATIS Intent class distribution - validation set'
            slots_filename = 'atis_validation_slots_distribution'
            slots_plot_title = 'ATIS Slots distribution - validation set'
        else:
            intent_filename = 'atis_training_intent_distribution'
            intent_plot_title = 'ATIS Intent class distribution - training set'
            slots_filename = 'atis_training_slots_distribution'
            slots_plot_title = 'ATIS Slots distribution - training set'

        # Plot intent class distribution
        intent_counts = count_intents([f])
        plot_distribution(intent_counts.values(), intent_counts.keys(), plot_number=1,
                          plot_title=intent_plot_title, save_file=True, filename=intent_filename)

        slot_counts = get_slot_statistics(f)
        nr_slot_figs = 3
        slots_split = split_list(list(slot_counts.keys()), nr_slot_figs)
        slots_counts_split = split_list(list(slot_counts.values()), nr_slot_figs)
        max_slot_count = max(slot_counts.values())
        print(max_slot_count)
        for i in range(2, 2 + nr_slot_figs):
            plot_distribution(slots_counts_split[i - 2], slots_split[i - 2], plot_number=i,
                              plot_title=slots_plot_title, ylim_top=max_slot_count,
                              save_file=True, filename=slots_filename)

        plt.show()

    elif args.dataset == Dataset.snips:
        if args.val:
            intent_filename = 'snips_validation_intent_distribution'
            intent_plot_title = 'SNIPS Intent class distribution - validation set'
            slots_filename = 'snips_validation_slots_distribution'
            slots_plot_title = 'SNIPS Slots distribution - validation set'
        else:
            intent_filename = 'snips_training_intent_distribution'
            intent_plot_title = 'SNIPS Intent class distribution - training set'
            slots_filename = 'snips_training_slots_distribution'
            slots_plot_title = 'SNIPS Slots distribution - training set'

        # Plot intent class distribution
        intent_counts = count_intents(files)
        plot_distribution(intent_counts.values(), intent_counts.keys(), plot_number=1,
                          plot_title=intent_plot_title, save_file=True, filename=intent_filename)

        for f in files:
            print('File: %s' % f)
            intent = get_intent(f)
            intent_slots_filename = intent + '_' + slots_filename
            intent_slots_plot_title = slots_plot_title + ' ({})'.format(intent)

            slot_counts = get_slot_statistics(f)
            nr_slot_figs = 1
            slots_split = split_list(list(slot_counts.keys()), nr_slot_figs)
            slots_counts_split = split_list(list(slot_counts.values()), nr_slot_figs)
            max_slot_count = max(slot_counts.values())
            print(max_slot_count)
            for i in range(2, 2 + nr_slot_figs):
                plot_distribution(slots_counts_split[i - 2], slots_split[i - 2], plot_number=i,
                                  plot_title=intent_slots_plot_title, ylim_top=max_slot_count,
                                  save_file=True, filename=intent_slots_filename)
            plt.show()
