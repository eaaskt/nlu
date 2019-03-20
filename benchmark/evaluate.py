import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def eval_seq_scores(y_true, y_pred):
    scores = dict()
    scores['f1'] = f1_score(y_true, y_pred)
    scores['accuracy'] = accuracy_score(y_true, y_pred)
    scores['precision'] = precision_score(y_true, y_pred)
    scores['recall'] = recall_score(y_true, y_pred)
    scores['clf_report'] = classification_report(y_true, y_pred)
    return scores


def extract_slot_labels(validation_file, response_file, verbose=False):
    with open(validation_file, errors='replace') as f:
        val_data = json.load(f)
    with open(response_file, errors='replace') as f:
        resp_data = json.load(f)

    y_true = []
    y_pred = []
    for v, r in zip(val_data, resp_data):
        if v['id'] != r['id']:
            print('Ids not matching! Something went wrong in the response file')
            return
        y_true.append(v['seq_labels'])
        y_pred.append(r['labels'])

        if verbose:
            print("Id - %d" % v['id'])
            print(v['seq_labels'])
            print(r['labels'])
            print()

    return y_true, y_pred


def extract_intents(validation_file, response_file, verbose=False):
    with open(validation_file, errors='replace') as f:
        val_data = json.load(f)
    with open(response_file, errors='replace') as f:
        resp_data = json.load(f)

    i_true = []
    i_pred = []
    intents = []
    for v, r in zip(val_data, resp_data):
        if v['id'] != r['id']:
            print('Ids not matching! Something went wrong in the response file')
            return

        found = False
        for entity in v['entities']:
            if entity['entity'] == 'intent':
                i_true.append(entity['value'])
                found = True
                if entity['value'] not in intents:
                    intents.append(entity['value'])
                if verbose:
                    print('Id - %d' % v['id'])
                    print(entity['value'])
                break
        if not found:
            i_true.append(' ')
            intents.append(' ')

        if 'intent' not in r['entities']:
            i_pred.append('x')
            intents.append('x')
        else:
            i_pred.append(r['entities']['intent'][0]['value'])
            if r['entities']['intent'][0]['value'] not in intents:
                intents.append(r['entities']['intent'][0]['value'])

            if verbose:
                print(r['entities']['intent'])

    return i_true, i_pred, intents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script',
        usage='evaluate.py <evaluation_file>')
    parser.add_argument('evaluation_file', help='File containing pairs validation - response files')
    args = parser.parse_args()

    files = []
    with open(args.evaluation_file, 'r') as f:
        for line in f:
            file_line = line.replace('\n', '').split(' ')
            if len(file_line) > 2:
                print('Incorrect format for the eval files %s' % str(file_line))
                exit(1)
            files.append((file_line[0], file_line[1]))

    # Slot filling evaluation
    total_y_true = []
    total_y_pred = []
    for f1, f2 in files:
        y_true, y_pred = extract_slot_labels(f1, f2)

        print(f1.split('\\')[-1])
        scores = eval_seq_scores(y_true, y_pred)
        print('F1 score: %lf' % scores['f1'])
        print('Accuracy: %lf' % scores['accuracy'])
        print('Precision: %lf' % scores['precision'])
        print('Recall: %lf' % scores['recall'])
        print(scores['clf_report'])

        total_y_true += y_true
        total_y_pred += y_pred

    # Overall slot filling evaluation
    scores = eval_seq_scores(total_y_true, total_y_pred)
    print('Overall evaluation')
    print('F1 score: %lf' % scores['f1'])
    print('Accuracy: %lf' % scores['accuracy'])
    print('Precision: %lf' % scores['precision'])
    print('Recall: %lf' % scores['recall'])
    print(scores['clf_report'])

    # Intent detection evaluation
    intents_true = []
    intents_pred = []
    intents = []
    for f1, f2 in files:
        i_true, i_pred, intents_list = extract_intents(f1, f2)

        intents_true += i_true
        intents_pred += i_pred
        intents = list(set(intents).union(intents_list))

    plot_confusion_matrix(intents_true, intents_pred, labels=intents,
                          title='Confusion matrix, without normalization')
    plt.show()
