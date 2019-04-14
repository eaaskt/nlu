import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as scikit_f1


class RespFormat(Enum):
    wit = 'wit'
    rasa = 'rasa'

    def __str__(self):
        return self.value


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
    plt.setp(ax.get_xticklabels(), rotation='vertical', ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
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
            print('%d != %d' % (v['id'], r['id']))
            return
        y_true.append(v['seq_labels'])
        y_pred.append(r['labels'])

        if verbose:
            print("Id - %d" % v['id'])
            print(v['seq_labels'])
            print(r['labels'])
            print()

    return y_true, y_pred


def extract_intents_wit(validation_file, response_file, verbose=False):
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

        text = v['text']
        found = False
        for entity in v['entities']:
            if entity['entity'] == 'intent':
                i_true.append((entity['value'], text))
                found = True
                if entity['value'] not in intents:
                    intents.append(entity['value'])
                if verbose:
                    print('Id - %d' % v['id'])
                    print(entity['value'])
                break
        if not found:
            i_true.append((' ', text))
            intents.append(' ')

        if 'intent' not in r['entities']:
            i_pred.append(('x', text))
            intents.append('x')
        else:
            i_pred.append((r['entities']['intent'][0]['value'], text))
            if r['entities']['intent'][0]['value'] not in intents:
                intents.append(r['entities']['intent'][0]['value'])

            if verbose:
                print(r['entities']['intent'])

    return i_true, i_pred, intents


def extract_intents_rasa(validation_file, response_file, verbose=False):
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

        text = v['text']
        found = False
        for entity in v['entities']:
            if entity['entity'] == 'intent':
                i_true.append((entity['value'], text))
                found = True
                if entity['value'] not in intents:
                    intents.append(entity['value'])
                if verbose:
                    print('Id - %d' % v['id'])
                    print(entity['value'])
                break
        if not found:
            i_true.append((' ', text))
            intents.append(' ')

        if 'intent' not in r:
            i_pred.append(('x', text))
            intents.append('x')
        else:
            i_pred.append((r['intent']['name'], text))
            if r['intent']['name'] not in intents:
                intents.append(r['intent']['name'])

            if verbose:
                print(r['intent'])

    return i_true, i_pred, intents


def create_conllu_file(validation_file, response_file, i):
    with open(validation_file, errors='replace') as f:
        val_data = json.load(f)
    with open(response_file, errors='replace') as f:
        resp_data = json.load(f)

    filename = 'output{}.txt'.format(i)
    with open(filename, 'w') as f:

        for v, r in zip(val_data, resp_data):
            if v['id'] != r['id']:
                print('Ids not matching! Something went wrong in the response file')
                return
            words = v['text'].split(' ')
            for w, tl, pl in zip(words, v['seq_labels'], r['labels']):
                f.write('%s - %s %s\n' % (w, tl, pl))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script',
        usage='evaluate.py <evaluation_file> <validation_resp_format')
    parser.add_argument('evaluation_file', help='File containing pairs validation - response files')
    parser.add_argument('resp_format', help='Format of response file', type=RespFormat, choices=list(RespFormat))
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
    # total_y_true = []
    # total_y_pred = []
    # for f1, f2 in files:
    #     y_true, y_pred = extract_slot_labels(f1, f2)
    #
    #     print(f1.split('\\')[-1])
    #     scores = eval_seq_scores(y_true, y_pred)
    #     print('F1 score: %lf' % scores['f1'])
    #     print('Accuracy: %lf' % scores['accuracy'])
    #     print('Precision: %lf' % scores['precision'])
    #     print('Recall: %lf' % scores['recall'])
    #     print(scores['clf_report'])
    #
    #     total_y_true += y_true
    #     total_y_pred += y_pred
    #
    # # Overall slot filling evaluation
    # scores = eval_seq_scores(total_y_true, total_y_pred)
    # print('Overall evaluation')
    # print('F1 score: %lf' % scores['f1'])
    # print('Accuracy: %lf' % scores['accuracy'])
    # print('Precision: %lf' % scores['precision'])
    # print('Recall: %lf' % scores['recall'])
    # print(scores['clf_report'])

    # Intent detection evaluation
    intents_true = []
    intents_pred = []
    intents_true_with_text = []
    intents_pred_with_text = []
    intents = []
    for f1, f2 in files:
        if args.resp_format == RespFormat.wit:
            i_true_with_text, i_pred_with_text, intents_list = extract_intents_wit(f1, f2)
        elif args.resp_format == RespFormat.rasa:
            i_true_with_text, i_pred_with_text, intents_list = extract_intents_rasa(f1, f2)

        i_true = [i for i,_ in i_true_with_text]
        i_pred = [i for i, _ in i_pred_with_text]

        intents_true += i_true
        intents_pred += i_pred
        intents_true_with_text += i_true_with_text
        intents_pred_with_text += i_pred_with_text

        intents = list(set(intents).union(intents_list))

    intents = sorted(intents)
    plot_confusion_matrix(intents_true, intents_pred, labels=intents,
                          title='Confusion matrix', normalize=True)
    plt.show()

    # F1 score per intent
    print('F1 score per intent')
    scores = scikit_f1(intents_true, intents_pred, average=None, labels=intents)
    for i, s in zip(intents, scores):
        print("%s : %lf" % (i, s))

    print('Overall F1 score (micro-averaged)')
    scores = scikit_f1(intents_true, intents_pred, average='micro', labels=intents)
    print(scores)

    # View incorrectly predicted intents
    # incorrect_intents = {}
    # print()
    # print('Incorrect intent predictions')
    # for t, p in zip(intents_true_with_text, intents_pred_with_text):
    #     if t[0] != p[0]:
    #         # print('Text: ' + t[1])
    #         # print('True intent: ' + t[0])
    #         # print('Pred intent: ' + p[0] + '\n')
    #         if t[0] not in incorrect_intents:
    #             incorrect_intents[t[0]] = []
    #         incorrect_intents[t[0]].append((t[1], p[0]))
    #
    # for k, v in incorrect_intents.items():
    #     print(k)
    #     for intent in v:
    #         print('{} -> {}'.format(intent[0], intent[1]))

    # for f1, f2 in files:
    #     with open(f1, errors='replace') as f:
    #         val_data = json.load(f)
    #     with open(f2, errors='replace') as f:
    #         pred_data = json.load(f)
    #     for v, p in zip(val_data, pred_data):
    #         for entity in v['entities']:
    #             if entity['entity'] == 'intent' and entity['value'] == 'SearchCreativeWork':
    #                 print(v['text'])
    #                 print(v['seq_labels'])
    #                 print(p['labels'])
    #                 print()
