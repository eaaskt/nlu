import math
import os
from random import *

import data_loader
import model
import flags
import util
import numpy as np
import tensorflow as tf
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from sklearn.metrics import accuracy_score as scikit_accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as scikit_f1
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, labels,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          numbers=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
        Args:
            y_true: true slot labels
            y_pred: predicted slot labels
            labels: list of class labels, will be places on the axes
            title: title of plot
            cmap: colormap
            numbers: True if numbers should be shown inside the confusion matrix, if many classes it is recommended
                     that this is set to False
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
    plt.setp(ax.get_xticklabels(), rotation='vertical', ha='right',
             rotation_mode='anchor')
    plt.tight_layout()

    # Loop over data dimensions and create text annotations.
    if numbers:

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
    # fig.tight_layout()
    return ax


def eval_seq_scores(y_true, y_pred):
    """ Performs sequence evaluation on slot labels
        Args:
            y_true: true slot labels
            y_pred: predicted slot labels
        Returns:
            scores: dict containing the evaluation scores: f1, accuracy, precision, recall
    """
    scores = dict()
    scores['f1'] = f1_score(y_true, y_pred)
    scores['accuracy'] = accuracy_score(y_true, y_pred)
    scores['precision'] = precision_score(y_true, y_pred)
    scores['recall'] = recall_score(y_true, y_pred)
    return scores


def evaluate_test(capsnet, data, FLAGS, sess, log_errs=False, epoch=0):
    """ Evaluates the model on the test set
        Args:
            capsnet: CapsNet model
            data: test data dict
            FLAGS: TensorFlow flags
            sess: TensorFlow session
            log_errs: if True, the intent and slot errors will be logged to a error file and confusion matrices will
                      be displayed
            epoch: current epoch
        Returns:
            f_score: intent detection F1 score
            scores['f1']: slot filling F1 score
    """
    x_te = data['x_te']
    sentences_length_te = data['sentences_len_te']
    y_intents_te = data['y_intents_te']
    y_slots_te = data['y_slots_te']
    slots_dict = data['slots_dict']
    intents_dict = data['intents_dict']
    one_hot_intents = data['one_hot_intents_te']
    one_hot_slots = data['one_hot_slots_te']
    if log_errs:
        x_text_te = data['x_text_te']

    total_intent_pred = []
    total_slots_pred = []

    num_samples = len(x_te)
    batch_size = FLAGS.batch_size
    test_batch = int(math.ceil(num_samples / float(batch_size)))
    for i in range(test_batch):
        begin_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_samples)
        batch_te = x_te[begin_index: end_index]
        batch_sentences_len = sentences_length_te[begin_index: end_index]
        batch_intents_one_hot = one_hot_intents[begin_index: end_index]
        batch_slots_one_hot = one_hot_slots[begin_index: end_index]

        [intent_outputs, slots_outputs, slot_weights_c] = sess.run([
            capsnet.intent_output_vectors, capsnet.slot_output_vectors, capsnet.slot_weights_c],
            feed_dict={capsnet.input_x: batch_te, capsnet.sentences_length: batch_sentences_len,
                       capsnet.encoded_intents: batch_intents_one_hot, capsnet.encoded_slots: batch_slots_one_hot,
                       capsnet.keep_prob: 1.0})

        intent_outputs_reduced_dim = tf.squeeze(intent_outputs, axis=[1, 4])
        intent_outputs_norm = util.safe_norm(intent_outputs_reduced_dim)
        slot_weights_c_reduced_dim = tf.squeeze(slot_weights_c, axis=[3, 4])

        [intent_predictions, slot_predictions] = sess.run([intent_outputs_norm, slot_weights_c_reduced_dim])

        te_batch_intent_pred = np.argmax(intent_predictions, axis=1)
        total_intent_pred += np.ndarray.tolist(te_batch_intent_pred)

        te_batch_slots_pred = np.argmax(slot_predictions, axis=2)
        total_slots_pred += (np.ndarray.tolist(te_batch_slots_pred))

    print('           TEST SET PERFORMANCE        ')
    print('Intent detection')
    intents_acc = scikit_accuracy(y_intents_te, total_intent_pred)
    y_intents_true = np.ndarray.tolist(y_intents_te)
    y_intent_labels_true = [intents_dict[i] for i in y_intents_true]
    y_intent_labels_pred = [intents_dict[i] for i in total_intent_pred]
    intents = sorted(list(set(y_intent_labels_true)))
    f_score = scikit_f1(y_intent_labels_true, y_intent_labels_pred, average='micro', labels=intents)
    print(classification_report(y_intent_labels_true, y_intent_labels_pred, digits=4))
    print('Intent accuracy %lf' % intents_acc)
    print('F score %lf' % f_score)

    y_slots_te_true = np.ndarray.tolist(y_slots_te)
    y_slot_labels_true = [[slots_dict[slot_idx] for slot_idx in ex] for ex in y_slots_te_true]
    y_slot_labels_pred = [[slots_dict[slot_idx] for slot_idx in ex] for ex in total_slots_pred]
    scores = eval_seq_scores(y_slot_labels_true, y_slot_labels_pred)
    print('Slot filling')
    print('F1 score: %lf' % scores['f1'])
    print('Accuracy: %lf' % scores['accuracy'])
    print('Precision: %lf' % scores['precision'])
    print('Recall: %lf' % scores['recall'])

    # Write errors to error log
    if log_errs:
        if FLAGS.scenario_num != '':
            errors_dir = FLAGS.errors_dir + 'scenario' + FLAGS.scenario_num + '/'
            if not os.path.exists(errors_dir):
                os.makedirs(errors_dir)
        else:
            errors_dir = FLAGS.errors_dir

        plot_confusion_matrix(y_intent_labels_true, y_intent_labels_pred, labels=intents,
                              title='Confusion matrix', normalize=True, numbers=False)
        plt.show()

        # For super-class confusion mat
        intent_classes = {'aprindeLumina': 'lumina',
                          'cresteIntensitateLumina': 'lumina',
                          'cresteTemperatura': 'temperatura',
                          'opresteMuzica': 'media',
                          'opresteTV': 'media',
                          'pornesteTV': 'media',
                          'puneMuzica': 'media',
                          'scadeIntensitateLumina': 'lumina',
                          'scadeTemperatura': 'temperatura',
                          'schimbaCanalTV': 'media',
                          'schimbaIntensitateMuzica': 'media',
                          'seteazaTemperatura': 'temperatura',
                          'stingeLumina': 'lumina',
                          'x': 'x'}

        if 'x' in y_intent_labels_true or 'x' in y_intent_labels_pred:
            intent_classes_labels = ['lumina', 'media', 'temperatura', 'x']
        else:
            intent_classes_labels = ['lumina', 'media', 'temperatura']
        intent_classes_true = [intent_classes[intent] for intent in y_intent_labels_true]
        intent_classes_pred = [intent_classes[intent] for intent in y_intent_labels_pred]
        plot_confusion_matrix(intent_classes_true, intent_classes_pred, labels=intent_classes_labels,
                              title='Confusion matrix', normalize=True, numbers=True)
        plt.show()

        incorrect_intents = {}
        i = 0
        for t, p in zip(y_intent_labels_true, y_intent_labels_pred):
            if t != p:
                if t not in incorrect_intents:
                    incorrect_intents[t] = []
                incorrect_intents[t].append((' '.join(x_text_te[i]), p))
            i += 1

        with open(os.path.join(errors_dir, 'errors.txt'), 'w', encoding='utf-8') as f:
            f.write('INTENT ERRORS\n')
            for k, v in incorrect_intents.items():
                f.write(k + '\n')
                for intent in v:
                    f.write('{} -> {}\n'.format(intent[0], intent[1]))
                f.write('\n')

            # View incorrect slot sequences
            f.write('SLOT ERRORS\n')
            i = 0
            for v, p in zip(y_slot_labels_true, y_slot_labels_pred):
                if v != p:
                    f.write(' '.join(x_text_te[i]) + '\n')
                    f.write(str(v) + '\n')
                    f.write(str(p) + '\n')
                    f.write('\n')
                i += 1

    return f_score, scores['f1']


def test(data, FLAGS):

    # Testing
    test_data = dict()
    test_data['x_te'] = data['x_te']
    test_data['x_text_te'] = data['x_text_te']
    test_data['y_intents_te'] = data['y_intents_te']
    test_data['y_slots_te'] = data['y_slots_te']
    test_data['sentences_len_te'] = data['sentences_len_te']
    test_data['slots_dict'] = data['slots_dict']
    test_data['intents_dict'] = data['intents_dict']
    test_data['one_hot_intents_te'] = data['encoded_intents_te']
    test_data['one_hot_slots_te'] = data['encoded_slots_te']

    tf.reset_default_graph()
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        capsnet = model.CapsNet(FLAGS)
        if FLAGS.scenario_num != '':
            ckpt_dir = FLAGS.ckpt_dir + 'scenario' + FLAGS.scenario_num + '/'
        else:
            ckpt_dir = FLAGS.ckpt_dir
        if os.path.exists(ckpt_dir):
            print('Restoring Variables from Checkpoint for testing')
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            intent_f_score, slot_f_score = evaluate_test(capsnet, test_data, FLAGS, sess, log_errs=True)
            print('Intent F1: %lf' % intent_f_score)
            print('Slot F1: %lf' % slot_f_score)
        else:
            print('No trained model exists in checkpoint dir!')


def main():
    word2vec_path = '../../romanian_word_vecs/cc.ro.300.vec'

    training_data_path = '../data-capsnets/scenario0/train.txt'
    test_data_path = '../data-capsnets/scenario0/test.txt'

    FLAGS = flags.define_app_flags('0')

    # Load data
    print('------------------load word2vec begin-------------------')
    w2v = data_loader.load_w2v(word2vec_path)
    print('------------------load word2vec end---------------------')
    data = data_loader.read_datasets(w2v, training_data_path, test_data_path, test=True)
    flags.set_data_flags(data)

    test(data, FLAGS)


if __name__ == '__main__':
    main()
