import math
import os
from random import *

import data_loader
import model_s2i
import util
import flags
import json
import numpy as np
import tensorflow as tf
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from sklearn.metrics import accuracy_score as scikit_accuracy
from sklearn.metrics import f1_score as scikit_f1
from sklearn.model_selection import StratifiedKFold

a = Random()
a.seed(1)


def dump_flags(FLAGS):
    """ Dumps the TF app flags in a JSON file. Filename will be determined based on the model name.
    Args:
        FLAGS: App flags
    """
    flags_dict = dict()
    for k, v in tf.flags.FLAGS.__flags.items():
        flags_dict[k] = v.value
    filename = FLAGS.scenario_num + '.json'
    with open(os.path.join(FLAGS.hyperparams_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(flags_dict, f, indent=4)


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


def evaluate_validation(capsnet, val_data, FLAGS, sess, epoch, fold, log=False, calculate_learning_curves=False):
    """ Evaluates the model on the validation set
        Args:
            capsnet: CapsNet model
            val_data: validation data dict
            FLAGS: TensorFlow flags
            sess: TensorFlow session in which the training was run
            epoch: current epoch of training
            fold: current fold of K-fold cross-validation
        Returns:
            f_score: intent detection F1 score
            scores['f1']: slot filling F1 score
    """
    x_te = val_data['x_val']
    sentences_length_te = val_data['sentences_len_val']
    y_intents_te = val_data['y_intents_val']
    y_slots_te = val_data['y_slots_val']
    one_hot_intents = val_data['one_hot_intents_val']
    one_hot_slots = val_data['one_hot_slots_val']
    slots_dict = val_data['slots_dict']
    intents_dict = val_data['intents_dict']

    # Define TensorBoard writer
    if log:
        writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation-' + str(fold), sess.graph)
    if calculate_learning_curves:
        writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation-lc', sess.graph)

    total_intent_pred = []
    total_slots_pred = []

    num_samples = len(x_te)
    batch_size = FLAGS.batch_size
    test_batch = int(math.ceil(num_samples / float(batch_size)))
    loss_val = 1
    for i in range(test_batch):
        begin_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_samples)
        batch_te = x_te[begin_index: end_index]
        batch_sentences_len = sentences_length_te[begin_index: end_index]
        batch_intents_one_hot = one_hot_intents[begin_index: end_index]
        batch_slots_one_hot = one_hot_slots[begin_index: end_index]
        batch_size = end_index - begin_index

        # Get predictions for current validation batch
        feed_dict = {capsnet.input_x: batch_te, capsnet.sentences_length: batch_sentences_len,
                     capsnet.encoded_intents: batch_intents_one_hot, capsnet.encoded_slots: batch_slots_one_hot,
                     capsnet.keep_prob: 1.0}
        if FLAGS.use_attention:
            mask = util.calculate_mask(batch_sentences_len, FLAGS.max_sentence_length, batch_size, FLAGS.r)
            feed_dict[capsnet.attention_mask] = mask

        [intent_outputs, slots_outputs, slot_weights_c, cross_entropy_summary,
         margin_loss_summary, loss_summary] = sess.run([
            capsnet.intent_output_vectors, capsnet.slot_output_vectors, capsnet.slot_weights_c,
            capsnet.cross_entropy_val_summary,
            capsnet.margin_loss_val_summary, capsnet.loss_tr_summary],
            feed_dict=feed_dict)
        loss_val = loss_summary
        # Add TensorBoard summaries to FileWriter
        if log:
            writer.add_summary(cross_entropy_summary, epoch * test_batch + i)
            writer.add_summary(margin_loss_summary, epoch * test_batch + i)
            writer.add_summary(loss_summary, epoch * test_batch + i)

        # Modify prediction vectors dimensions to prepare for argmax
        intent_outputs_reduced_dim = tf.squeeze(intent_outputs, axis=[1, 4])
        intent_outputs_norm = util.safe_norm(intent_outputs_reduced_dim)
        slot_weights_c_reduced_dim = tf.squeeze(slot_weights_c, axis=[3, 4])

        [intent_predictions, slot_predictions] = sess.run([intent_outputs_norm, slot_weights_c_reduced_dim])

        # Obtain intent prediction
        te_batch_intent_pred = np.argmax(intent_predictions, axis=1)
        total_intent_pred += np.ndarray.tolist(te_batch_intent_pred)

        # Obtain slots prediction
        te_batch_slots_pred = np.argmax(slot_predictions, axis=2)
        total_slots_pred += (np.ndarray.tolist(te_batch_slots_pred))

    if calculate_learning_curves:
        writer.add_summary(loss_val, fold)
    print('           VALIDATION SET PERFORMANCE        ')
    print('Intent detection')
    intents_acc = scikit_accuracy(y_intents_te, total_intent_pred)
    y_intents_true = np.ndarray.tolist(y_intents_te)
    y_intent_labels_true = [intents_dict[i] for i in y_intents_true]
    y_intent_labels_pred = [intents_dict[i] for i in total_intent_pred]
    intents = sorted(list(set(y_intent_labels_true)))
    f_score = scikit_f1(y_intent_labels_true, y_intent_labels_pred, average='micro', labels=intents)
    # print(classification_report(y_intent_labels_true, y_intent_labels_pred, digits=4))
    print('Intent accuracy %lf' % intents_acc)
    print('F score %lf' % f_score)

    y_slots_te_true = np.ndarray.tolist(y_slots_te)
    y_slot_labels_true = [[slots_dict[slot_idx] for slot_idx in ex] for ex in y_slots_te_true]
    y_slot_labels_pred = [[slots_dict[slot_idx] for slot_idx in ex] for ex in total_slots_pred]
    scores = eval_seq_scores(y_slot_labels_true, y_slot_labels_pred)
    print('Slot filling')
    print('F1 score: %lf' % scores['f1'])
    print('Accuracy: %lf' % scores['accuracy'])
    # print('Precision: %lf' % scores['precision'])
    # print('Recall: %lf' % scores['recall'])

    return f_score, scores['f1']


def generate_batch(n, batch_size):
    """ Generates a set of batch indices
        Args:
            n: total number of samples in set
            batch_size: size of batch
        Returns:
            batch_index: a list of length batch_size containing randomly sampled indices
    """
    batch_index = a.sample(range(n), batch_size)
    return batch_index


def assign_pretrained_word_embedding(sess, embedding, capsnet):
    """ Assigns word embeddings to the CapsNet model
        Args:
            sess: TensorFlow session
            embedding: array containing the word embeddings
            capsnet: CapsNet model
    """
    print('using pre-trained word emebedding.begin...')
    word_embedding_placeholder = tf.placeholder(dtype=tf.float32, shape=embedding.shape)
    sess.run(capsnet.Embedding.assign(word_embedding_placeholder), {word_embedding_placeholder: embedding})
    print('using pre-trained word emebedding.ended...')


def train_cross_validation(model, train_data, val_data, embedding, FLAGS, fold, best_f_score, batches_rand=False, log=False,
                           calculate_learning_curves=False):
    """ Trains the model for one cross-validation fold
        Args:
            train_data: training data dictionary
            val_data: validation data dictionary
            embedding: array containing pre-trained word embeddings
            FLAGS: TensorFlow application flags
            fold: current fold index
            best_f_score: best overall F1 score (across all folds so far)
            batches_rand: whether to random sample mini batches or not (shuffle + seq)
            log: toggle TensorBoard visualization on/off
        Returns:
            best_f_score: best overall F1 score (across all folds so far, including after this one)
            best_f_score_mean_fold: best overall F1 score for this fold
            best_f_score_intent_fold: best intent F1 score for this fold
            best_f_score_slot_fold: best slot F1 score for this fold
    """
    # start
    x_train = train_data['x_tr']
    sentences_length_train = train_data['sentences_len_tr']
    one_hot_intents_train = train_data['one_hot_intents_tr']
    one_hot_slots_train = train_data['one_hot_slots_tr']

    best_f_score_mean_fold = 0.0
    best_f_score_intent_fold = 0.0
    best_f_score_slot_fold = 0.0

    # We must reset the graph to start a brand new training of the model
    tf.reset_default_graph()
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        capsnet = model(FLAGS)

        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        if FLAGS.use_embedding:
            # load pre-trained word embedding
            assign_pretrained_word_embedding(sess, embedding, capsnet)

        # Initial evaluation on validation set
        intent_f_score, slot_f_score = evaluate_validation(capsnet, val_data, FLAGS, sess, epoch=0, fold=fold)
        f_score_mean = (intent_f_score + slot_f_score) / 2
        if f_score_mean > best_f_score:
            best_f_score = f_score_mean
        var_saver = tf.train.Saver()

        if f_score_mean > best_f_score_mean_fold:
            # best mean in this fold, save scores
            best_f_score_mean_fold = f_score_mean
            best_f_score_intent_fold = intent_f_score
            best_f_score_slot_fold = slot_f_score

        if log:
            train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train-fold' + str(fold), sess.graph)

        # Training cycle
        train_sample_num = x_train.shape[0]
        batch_num = int(math.ceil(train_sample_num / FLAGS.batch_size))
        loss_train = 1
        for epoch in range(FLAGS.num_epochs):
            for batch in range(batch_num):
                if batches_rand:
                    batch_index = generate_batch(train_sample_num, FLAGS.batch_size)
                    batch_x = x_train[batch_index]
                    batch_sentences_len = sentences_length_train[batch_index]
                    batch_intents_one_hot = one_hot_intents_train[batch_index]
                    batch_slots_one_hot = one_hot_slots_train[batch_index]
                    batch_size = FLAGS.batch_size

                else:
                    # Training samples are already shuffled in the file
                    begin_index = batch * FLAGS.batch_size
                    end_index = min((batch + 1) * FLAGS.batch_size, train_sample_num)
                    batch_x = x_train[begin_index: end_index]
                    batch_sentences_len = sentences_length_train[begin_index: end_index]
                    batch_intents_one_hot = one_hot_intents_train[begin_index: end_index]
                    batch_slots_one_hot = one_hot_slots_train[begin_index: end_index]
                    batch_size = end_index - begin_index

                feed_dict = {capsnet.input_x: batch_x,
                             capsnet.encoded_intents: batch_intents_one_hot,
                             capsnet.encoded_slots: batch_slots_one_hot,
                             capsnet.sentences_length: batch_sentences_len,
                             capsnet.keep_prob: FLAGS.keep_prob}
                if FLAGS.use_attention:
                    mask = util.calculate_mask(batch_sentences_len, FLAGS.max_sentence_length, batch_size, FLAGS.r)
                    feed_dict[capsnet.attention_mask] = mask

                [_, loss, _, _,
                 cross_entropy_summary, margin_loss_summary,
                 loss_summary] = sess.run([capsnet.train_op, capsnet.loss_val,
                                           capsnet.intent_output_vectors,
                                           capsnet.slot_output_vectors, capsnet.cross_entropy_tr_summary,
                                           capsnet.margin_loss_tr_summary, capsnet.loss_tr_summary],
                                          feed_dict=feed_dict)
                loss_train = loss_summary
                if log:
                    train_writer.add_summary(cross_entropy_summary, batch_num * epoch + batch)
                    train_writer.add_summary(margin_loss_summary, batch_num * epoch + batch)
                    train_writer.add_summary(loss_summary, batch_num * epoch + batch)

            print('------------------epoch : ', epoch, ' Loss: ', loss, '----------------------')
            intent_f_score, slot_f_score = evaluate_validation(capsnet, val_data, FLAGS,
                                                               sess, epoch=epoch + 1, fold=fold, log=log,
                                                               calculate_learning_curves=calculate_learning_curves)
            f_score_mean = (intent_f_score + slot_f_score) / 2
            if f_score_mean > best_f_score:
                # best score overall -> save model
                best_f_score = f_score_mean
                if FLAGS.scenario_num != '':
                    ckpt_dir = FLAGS.ckpt_dir + 'scenario' + FLAGS.scenario_num + '/'
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                else:
                    ckpt_dir = FLAGS.ckpt_dir
                var_saver.save(sess, os.path.join(ckpt_dir, 'model.ckpt'), 1)
            print('Current F score mean', f_score_mean)
            print('Best F score mean', best_f_score)

            if f_score_mean > best_f_score_mean_fold:
                # best mean in this fold, save scores
                best_f_score_mean_fold = f_score_mean
                best_f_score_intent_fold = intent_f_score
                best_f_score_slot_fold = slot_f_score
        if calculate_learning_curves:
            train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train-lc', sess.graph)
            train_writer.add_summary(loss_train, fold)
    return best_f_score, best_f_score_mean_fold, best_f_score_intent_fold, best_f_score_slot_fold


def train(model, data, FLAGS, batches_rand=False, log=False):
    # Dump flags in log file
    dump_flags(FLAGS)

    x_tr = data['x_tr']
    y_intents_tr = data['y_intents_tr']
    y_slots_tr = data['y_slots_tr']
    sentences_length_tr = data['sentences_len_tr']

    one_hot_intents_tr = data['encoded_intents_tr']
    one_hot_slots_tr = data['encoded_slots_tr']

    embedding = data['embedding']

    # k-fold cross-validation
    intent_scores = 0
    slot_scores = 0
    mean_scores = 0
    best_f_score = 0.0
    print('------------------start cross-validation-------------------')
    fold = 1
    for train_index, val_index in StratifiedKFold(FLAGS.n_splits).split(x_tr, y_intents_tr):
        print('FOLD %d' % fold)

        # Split the data according to train_index, val_index
        x_train, x_val = x_tr[train_index], x_tr[val_index]
        y_intents_train, y_intents_val = y_intents_tr[train_index], y_intents_tr[val_index]
        y_slots_train, y_slots_val = y_slots_tr[train_index], y_slots_tr[val_index]
        sentences_length_train, sentences_length_val = sentences_length_tr[train_index], sentences_length_tr[
            val_index]
        one_hot_intents_train, one_hot_intents_val = one_hot_intents_tr[train_index], one_hot_intents_tr[
            val_index]
        one_hot_slots_train, one_hot_slots_val = one_hot_slots_tr[train_index], one_hot_slots_tr[val_index]

        train_data = dict()
        train_data['x_tr'] = x_train
        train_data['y_intents_tr'] = y_intents_train
        train_data['y_slots_tr'] = y_slots_train
        train_data['sentences_len_tr'] = sentences_length_train
        train_data['one_hot_intents_tr'] = one_hot_intents_train
        train_data['one_hot_slots_tr'] = one_hot_slots_train

        val_data = dict()
        val_data['x_val'] = x_val
        val_data['y_intents_val'] = y_intents_val
        val_data['y_slots_val'] = y_slots_val
        val_data['sentences_len_val'] = sentences_length_val
        val_data['one_hot_intents_val'] = one_hot_intents_val
        val_data['one_hot_slots_val'] = one_hot_slots_val
        val_data['slots_dict'] = data['slots_dict']
        val_data['intents_dict'] = data['intents_dict']

        # Train on split
        best_f_score, best_f_score_mean_fold, best_f_score_intent_fold, best_f_score_slot_fold = train_cross_validation(model,
            train_data, val_data, embedding, FLAGS, fold, best_f_score, batches_rand=batches_rand, log=log)

        fold += 1

        # For each fold, add best scores to mean
        intent_scores += best_f_score_intent_fold
        slot_scores += best_f_score_slot_fold
        mean_scores += best_f_score_mean_fold

    # Compute mean score
    mean_intent_score = intent_scores / FLAGS.n_splits
    mean_slot_score = slot_scores / FLAGS.n_splits
    mean_score = mean_scores / FLAGS.n_splits
    print('Mean intent F1 score %lf' % mean_intent_score)
    print('Mean slot F1 score %lf' % mean_slot_score)
    print('Mean F1 score %lf' % mean_score)


def main():
    word2vec_path = '../../romanian_word_vecs/cleaned-vectors-diacritice.vec'

    training_data_path = '../data-capsnets/diacritics/scenario1/train.txt'
    test_data_path = '../data-capsnets/diacritics/scenario1/test.txt'

    # Define the flags
    FLAGS = flags.define_app_flags('1-test-use-attention')

    # Load data
    print('------------------load word2vec begin-------------------')
    w2v = data_loader.load_w2v(word2vec_path)
    print('------------------load word2vec end---------------------')
    data = data_loader.read_datasets(w2v, training_data_path, test_data_path)

    flags.set_data_flags(data)

    train(model_s2i.CapsNetS2I, data, FLAGS, log=True)


if __name__ == '__main__':
    main()
