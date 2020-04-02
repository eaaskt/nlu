import os

import random
import data_loader
import model
import train
import test
import flags
import csv
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def write_csv(run_results):
    with open('percentage_split_results.csv', 'w') as f:
        intent_title_line = ['Intent detection']
        header_line = ['Scenario', 'Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Std']
        writer = csv.writer(f)
        writer.writerow(intent_title_line)
        writer.writerow(header_line)
        for k, v in run_results.items():
            scenario_num = [k]
            runs_scores = v['intent_scores']
            std = ['{:.3f}'.format(v['std_intent'])]
            l = scenario_num + runs_scores + std
            writer.writerow(l)

        intent_title_line = ['Slot filling']
        header_line = ['Scenario', 'Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Std']
        writer = csv.writer(f)
        writer.writerow(intent_title_line)
        writer.writerow(header_line)
        for k, v in run_results.items():
            scenario_num = [k]
            runs_scores = v['slot_scores']
            std = ['{:.3f}'.format(v['std_slots'])]
            l = scenario_num + runs_scores + std
            writer.writerow(l)


def train_and_test(train_data, val_data, test_data, embedding, FLAGS, fold):
    """ Trains the model and evaluates performance on test data
        Args:
            train_data: training data dictionary
            val_data: validation data dictionary
            test_data: test data dictionary
            embedding: array containing pre-trained word embeddings
            FLAGS: TensorFlow application flags
            fold: current fold index
        Returns:
            intent_f_score: intent detection F1 score on the test set
            slot_f_score: slot filling F1 score on the test set
    """
    # start
    x_train = train_data['x_tr']
    sentences_length_train = train_data['sentences_len_tr']
    one_hot_intents_train = train_data['one_hot_intents_tr']
    one_hot_slots_train = train_data['one_hot_slots_tr']

    # We must reset the graph to start a brand new training of the model
    tf.reset_default_graph()
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        capsnet = model.CapsNet(FLAGS)

        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        if FLAGS.use_embedding:
            # load pre-trained word embedding
            train.assign_pretrained_word_embedding(sess, embedding, capsnet)

        best_f_score = 0.0
        intent_f_score, slot_f_score = train.evaluate_validation(capsnet, val_data, FLAGS, sess, 0, 0)
        f_score_mean = (intent_f_score + slot_f_score) / 2
        if f_score_mean > best_f_score:
            # save model
            best_f_score = f_score_mean
        var_saver = tf.train.Saver()

        # Training cycle
        train_sample_num = x_train.shape[0]
        batch_num = int(train_sample_num / FLAGS.batch_size)
        for epoch in range(FLAGS.num_epochs):
            for batch in range(batch_num):
                batch_index = train.generate_batch(train_sample_num, FLAGS.batch_size)
                batch_x = x_train[batch_index]
                batch_sentences_len = sentences_length_train[batch_index]
                batch_intents_one_hot = one_hot_intents_train[batch_index]
                batch_slots_one_hot = one_hot_slots_train[batch_index]

                [_, loss, intent_pred, slot_pred,
                 cross_entropy_summary, margin_loss_summary,
                 loss_summary] = sess.run([capsnet.train_op, capsnet.loss_val,
                                           capsnet.intent_output_vectors,
                                           capsnet.slot_output_vectors, capsnet.cross_entropy_tr_summary,
                                           capsnet.margin_loss_tr_summary, capsnet.loss_tr_summary],
                                          feed_dict={capsnet.input_x: batch_x,
                                                     capsnet.encoded_intents: batch_intents_one_hot,
                                                     capsnet.encoded_slots: batch_slots_one_hot,
                                                     capsnet.sentences_length: batch_sentences_len,
                                                     capsnet.keep_prob: 0.8})


            print("------------------epoch : ", epoch, " Loss: ", loss, "----------------------")
            intent_f_score, slot_f_score = train.evaluate_validation(capsnet, val_data, FLAGS, sess, 0, 0)

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

        print('----- Model training done ------')

        # Test model
        intent_f1, slots_f1 = test.evaluate_test(capsnet, test_data, FLAGS, sess, log_errs=False)

    return intent_f1, slots_f1


def main():
    training_data_paths = [
        '../data-capsnets/scenario0/train.txt',
        '../data-capsnets/scenario1/train.txt',
        # '../data-capsnets/scenario2/train.txt',
        # '../data-capsnets/scenario3.1/train.txt',
        # '../data-capsnets/scenario3.2/train.txt',
        # '../data-capsnets/scenario3.3/train.txt',
    ]
    test_data_paths = [
        '../data-capsnets/scenario0/test.txt',
        '../data-capsnets/scenario1/test.txt',
        # '../data-capsnets/scenario2/test.txt',
        # '../data-capsnets/scenario3.1/test.txt',
        # '../data-capsnets/scenario3.2/test.txt',
        # '../data-capsnets/scenario3.3/test.txt',
    ]
    scenario_nums = [
        '0_perc',
        '1_perc',
        # '2_perc',
        # '31_perc',
        # '32_perc',
        # '33_perc',
    ]

    run_results = dict()

    for i in range(len(training_data_paths)):
        # Define the flags
        FLAGS = flags.define_app_flags(scenario_num=scenario_nums[i])

        # Load data
        data = data_loader.read_datasets(training_data_paths[i], test_data_paths[i], test=True)
        x_tr = data['x_tr']
        y_intents_tr = data['y_intents_tr']
        y_slots_tr = data['y_slots_tr']
        sentences_length_tr = data['sentences_len_tr']

        one_hot_intents_tr = data['encoded_intents_tr']
        one_hot_slots_tr = data['encoded_slots_tr']

        x_te = data['x_te']
        x_text_te = data['x_text_te']
        y_intents_te = data['y_intents_te']
        y_slots_te = data['y_slots_te']
        sentences_length_te = data['sentences_len_te']
        slots_dict = data['slots_dict']
        intents_dict = data['intents_dict']
        one_hot_intents_te = data['encoded_intents_te']
        one_hot_slots_te = data['encoded_slots_te']

        embedding = data['embedding']

        flags.set_data_flags(data)

        # percentage split
        print('------------------start subsampling-------------------')
        subsampling_runs = 5
        fold = 1
        intent_scores = []
        slot_scores = []
        for train_index, val_index in StratifiedShuffleSplit(n_splits=subsampling_runs, train_size=0.9).split(x_tr, y_intents_tr):
            print('RUN %d\n' % fold)

            # Split the data according to train_index
            x_train = x_tr[train_index]
            y_intents_train = y_intents_tr[train_index]
            y_slots_train = y_slots_tr[train_index]
            sentences_length_train = sentences_length_tr[train_index]
            one_hot_intents_train = one_hot_intents_tr[train_index]
            one_hot_slots_train = one_hot_slots_tr[train_index]

            x_val = x_tr[val_index]
            y_intents_val = y_intents_tr[val_index]
            y_slots_val = y_slots_tr[val_index]
            sentences_length_val = sentences_length_tr[val_index]
            one_hot_intents_val = one_hot_intents_tr[val_index]
            one_hot_slots_val = one_hot_slots_tr[val_index]

            # Randomly sample 10% of test data
            _, test_index = StratifiedShuffleSplit(n_splits=2, test_size=0.1).split(x_te, y_intents_te)
            # Split the data according to test_index
            test_index = test_index[0]
            x_test = x_te[test_index]
            y_intents_test = y_intents_te[test_index]
            y_slots_test = y_slots_te[test_index]
            sentences_length_test = sentences_length_te[test_index]
            one_hot_intents_test = one_hot_intents_te[test_index]
            one_hot_slots_test = one_hot_slots_te[test_index]

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
            val_data['slots_dict'] = slots_dict
            val_data['intents_dict'] = intents_dict

            test_data = dict()
            test_data['x_te'] = x_test
            test_data['y_intents_te'] = y_intents_test
            test_data['y_slots_te'] = y_slots_test
            test_data['sentences_len_te'] = sentences_length_test
            test_data['one_hot_intents_te'] = one_hot_intents_test
            test_data['one_hot_slots_te'] = one_hot_slots_test
            test_data['slots_dict'] = slots_dict
            test_data['intents_dict'] = intents_dict

            # Train and test on split
            f_intent, f_slot = train_and_test(train_data, val_data, test_data, embedding, FLAGS, fold)

            fold += 1

            # For each iteration, add score to list
            intent_scores.append(f_intent)
            slot_scores.append(f_slot)

        # Compute standard deviation
        std_intent = np.std(intent_scores)
        std_slot = np.std(slot_scores)
        print('Std intent F1 score %lf' % std_intent)
        print('Std slot F1 score %lf' % std_slot)

        run_results[scenario_nums[i]] = dict()
        run_results[scenario_nums[i]]['intent_scores'] = intent_scores
        run_results[scenario_nums[i]]['slot_scores'] = slot_scores
        run_results[scenario_nums[i]]['std_intent'] = std_intent
        run_results[scenario_nums[i]]['std_slots'] = std_slot

        flags.del_all_flags(FLAGS)

    write_csv(run_results)


if __name__ == '__main__':
    main()