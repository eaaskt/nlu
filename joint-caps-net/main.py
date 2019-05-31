import os
from random import *
import data_loader
import numpy as np
import tensorflow as tf
import model
import math
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as scikit_f1
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score
from sklearn.model_selection import StratifiedKFold

a = Random()
a.seed(1)


def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    _, max_sentence_length = data['x_tr'].shape
    intents_number = len(data['intents_dict'])
    slots_number = len(data['slots_dict'])

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_float("keep_prob", 0.8, "embedding dropout keep rate")
    tf.app.flags.DEFINE_integer("n_splits", 3, "Number of cross-validation splits")
    tf.app.flags.DEFINE_integer("hidden_size", 32, "embedding vector size")
    tf.app.flags.DEFINE_integer("batch_size", 64, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("num_epochs", 2, "num of epochs")
    tf.app.flags.DEFINE_integer("vocab_size", vocab_size, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("max_sentence_length", max_sentence_length, "max number of words in one sentence")
    tf.app.flags.DEFINE_integer("intents_nr", intents_number, "intents_number")  #
    tf.app.flags.DEFINE_integer("slots_nr", slots_number, "slots_number")  #
    tf.app.flags.DEFINE_integer("word_emb_size", word_emb_size, "embedding size of word vectors")
    tf.app.flags.DEFINE_string("ckpt_dir", './saved_models/', "check point dir")
    tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
    tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    tf.app.flags.DEFINE_float("margin", 1.0, "ranking loss margin")
    tf.app.flags.DEFINE_integer("slot_routing_num", 2, "slot routing num")
    tf.app.flags.DEFINE_integer("intent_routing_num", 2, "intent routing num")
    tf.app.flags.DEFINE_integer("re_routing_num", 2, "re routing num")
    tf.app.flags.DEFINE_integer("intent_output_dim", 32, "intent output dimension")
    tf.app.flags.DEFINE_integer("slot_output_dim", 64, "slot output dimension")
    tf.app.flags.DEFINE_boolean("save_model", False, "save model to disk")
    tf.app.flags.DEFINE_boolean("test", False, "Evaluate model on test data")
    tf.app.flags.DEFINE_string("summaries_dir", './logs', "tensorboard summaries")

    return FLAGS


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def eval_seq_scores(y_true, y_pred):
    scores = dict()
    scores['f1'] = f1_score(y_true, y_pred)
    scores['accuracy'] = accuracy_score(y_true, y_pred)
    scores['precision'] = precision_score(y_true, y_pred)
    scores['recall'] = recall_score(y_true, y_pred)
    return scores


def evaluate_test(capsnet, data, FLAGS, sess):
    x_te = data['x_te']
    sentences_length_te = data['sentences_len_te']
    y_intents_te = data['y_intents_te']
    y_slots_te = data['y_slots_te']
    slots_dict = data['slots_dict']
    intents_dict = data['intents_dict']

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

        [intent_outputs, slots_outputs, slot_weights_c] = sess.run([
            capsnet.intent_output_vectors, capsnet.slot_output_vectors, capsnet.slot_weights_c],
            feed_dict={capsnet.input_x: batch_te, capsnet.sentences_length: batch_sentences_len})

        intent_outputs_reduced_dim = tf.squeeze(intent_outputs)
        intent_outputs_norm = safe_norm(intent_outputs_reduced_dim)
        slot_weights_c_reduced_dim = tf.squeeze(slot_weights_c)

        [intent_predictions, slot_predictions] = sess.run([intent_outputs_norm, slot_weights_c_reduced_dim])

        te_batch_intent_pred = np.argmax(intent_predictions, axis=1)
        total_intent_pred += np.ndarray.tolist(te_batch_intent_pred)

        te_batch_slots_pred = np.argmax(slot_predictions, axis=2)
        total_slots_pred += (np.ndarray.tolist(te_batch_slots_pred))

    print("           TEST SET PERFORMANCE        ")
    print("Intent detection")
    intents_acc = accuracy_score(y_intents_te, total_intent_pred)
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
    print("Slot filling")
    print('F1 score: %lf' % scores['f1'])
    print('Accuracy: %lf' % scores['accuracy'])
    print('Precision: %lf' % scores['precision'])
    print('Recall: %lf' % scores['recall'])

    return f_score, scores['f1']


def evaluate_validation(capsnet, val_data, FLAGS, sess, epoch):
    x_te = val_data['x_val']
    sentences_length_te = val_data['sentences_len_val']
    y_intents_te = val_data['y_intents_val']
    y_slots_te = val_data['y_slots_val']
    one_hot_intents = val_data['one_hot_intents_val']
    one_hot_slots = val_data['one_hot_slots_val']
    slots_dict = val_data['slots_dict']
    intents_dict = val_data['intents_dict']

    writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation', sess.graph)

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

        [intent_outputs, slots_outputs, slot_weights_c, cross_entropy_summary,
         margin_loss_summary, loss_summary] = sess.run([
            capsnet.intent_output_vectors, capsnet.slot_output_vectors, capsnet.slot_weights_c,
            capsnet.cross_entropy_val_summary,
            capsnet.margin_loss_val_summary, capsnet.loss_val_summary],
            feed_dict={capsnet.input_x: batch_te, capsnet.sentences_length: batch_sentences_len,
                       capsnet.encoded_intents: batch_intents_one_hot, capsnet.encoded_slots: batch_slots_one_hot})

        writer.add_summary(cross_entropy_summary, epoch * test_batch + i)
        writer.add_summary(margin_loss_summary, epoch * test_batch + i)
        writer.add_summary(loss_summary, epoch * test_batch + i)

        intent_outputs_reduced_dim = tf.squeeze(intent_outputs)
        intent_outputs_norm = safe_norm(intent_outputs_reduced_dim)
        slot_weights_c_reduced_dim = tf.squeeze(slot_weights_c)

        [intent_predictions, slot_predictions] = sess.run([intent_outputs_norm, slot_weights_c_reduced_dim])

        te_batch_intent_pred = np.argmax(intent_predictions, axis=1)
        total_intent_pred += np.ndarray.tolist(te_batch_intent_pred)

        te_batch_slots_pred = np.argmax(slot_predictions, axis=2)
        total_slots_pred += (np.ndarray.tolist(te_batch_slots_pred))

    print("           VALIDATION SET PERFORMANCE        ")
    print("Intent detection")
    intents_acc = accuracy_score(y_intents_te, total_intent_pred)
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
    print("Slot filling")
    print('F1 score: %lf' % scores['f1'])
    print('Accuracy: %lf' % scores['accuracy'])
    # print('Precision: %lf' % scores['precision'])
    # print('Recall: %lf' % scores['recall'])

    return f_score, scores['f1']


def generate_batch(n, batch_size):
    batch_index = a.sample(range(n), batch_size)
    return batch_index


def assign_pretrained_word_embedding(sess, embedding, capsnet):
    print("using pre-trained word emebedding.begin...")
    word_embedding = tf.constant(embedding, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(capsnet.Embedding, word_embedding)  # assign this value to model
    sess.run(t_assign_embedding)
    print("using pre-trained word emebedding.ended...")


def main():
    # load data
    data = data_loader.read_datasets()
    x_tr = data['x_tr']
    y_intents_tr = data['y_intents_tr']
    y_slots_tr = data['y_slots_tr']
    sentences_length_tr = data['sentences_len_tr']

    one_hot_intents_tr = data['encoded_intents_tr']
    one_hot_slots_tr = data['encoded_slots_tr']

    embedding = data['embedding']

    # load settings
    FLAGS = setting(data)

    if not FLAGS.test:
        # k-fold cross-validation
        intent_scores = 0
        slot_scores = 0
        mean_scores = 0
        best_f_score = 0.0
        print("------------------start cross-validation-------------------")
        fold = 1
        for train_index, val_index in StratifiedKFold(FLAGS.n_splits).split(x_tr, y_intents_tr):
            print("FOLD %d" % fold)
            fold += 1
            x_train, x_val = x_tr[train_index], x_tr[val_index]
            y_intents_train, y_intents_val = y_intents_tr[train_index], y_intents_tr[val_index]
            y_slots_train, y_slots_val = y_slots_tr[train_index], y_slots_tr[val_index]
            sentences_length_train, sentences_length_val = sentences_length_tr[train_index], sentences_length_tr[
                val_index]
            one_hot_intents_train, one_hot_intents_val = one_hot_intents_tr[train_index], one_hot_intents_tr[val_index]
            one_hot_slots_train, one_hot_slots_val = one_hot_slots_tr[train_index], one_hot_slots_tr[val_index]

            val_data = dict()
            val_data['x_val'] = x_val
            val_data['y_intents_val'] = y_intents_val
            val_data['y_slots_val'] = y_slots_val
            val_data['sentences_len_val'] = sentences_length_val
            val_data['one_hot_intents_val'] = one_hot_intents_val
            val_data['one_hot_slots_val'] = one_hot_slots_val
            val_data['slots_dict'] = data['slots_dict']
            val_data['intents_dict'] = data['intents_dict']

            # start
            tf.reset_default_graph()
            config = tf.ConfigProto()
            with tf.Session(config=config) as sess:
                # Instantiate Model
                capsnet = model.capsnet(FLAGS)

                print('Initializing Variables')
                sess.run(tf.global_variables_initializer())
                if FLAGS.use_embedding:
                    # load pre-trained word embedding
                    assign_pretrained_word_embedding(sess, embedding, capsnet)

                intent_f_score, slot_f_score = evaluate_validation(capsnet, val_data, FLAGS, sess, epoch=0)
                f_score_mean = (intent_f_score + slot_f_score) / 2
                if f_score_mean > best_f_score:
                    best_f_score = f_score_mean
                var_saver = tf.train.Saver()

                train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

                # Training cycle
                train_sample_num = x_train.shape[0]
                batch_num = int(train_sample_num / FLAGS.batch_size)
                for epoch in range(FLAGS.num_epochs):
                    for batch in range(batch_num):
                        batch_index = generate_batch(train_sample_num, FLAGS.batch_size)
                        batch_x = x_train[batch_index]
                        batch_y_intents = y_intents_train[batch_index]
                        batch_y_slots = y_slots_train[batch_index]
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
                                                             capsnet.sentences_length: batch_sentences_len})

                        train_writer.add_summary(cross_entropy_summary, batch_num * epoch + batch)
                        train_writer.add_summary(margin_loss_summary, batch_num * epoch + batch)
                        train_writer.add_summary(loss_summary, batch_num * epoch + batch)

                    print("------------------epoch : ", epoch, " Loss: ", loss, "----------------------")
                    intent_f_score, slot_f_score = evaluate_validation(capsnet, val_data, FLAGS, sess, epoch=epoch + 1)
                    f_score_mean = (intent_f_score + slot_f_score) / 2
                    if f_score_mean > best_f_score:
                        # save model
                        best_f_score = f_score_mean
                        var_saver.save(sess, os.path.join(FLAGS.ckpt_dir, "model.ckpt"), 1)
                    print("Current F score mean", f_score_mean)
                    print("Best F score mean", best_f_score)
                    intent_scores += intent_f_score
                    slot_scores += slot_f_score
                    mean_scores += f_score_mean

        mean_intent_score = intent_scores / FLAGS.n_splits
        mean_slot_score = slot_scores / FLAGS.n_splits
        mean_score = mean_scores / FLAGS.n_splits
        print('Mean intent F1 score %lf' % mean_intent_score)
        print('Mean slot F1 score %lf' % mean_slot_score)
        print('Mean F1 score %lf' % mean_score)

    else:
        # testing
        test_data = dict()
        test_data['x_te'] = data['x_te']
        test_data['y_intents_te'] = data['y_intents_te']
        test_data['y_slots_te'] = data['y_slots_te']
        test_data['sentences_len_te'] = data['sentences_len_te']
        test_data['slots_dict'] = data['slots_dict']
        test_data['intents_dict'] = data['intents_dict']

        tf.reset_default_graph()
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            # Instantiate Model
            capsnet = model.capsnet(FLAGS)
            if os.path.exists(FLAGS.ckpt_dir):
                print("Restoring Variables from Checkpoint for testing")
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
                intent_f_score, slot_f_score = evaluate_test(capsnet, test_data, FLAGS, sess)
            else:
                print("No trained model exists in checkpoint dir!")


if __name__ == "__main__":
    main()
