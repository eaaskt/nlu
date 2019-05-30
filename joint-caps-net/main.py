import os
from random import *
import data_loader
import numpy as np
import tensorflow as tf
import model
import tool
import math
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score

a = Random()
a.seed(1)


def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_sentence_length = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    intents_number = len(data['intents_dict'])
    slots_number = len(data['slots_dict'])

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_float("keep_prob", 0.8, "embedding dropout keep rate")
    tf.app.flags.DEFINE_integer("hidden_size", 128, "embedding vector size")
    tf.app.flags.DEFINE_integer("batch_size", 64, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("num_epochs", 100, "num of epochs")
    tf.app.flags.DEFINE_integer("vocab_size", vocab_size, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("max_sentence_length", max_sentence_length, "max number of words in one sentence")
    tf.app.flags.DEFINE_integer("sample_num", sample_num, "sample number of training data")
    tf.app.flags.DEFINE_integer("test_num", test_num, "number of test data")
    tf.app.flags.DEFINE_integer("intents_nr", intents_number, "intents_number") #
    tf.app.flags.DEFINE_integer("slots_nr", slots_number, "slots_number") #
    tf.app.flags.DEFINE_integer("word_emb_size", word_emb_size, "embedding size of word vectors")
    tf.app.flags.DEFINE_string("ckpt_dir", './saved_models/' , "check point dir")
    tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
    tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    tf.app.flags.DEFINE_float("margin", 1.0, "ranking loss margin")
    tf.app.flags.DEFINE_integer("slot_routing_num", 2, "slot routing num")
    tf.app.flags.DEFINE_integer("intent_routing_num", 2, "intent routing num")
    tf.app.flags.DEFINE_integer("re_routing_num", 2, "re routing num")
    tf.app.flags.DEFINE_integer("intent_output_dim", 32, "intent output dimension")
    tf.app.flags.DEFINE_integer("slot_output_dim", 256, "slot output dimension")
    tf.app.flags.DEFINE_boolean("save_model", False, "save model to disk")

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


def evaluate_test(data, FLAGS, sess):
    x_te = data['x_te']
    sentences_length_te = data['sentences_len_te']
    y_intents_te = data['y_intents_te']
    y_slots_te = data['y_slots_te']
    slots_dict = data['slots_dict']

    total_intent_pred = []
    total_slots_pred = []

    batch_size = FLAGS.batch_size
    test_batch = int(math.ceil(FLAGS.test_num / float(batch_size)))
    for i in range(test_batch):
        begin_index = i * batch_size
        end_index = min((i + 1) * batch_size, FLAGS.test_num)
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
    acc = accuracy_score(y_intents_te, total_intent_pred)
    print(classification_report(y_intents_te, total_intent_pred, digits=4))

    y_slots_te_true = np.ndarray.tolist(y_slots_te)
    y_slot_labels_true = [[slots_dict[slot_idx] for slot_idx in ex] for ex in y_slots_te_true]
    y_slot_labels_pred = [[slots_dict[slot_idx] for slot_idx in ex] for ex in total_slots_pred]
    scores = eval_seq_scores(y_slot_labels_true, y_slot_labels_pred)
    print("Slot filling")
    print('F1 score: %lf' % scores['f1'])
    print('Accuracy: %lf' % scores['accuracy'])
    print('Precision: %lf' % scores['precision'])
    print('Recall: %lf' % scores['recall'])
    return acc


def generate_batch(n, batch_size):
    batch_index = a.sample(range(n), batch_size)
    return batch_index


def assign_pretrained_word_embedding(sess, data, textRNN):
    print("using pre-trained word emebedding.begin...")
    embedding = data['embedding']

    word_embedding = tf.constant(embedding, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding, word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    # load data
    data = data_loader.read_datasets()
    x_tr = data['x_tr']
    y_intents_tr = data['y_intents_tr']
    y_slots_tr = data['y_slots_tr']

    one_hot_y_intents_tr = data['encoded_intents']
    one_hot_y_slots_tr = data['encoded_slots']

    sentences_length_tr = data['sentences_len_tr']
    embedding = data['embedding']

    # load settings
    FLAGS = setting(data)

    # start
    tf.reset_default_graph()
    config=tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        capsnet = model.capsnet(FLAGS)

        if os.path.exists(FLAGS.ckpt_dir):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                # load pre-trained word embedding
                assign_pretrained_word_embedding(sess, data, capsnet)

        best_acc = 0.0
        cur_acc = evaluate_test(data, FLAGS, sess)
        if cur_acc > best_acc:
            best_acc = cur_acc
        var_saver = tf.train.Saver()

        # Training cycle
        batch_num = int(FLAGS.sample_num / FLAGS.batch_size)
        for epoch in range(FLAGS.num_epochs):
            for batch in range(batch_num):
                batch_index = generate_batch(FLAGS.sample_num, FLAGS.batch_size)
                batch_x = x_tr[batch_index]
                batch_y_intents = y_intents_tr[batch_index]
                batch_y_slots = y_slots_tr[batch_index]
                batch_sentences_len = sentences_length_tr[batch_index]
                batch_intents_one_hot = one_hot_y_intents_tr[batch_index]
                batch_slots_one_hot = one_hot_y_slots_tr[batch_index]

                [_, loss, intent_pred, slot_pred] = sess.run([capsnet.train_op, capsnet.loss_val,
                                                              capsnet.intent_output_vectors,
                                                              capsnet.slot_output_vectors],
                                                             feed_dict={capsnet.input_x: batch_x,
                                                                        capsnet.encoded_intents: batch_intents_one_hot,
                                                                        capsnet.encoded_slots: batch_slots_one_hot,
                                                                        capsnet.sentences_length: batch_sentences_len})

            print("------------------epoch : ", epoch, " Loss: ", loss, "----------------------")
            cur_acc = evaluate_test(data, FLAGS, sess)
            if cur_acc > best_acc:
                # save model
                best_acc = cur_acc
                var_saver.save(sess, os.path.join(FLAGS.ckpt_dir, "model.ckpt"), 1)
            print("cur_acc", cur_acc)
            print("best_acc", best_acc)
