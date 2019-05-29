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

a = Random();
a.seed(1)

def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_sentence_length = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    intents_number = len(data['intents_dict'])
    slots_number = len(data['slots_dict'])

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_float("keep_prob", 0.8, "embedding dropout keep rate")
    tf.app.flags.DEFINE_integer("hidden_size", 32, "embedding vector size")
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
    tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
    tf.app.flags.DEFINE_float("margin", 1.0, "ranking loss margin")
    tf.app.flags.DEFINE_integer("slot_routing_num", 2, "slot routing num")
    tf.app.flags.DEFINE_integer("intent_routing_num", 2, "intent routing num")
    tf.app.flags.DEFINE_integer("re_routing_num", 2, "re routing num")
    tf.app.flags.DEFINE_integer("intent_output_dim", 8, "intent output dimension")
    tf.app.flags.DEFINE_integer("slot_output_dim", 10, "slot output dimension")
    tf.app.flags.DEFINE_boolean("save_model", False, "save model to disk")

    return FLAGS

def get_sim(data):
    # get unseen and seen categories similarity
    s = normalize(data['sc_vec'])
    u = normalize(data['uc_vec'])
    sim = tool.compute_label_sim(u, s, FLAGS.sim_scale)
    return sim

def evaluate_test(data, FLAGS, sess):
    # zero-shot testing state
    # seen votes shape (110, 2, 34, 10)
    x_te = data['x_te']
    y_te_id = data['y_te']
    u_len = data['u_len']

    # get unseen and seen categories similarity
    # sim shape (8, 34)
    sim_ori = get_sim(data)
    total_unseen_pred = np.array([], dtype=np.int64)

    batch_size  = FLAGS.test_num
    test_batch = int(math.ceil(FLAGS.test_num / float(batch_size)))
    #test_batch = int(math.ceil(FLAGS.test_num / float(FLAGS.batch_size)))
    for i in range(test_batch):
        begin_index = i * batch_size
        end_index = min((i + 1) * batch_size, FLAGS.test_num)
        batch_te = x_te[begin_index : end_index]
        batch_id = y_te_id[begin_index : end_index]
        batch_len = u_len[begin_index : end_index]

        [attentions, seen_logits, seen_votes, seen_weights_c] = sess.run([
            lstm.attention, lstm.logits, lstm.votes, lstm.weights_c],
            feed_dict={lstm.input_x: batch_te, lstm.s_len: batch_len})

        sim = tf.expand_dims(sim_ori, [0])
        sim = tf.tile(sim, [seen_votes.shape[1],1,1])
        sim = tf.expand_dims(sim, [0])
        sim = tf.tile(sim, [seen_votes.shape[0],1,1,1])
        seen_weights_c = np.tile(np.expand_dims(seen_weights_c, -1), [1, 1, 1, FLAGS.output_atoms])
        mul = np.multiply(seen_votes, seen_weights_c)

        # compute unseen features
        # unseen votes shape (110, 2, 8, 10)
        unseen_votes = tf.matmul(sim, mul)

        # routing unseen classes
        u_activations, u_weights_c = update_unseen_routing(unseen_votes, FLAGS, 3)
        unseen_logits = tf.norm(u_activations, axis=-1)
        te_votes, te_logits, te_weights, te_activations = sess.run([
            unseen_votes, unseen_logits, u_weights_c, u_activations])

        te_batch_pred = np.argmax(te_logits, 1)
        total_unseen_pred = np.concatenate((total_unseen_pred, te_batch_pred))

    print("           zero-shot intent detection test set performance        ")
    acc = accuracy_score(y_te_id, total_unseen_pred)
    print(classification_report(y_te_id, total_unseen_pred, digits=4))
    return acc

def generate_batch(n, batch_size):
    batch_index = a.sample(range(n), batch_size)
    return batch_index

def assign_pretrained_word_embedding(sess, data, textRNN):
    print("using pre-trained word emebedding.begin...")
    embedding = data['embedding']

    word_embedding = tf.constant(embedding, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("using pre-trained word emebedding.ended...")

def squash(input_tensor):
    norm = tf.norm(input_tensor, axis=2, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

def update_unseen_routing(votes, FLAGS, num_routing=3):
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    votes_trans = tf.transpose(votes, votes_t_shape)
    num_dims = 4
    input_dim = FLAGS.r
    output_dim = FLAGS.u_cnum
    input_shape = tf.shape(votes)
    logit_shape = tf.stack([input_shape[0], input_dim, output_dim])

    def _body(i, logits, activations, route):
        route = tf.nn.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1)
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations, route)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    route = tf.nn.softmax(logits, dim=2)
    _, logits, activations, route = tf.while_loop(
        lambda i, logits, activations, route: i < num_routing,
        _body,
        loop_vars=[i, logits, activations, route],
        swap_memory=True)

    return activations.read(num_routing - 1), route

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

    x_te = data['x_te']
    sentences_length_te = data['sentences_len_te']
    y_intents_te = data['y_intents_te']
    y_slots_te = data['y_slots_te']

    # load settings
    FLAGS = setting(data)

    # start
    tf.reset_default_graph()
    config=tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        lstm = model.capsnet(FLAGS)

        if os.path.exists(FLAGS.ckpt_dir):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, data, lstm)

        best_acc = 0.0
        cur_acc = evaluate_test(data, FLAGS, sess)
        if cur_acc > best_acc:
            best_acc = cur_acc
        var_saver = tf.train.Saver()

        # Training cycle
        batch_num = FLAGS.sample_num / FLAGS.batch_size
        for epoch in range(FLAGS.num_epochs):
            for batch in range(batch_num):
                batch_index = generate_batch(FLAGS.sample_num, FLAGS.batch_size)
                batch_x = x_tr[batch_index]
                batch_y_id = y_tr_id[batch_index]
                batch_len = s_len[batch_index]
                batch_ind = y_ind[batch_index]

                [_, loss, logits] = sess.run([lstm.train_op, lstm.loss_val, lstm.logits],
                        feed_dict={lstm.input_x: batch_x, lstm.IND: batch_ind, lstm.s_len: batch_len})

            print("------------------epoch : ", epoch, " Loss: ", loss, "----------------------")
            cur_acc = evaluate_test(data, FLAGS, sess)
            if cur_acc > best_acc:
                # save model
                best_acc = cur_acc
                var_saver.save(sess, os.path.join(FLAGS.ckpt_dir, "model.ckpt"), 1)
            print("cur_acc", cur_acc)
            print("best_acc", best_acc)
