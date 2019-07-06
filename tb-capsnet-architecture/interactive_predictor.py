import tensorflow as tf
import data_loader
import model
import main_module
import os
import numpy as np
model_dir = './saved_models/Scenario0'


def evaluate_sample(capsnet, data, sess):
    x_te = data['sample_utterance']

    slots_dict = data['slots_dict']

    [intent_outputs, slots_outputs, slot_weights_c] = sess.run([
        capsnet.intent_output_vectors, capsnet.slot_output_vectors, capsnet.slot_weights_c],
        feed_dict={capsnet.input_x: [x_te], capsnet.sentences_length: [len(x_te)],
                   capsnet.keep_prob: 1.0})

    intent_outputs_reduced_dim = tf.squeeze(intent_outputs)
    intent_outputs_norm = main_module.safe_norm(intent_outputs_reduced_dim)
    sliced_slot_weights_c = tf.slice(slot_weights_c, begin=[0, 0, 0, 0, 0],
                                     size=[-1, capsnet.max_sentence_length, -1, -1, -1])
    slot_weights_c_reduced_dim = tf.squeeze(sliced_slot_weights_c)

    [intent_predictions, slot_predictions] = sess.run([intent_outputs_norm, slot_weights_c_reduced_dim])

    y_intent_label_pred = np.argmax(intent_predictions, axis=0)

    te_slots_pred = np.argmax(slot_predictions, axis=1)
    total_slots_pred = (np.ndarray.tolist(te_slots_pred))

    y_slot_labels_pred = [slots_dict[slot_idx] for slot_idx in total_slots_pred]

    print("Utterance classified as: " + data['intents_dict'][y_intent_label_pred])
    print("Slots:\n" + str(y_slot_labels_pred))

    return y_slot_labels_pred, y_intent_label_pred

def main():
    data = data_loader.read_datasets()

    vocab_size, word_emb_size = data['embedding'].shape
    _, max_sentence_length = data['x_tr'].shape
    intents_number = len(data['intents_dict'])
    slots_number = len(data['slots_dict'])

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_float("keep_prob", 0.8, "embedding dropout keep rate")
    tf.app.flags.DEFINE_integer("hidden_size", 32, "embedding vector size")
    tf.app.flags.DEFINE_integer("batch_size", 1, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("num_epochs", 20, "num of epochs")
    tf.app.flags.DEFINE_integer("vocab_size", vocab_size, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("max_sentence_length", max_sentence_length, "max number of words in one sentence")
    tf.app.flags.DEFINE_integer("intents_nr", intents_number, "intents_number")  #
    tf.app.flags.DEFINE_integer("slots_nr", slots_number, "slots_number")  #
    tf.app.flags.DEFINE_integer("word_emb_size", word_emb_size, "embedding size of word vectors")
    tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
    tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    tf.app.flags.DEFINE_float("margin", 1.0, "ranking loss margin")
    tf.app.flags.DEFINE_integer("slot_routing_num", 4, "slot routing num")
    tf.app.flags.DEFINE_integer("intent_routing_num", 4, "intent routing num")
    tf.app.flags.DEFINE_integer("re_routing_num", 3, "re routing num")
    tf.app.flags.DEFINE_integer("intent_output_dim", 64, "intent output dimension")
    tf.app.flags.DEFINE_integer("slot_output_dim", 128, "slot output dimension")
    tf.app.flags.DEFINE_integer("attention_output_dimenison", 20, "self attention weight hidden units number")
    tf.app.flags.DEFINE_float("alpha", 0.001, "coefficient for self attention loss")
    tf.app.flags.DEFINE_integer("r", 3, "self attention weight hops")
    tf.app.flags.DEFINE_boolean("save_model", False, "save model to disk")
    tf.app.flags.DEFINE_boolean("test", False, "Evaluate model on test data")
    tf.app.flags.DEFINE_boolean("crossval", False, "Perform k-fold cross validation")
    tf.app.flags.DEFINE_integer("n_splits", 3, "Number of cross-validation splits")
    tf.app.flags.DEFINE_string("summaries_dir", './logs', "tensorboard summaries")
    tf.app.flags.DEFINE_string("ckpt_dir", './saved_models/Scenario0', "check point dir")
    tf.app.flags.DEFINE_string("scenario_num", '', "Scenario number")

    input_sentence = 'fa mai mica intensitatea iluminatului in sufragerie la 4'
    print(input_sentence)

    print(input_sentence)
    print("ba")
    sentence_words = input_sentence.split(' ')
    encoded_words = []
    for w in sentence_words:
        encoded_words.append(data_loader.w2v_dict['w2v'].vocab[w].index)

    print(encoded_words)

    max_len = data['max_len']
    print(len(encoded_words))
    encoded_padded_words = np.append(encoded_words, np.zeros((max_len - len(encoded_words),), dtype=np.int64))

    data['sample_utterance'] = encoded_padded_words

    tf.reset_default_graph()
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        capsnet = model.capsnet(FLAGS)
        if os.path.exists(FLAGS.ckpt_dir):
            print("Restoring Variables from Checkpoint for testing")
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            slots, intents = evaluate_sample(capsnet, data, sess)
            print("Finished inferring")
        else:
            print("No trained model exists in checkpoint dir!")

if __name__ == "__main__":
    main()