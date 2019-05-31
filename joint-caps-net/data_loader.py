""" input data preprocess.
"""

import numpy as np
import tool
from random import Random
from datetime import datetime
from gensim.models.keyedvectors import KeyedVectors

word2vec_path = 'data/word-vec/wiki.ro.vec'
training_data_path = 'data/scenario0/train.txt'
test_data_path = 'data/scenario0/train.txt'


def load_w2v(file_name):
    """ load w2v model
        input: model file name
        output: w2v model
    """
    w2v = KeyedVectors.load_word2vec_format(
            file_name, binary=False)
    return w2v


def load_vec(file_path, w2v, in_max_len):
    """ load input data
        input:
            file_path: input data file
            w2v: word2vec model
            in_max_len: max length of sentence
        output:
            input_x: input sentence word ids
            input_y: input label ids
            input_y_s: input slot ids
            s_len: input sentence length
            max_len: max length of sentence
            intent_dict: intents dictionary
            slot_dict: slots dictionary
    """
    input_x = [] # input sentence word ids
    input_y = [] # input label ids
    input_y_s = [] # input sentence slots
    sentences_length = [] # input sentence length
    max_len = 0

    intent_dict = {}
    intent_id_dict = {}
    intent_id = 0
    slot_dict = {}
    slot_id_dict = {}
    slot_id = 0

    for line in open(file_path):
        arr = line.strip().split('\t')
        intent = arr[0]
        slots = arr[1].split(' ')
        text = arr[2].split(' ')

        if intent not in intent_dict:
            intent_dict[intent] = intent_id
            intent_id_dict[intent_id] = intent
            intent_id += 1

        if len(slots) != len(text):
            continue

        # trans words into indexes
        x_vectors = []
        y_slots = []
        for w, s in zip(text, slots):

            if w in w2v.vocab:
                if s not in slot_dict:
                    slot_dict[s] = slot_id
                    slot_id_dict[slot_id] = s
                    slot_id += 1
                x_vectors.append(w2v.vocab[w].index)
                y_slots.append(slot_dict[s])

        sentence_length = len(x_vectors)
        if sentence_length <= 1:
            continue
        if in_max_len == 0:
            if sentence_length > max_len:
                max_len = sentence_length

        input_x.append(np.asarray(x_vectors))
        input_y.append(intent_dict[intent])
        input_y_s.append(np.asarray(y_slots))
        sentences_length.append(sentence_length)

    # add paddings
    max_len = max(in_max_len, max_len)
    x_padding = []
    y_s_padding = []
    for i in range(len(input_x)):
        if max_len < sentences_length[i]:
            x_padding.append(input_x[i][0:max_len])
            sentences_length[i] = max_len
            y_s_padding.append(input_y_s[i][0:max_len])
            continue
        tmp = np.append(input_x[i], np.zeros((max_len - sentences_length[i],), dtype=np.int64))
        x_padding.append(tmp)
        tmp = np.append(input_y_s[i], np.zeros((max_len - sentences_length[i],), dtype=np.int64))
        y_s_padding.append(tmp)

    x_padding = np.asarray(x_padding)
    input_y = np.asarray(input_y)
    input_y_s = np.asarray(y_s_padding)
    sentences_length = np.asarray(sentences_length)
    return x_padding, input_y, input_y_s, sentences_length, max_len, intent_id_dict, slot_id_dict


def get_label(data):
    y_intents_tr = data['y_intents_tr']
    y_slots_tr = data['y_slots_tr']
    y_intents_val = data['y_intents_val']
    y_slots_val = data['y_slots_val']
    max_len = data['max_len']
    sample_num_tr = y_intents_tr.shape[0]
    sample_num_val = y_intents_val.shape[0]
    nr_intents = len(data['intents_dict'])
    nr_slots = len(data['slots_dict'])
    intents_id = range(nr_intents)
    slots_id = range(nr_slots)

    # get label index
    ind_intents_tr = np.zeros((sample_num_tr, nr_intents), dtype=np.float32)
    ind_slots_tr = np.zeros((sample_num_tr, max_len, nr_slots), dtype=np.float32)
    for i in range(nr_intents):
        ind_intents_tr[y_intents_tr == intents_id[i], i] = 1
    for i in range(sample_num_tr):
        for j in range(nr_slots):
            ind_slots_tr[i, y_slots_tr[i] == slots_id[j], j] = 1

    ind_intents_val = np.zeros((sample_num_val, nr_intents), dtype=np.float32)
    ind_slots_val = np.zeros((sample_num_val, max_len, nr_slots), dtype=np.float32)
    for i in range(nr_intents):
        ind_intents_val[y_intents_val == intents_id[i], i] = 1
    for i in range(sample_num_val):
        for j in range(nr_slots):
            ind_slots_val[i, y_slots_val[i] == slots_id[j], j] = 1
    return ind_intents_tr, ind_slots_tr, ind_intents_val, ind_slots_val


def read_datasets():
    print("------------------read datasets begin-------------------")
    data = {}

    # load word2vec model
    print("------------------load word2vec begin-------------------")
    w2v = load_w2v(word2vec_path)
    print("------------------load word2vec end---------------------")

    # load normalized word embeddings
    embedding = w2v.vectors
    data['embedding'] = embedding
    norm_embedding = tool.norm_matrix(embedding)
    data['embedding'] = norm_embedding

    # trans data into embedding vectors
    max_len = 0
    x_tr, y_intents_tr, y_slots_tr, sentences_length_tr, max_len, intents_dict, slots_dict = load_vec(training_data_path, w2v, max_len)
    x_te, y_intents_te, y_slots_te, sentences_length_te, max_len, test_intents_dict, test_slots_dict = load_vec(test_data_path, w2v, max_len)

    # hold data for validation
    gen = Random()
    gen.seed(datetime.now())
    data_length = len(x_tr)
    validation_index = gen.sample(range(data_length), int(0.2 * data_length))
    tr_index = [i for i in range(data_length) if i not in validation_index]

    x_val = x_tr[validation_index]
    y_intents_val = y_intents_tr[validation_index]
    y_slots_val = y_slots_tr[validation_index]
    sentences_length_val = sentences_length_tr[validation_index]

    x_train = x_tr[tr_index]
    y_intents_train = y_intents_tr[tr_index]
    y_slots_train = y_slots_tr[tr_index]
    sentences_length_train = sentences_length_tr[tr_index]

    data['x_tr'] = x_train
    data['y_intents_tr'] = y_intents_train
    data['y_slots_tr'] = y_slots_train
    data['sentences_len_tr'] = sentences_length_train

    data['x_val'] = x_val
    data['y_intents_val'] = y_intents_val
    data['y_slots_val'] = y_slots_val
    data['sentences_len_val'] = sentences_length_val

    data['intents_dict'] = intents_dict
    data['slots_dict'] = slots_dict

    data['x_te'] = x_te
    data['y_intents_te'] = y_intents_te
    data['y_slots_te'] = y_slots_te

    data['sentences_len_te'] = sentences_length_te

    data['max_len'] = max_len

    one_hot_y_intents_tr, one_hot_y_slots_tr, one_hot_y_intents_val, one_hot_y_slots_val = get_label(data)
    data['encoded_intents_tr'] = one_hot_y_intents_tr
    data['encoded_slots_tr'] = one_hot_y_slots_tr
    data['encoded_intents_val'] = one_hot_y_intents_val
    data['encoded_slots_val'] = one_hot_y_slots_val
    print("------------------read datasets end---------------------")
    return data

