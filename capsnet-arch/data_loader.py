import numpy as np
import util
from sklearn.model_selection import StratifiedKFold

from gensim.models.keyedvectors import KeyedVectors
import time


def load_w2v(file_name):
    """ Load Word2Vec model
        Args:
            file_name: model file name
        Returns:
            w2v: w2v model
    """
    start = time.time()
    # w2v = FastText.load_fasttext_format(file_name)
    w2v = KeyedVectors.load_word2vec_format(file_name, binary=False)
    end = time.time()
    print("loading time took %06.2f" % (end - start))
    return w2v


def load_vec(file_path, w2v, in_max_len, intent_dict, intent_id, slot_dict, slot_id, load_text=False, lowercase=False):
    """ Load input data in w2v index format
        Args:
            file_path: path to input data file
            w2v: word2vec model
            in_max_len: max length of sentence so far (used if loading test data)
            intent_dict: mapping of intent class id to intent class label (used if loading test data)
            intent_id: last used intent class id
            slot_dict: mapping of slot class id to slot class label (used if loading test data)
            slot_id: last used slot class id
            load_text: True if the original text should be loaded, False otherwise
            lowercase: True if all words should be preprocessed to lowercase

        Returns:
            x_padding: input sentence word ids (with padding if necessary
            input_y: input intent ids
            input_y_s: input slot ids
            sentences_len: input sentences length
            max_len: max length of sentence
            intent_dict: intents dictionary
            intent_id: last used intent id
            slot_dict: slots dictionary
            slot_id: last used slot id
            x_text_padding: input sentences (in raw text format)
    """
    input_x = []
    input_y = []
    input_y_s = []
    sentences_length = []
    max_len = 0

    input_x_text = []

    for line in open(file_path, encoding='utf-8'):
        arr = line.strip().split('\t')
        intent = arr[0]
        slots = arr[1].split(' ')
        sentence = arr[2]
        if lowercase:
            sentence = sentence.lower()
        text = sentence.split(' ')

        if intent not in intent_dict:
            intent_dict[intent] = intent_id
            intent_id += 1

        if len(slots) != len(text):
            continue

        # trans words into indexes
        x_vectors = []
        y_slots = []
        x_text = []
        for w, s in zip(text, slots):

            if w in w2v.vocab:
                if s not in slot_dict:
                    slot_dict[s] = slot_id
                    slot_id += 1
                x_vectors.append(w2v.vocab[w].index)
                y_slots.append(slot_dict[s])

                if load_text:
                    x_text.append(w)
            else:
                print("Word {} not in W2V vocabulary!".format(w))

        sentence_length = len(x_vectors)
        # todo look into this 1 limit
        if sentence_length <= 1:
            continue
        # todo why is it like this?
        # if in_max_len == 0:
        if sentence_length > max_len:
            max_len = sentence_length

        input_x.append(np.asarray(x_vectors))
        input_y.append(intent_dict[intent])
        input_y_s.append(np.asarray(y_slots))
        sentences_length.append(sentence_length)

        if load_text:
            input_x_text.append(np.asarray(x_text))

    max_len = max(in_max_len, max_len)
    return input_x, input_y, input_y_s, input_x_text, sentences_length, max_len, intent_dict, intent_id, slot_dict, slot_id


def addPadding(input_x, input_y_s, input_y, input_x_text, sentences_length, max_len, load_text=False):
    # add paddings
    x_padding = []
    y_s_padding = []
    x_text_padding = []
    for i in range(len(input_x)):
        tmp = np.append(input_x[i], np.zeros((max_len - sentences_length[i],), dtype=np.int64))
        x_padding.append(tmp)
        tmp = np.append(input_y_s[i], np.zeros((max_len - sentences_length[i],), dtype=np.int64))
        y_s_padding.append(tmp)
        if load_text:
            tmp = np.append(input_x_text[i], np.zeros((max_len - sentences_length[i],), dtype=np.int64))
            x_text_padding.append(tmp)

    x_padding = np.asarray(x_padding)
    input_y = np.asarray(input_y)
    input_y_s = np.asarray(y_s_padding)
    sentences_length = np.asarray(sentences_length)
    if load_text:
        x_text_padding = np.asarray(x_text_padding)
    else:
        x_text_padding = None

    return x_padding, input_y, input_y_s, x_text_padding, sentences_length


def get_label(data, test=False):
    """ Encodes the intent and slot labels in one-hot format
        Args:
            data: data dictionary
            test: True if should encode test labels
        Returns:
            ind_intents: one-hot encoded intents (of shape [nr_samples, nr_intents])
            ind_slots: one-hot encoded slots (of shape [nr_samples, max_len, nr_slots])
    """
    if test:
        y_intents = data['y_intents_te']
        y_slots = data['y_slots_te']
    else:
        y_intents = data['y_intents_tr']
        y_slots = data['y_slots_tr']
    max_len = data['max_len']
    print("max length is", max_len)
    sample_num_tr = y_intents.shape[0]
    nr_intents = len(data['intents_dict'])
    nr_slots = len(data['slots_dict'])
    intents_id = range(nr_intents)
    slots_id = range(nr_slots)

    # get label index
    ind_intents = np.zeros((sample_num_tr, nr_intents), dtype=np.float32)
    ind_slots = np.zeros((sample_num_tr, max_len, nr_slots), dtype=np.float32)
    for i in range(nr_intents):
        ind_intents[y_intents == intents_id[i], i] = 1
    for i in range(sample_num_tr):
        for j in range(nr_slots):
            ind_slots[i, y_slots[i] == slots_id[j], j] = 1
    return ind_intents, ind_slots


def read_datasets(w2v, training_data_path, test_data_path, test=False, lowercase=False):
    """ Reads the data from the given input files
        Args:
            w2v: Word2Vec model
            training_data_path: path to training data file
            test_data_path: path to test data file
            test: True if running test on the model -- this will load the test data in raw text format

        Returns:
            data: data dictionary
    """
    print('------------------read datasets begin-------------------')
    data = {}

    # TODO: figure out if normalizing these is something we want to do
    # load normalized word embeddings
    embedding = w2v.vectors
    norm_embedding = util.norm_matrix(embedding)
    data['embedding'] = norm_embedding

    # trans data into embedding vectors
    max_len = 0
    slots_dict = dict()
    intents_dict = dict()
    slot_id = 0
    intent_id = 0

    # First load the sentences vecs and find max len from train + test
    (input_x_tr, input_y_tr, input_y_s_tr, input_x_text_tr, sentences_length_tr,
     max_len, intents_dict, intent_id, slots_dict, slot_id) = load_vec(training_data_path, w2v, max_len, intents_dict,
                                                                       intent_id, slots_dict, slot_id, load_text=False,
                                                                       lowercase=lowercase)
    (input_x_te, input_y_te, input_y_s_te, input_x_text_te, sentences_length_te,
     max_len, intents_dict, intent_id, slots_dict, slot_id) = load_vec(test_data_path, w2v, max_len, intents_dict,
                                                                       intent_id, slots_dict, slot_id, load_text=test,
                                                                       lowercase=lowercase)

    # Add padding to the sentences to make vectorization easier
    (x_tr, y_intents_tr, y_slots_tr, x_text_tr, sentences_length_tr) = addPadding(input_x_tr, input_y_s_tr, input_y_tr,
                                                                                  input_x_text_tr, sentences_length_tr,
                                                                                  max_len, load_text=False)
    (x_te, y_intents_te, y_slots_te, x_text_te, sentences_length_te) = addPadding(input_x_te, input_y_s_te, input_y_te,
                                                                                  input_x_text_te, sentences_length_te,
                                                                                  max_len, load_text=test)

    intents_id_dict = {v: k for k, v in intents_dict.items()}
    slots_id_dict = {v: k for k, v in slots_dict.items()}

    data['x_tr'] = x_tr
    data['y_intents_tr'] = y_intents_tr
    data['y_slots_tr'] = y_slots_tr
    data['sentences_len_tr'] = sentences_length_tr

    data['intents_dict'] = intents_id_dict
    data['slots_dict'] = slots_id_dict

    data['x_te'] = x_te
    data['y_intents_te'] = y_intents_te
    data['y_slots_te'] = y_slots_te
    data['sentences_len_te'] = sentences_length_te

    if test:
        data['x_text_te'] = x_text_te

    data['max_len'] = max_len

    one_hot_y_intents_tr, one_hot_y_slots_tr = get_label(data)
    data['encoded_intents_tr'] = one_hot_y_intents_tr
    data['encoded_slots_tr'] = one_hot_y_slots_tr

    one_hot_y_intents_te, one_hot_y_slots_te = get_label(data, test=True)
    data['encoded_intents_te'] = one_hot_y_intents_te
    data['encoded_slots_te'] = one_hot_y_slots_te
    print('------------------read datasets end---------------------')
    return data


def extract_validation(data, nr_splits):
    x_tr = data['x_tr']
    y_intents_tr = data['y_intents_tr']
    y_slots_tr = data['y_slots_tr']
    sentences_length_tr = data['sentences_len_tr']

    one_hot_intents_tr = data['encoded_intents_tr']
    one_hot_slots_tr = data['encoded_slots_tr']
    folding_result = list(StratifiedKFold(nr_splits).split(x_tr, y_intents_tr))

    train_index = folding_result[0][0]
    val_index = folding_result[0][1]
    x_train, x_val = x_tr[train_index], x_tr[val_index]
    y_intents_train, y_intents_val = y_intents_tr[train_index], y_intents_tr[val_index]
    y_slots_train, y_slots_val = y_slots_tr[train_index], y_slots_tr[val_index]
    sentences_length_train, sentences_length_val = sentences_length_tr[train_index], sentences_length_tr[
        val_index]
    one_hot_intents_train, one_hot_intents_val = one_hot_intents_tr[train_index], one_hot_intents_tr[
        val_index]
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

    train_data = dict()
    train_data['x_tr'] = x_train
    train_data['y_intents_tr'] = y_intents_train
    train_data['y_slots_tr'] = y_slots_train
    train_data['sentences_len_tr'] = sentences_length_train
    train_data['one_hot_intents_tr'] = one_hot_intents_train
    train_data['one_hot_slots_tr'] = one_hot_slots_train
    train_data['encoded_intents_tr'] = one_hot_intents_train
    train_data['encoded_slots_tr'] =  one_hot_slots_train

    return val_data, train_data


def data_subset(data, splitsNumber, index):
    x_tr = data['x_tr']
    y_intents_tr = data['y_intents_tr']
    y_slots_tr = data['y_slots_tr']
    sentences_length_tr = data['sentences_len_tr']

    one_hot_intents_tr = data['encoded_intents_tr']
    one_hot_slots_tr = data['encoded_slots_tr']
    folding_result = list(StratifiedKFold(splitsNumber).split(x_tr, y_intents_tr))

    train_index = np.array([], dtype=int)

    for i in range(0, index):
        train_index = np.concatenate((train_index, folding_result[i][1]))


    x_train = x_tr[train_index]
    y_intents_train = y_intents_tr[train_index]
    y_slots_train = y_slots_tr[train_index]
    sentences_length_train = sentences_length_tr[train_index]
    one_hot_intents_train = one_hot_intents_tr[train_index]
    one_hot_slots_train = one_hot_slots_tr[train_index]

    train_data = dict()
    train_data['x_tr'] = x_train
    train_data['y_intents_tr'] = y_intents_train
    train_data['y_slots_tr'] = y_slots_train
    train_data['sentences_len_tr'] = sentences_length_train
    train_data['one_hot_intents_tr'] = one_hot_intents_train
    train_data['one_hot_slots_tr'] = one_hot_slots_train

    return train_data
