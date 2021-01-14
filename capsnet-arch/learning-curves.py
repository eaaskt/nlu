import train
import data_loader
import flags
import model_s2i


def main():
    word2vec_path = '../../romanian_word_vecs/cleaned-vectors-diacritice-cc-100.vec'

    training_data_path = '../data-capsnets/diacritics/scenario1/train.txt'
    test_data_path = '../data-capsnets/diacritics/scenario1/test.txt'

    # Define the flags
    FLAGS = flags.define_app_flags('1-tensorboard-40-examples')

    # Load data
    print('------------------load word2vec begin-------------------')
    w2v = data_loader.load_w2v(word2vec_path)
    print('------------------load word2vec end---------------------')
    data = data_loader.read_datasets(w2v, training_data_path, test_data_path)

    flags.set_data_flags(data)

    val_data, train_data = data_loader.extract_validation(data, nr_splits=3)

    for i in range(1, 13):
        print('----- ITER {} -----'.format(i))
        stepData = data_loader.data_subset(train_data, 12, i)
        train.train_cross_validation(model_s2i.CapsNetS2I, stepData, val_data, data['embedding'],  FLAGS, fold=i*5,
                                     best_f_score=0, calculate_learning_curves=True)


if __name__ == '__main__':
    main()
