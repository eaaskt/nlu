from data_loader import *
from tool import *

model_dir = './saved_models/'


def main():
    print("------------------read datasets begin-------------------")
    data = {}

    # load word2vec model
    print("------------------load word2vec begin-------------------")
    w2v = load_w2v(word2vec_path)
    print("------------------load word2vec end---------------------")

    # load normalized word embeddings
    embedding = w2v.vectors
    norm_embedding = norm_matrix(embedding)
    data['embedding'] = norm_embedding


if __name__ == "__main__":
    main()