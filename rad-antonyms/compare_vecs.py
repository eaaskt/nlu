import numpy as np
from numpy import dot
from numpy.linalg import norm
import tools

ORIGINAL_VECS = '../../romanian_word_vecs/cc.ro.300.vec'

CF_VECS = 'counterfitting-results/FastText-300-datasets-synAntVSP-dia.vec'
VOCABULARY = 'lang/vocab_small_diac.txt'


def cosine_sim(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def compare_vecs(original_vecs_path, cf_vecs_path, vocab_path):
    vocab = set()
    with open(file=vocab_path, mode="r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            vocab.add(line.strip())

    # Load the word vectors
    original_dim, original_vecs = tools.load_vectors(original_vecs_path, vocab)

    # Load the new vectors
    new_dim, new_vecs = tools.load_vectors(cf_vecs_path, vocab)

    for k, v in new_vecs.items():
        old_values = original_vecs[k]
        if not np.array_equal(v, old_values):
            cos = cosine_sim(v, old_values)
            print('{} diff value -- cosine sim {}'.format(k, cos))


def main():
    compare_vecs(ORIGINAL_VECS, CF_VECS, VOCABULARY)


if __name__ == "__main__":
    main()