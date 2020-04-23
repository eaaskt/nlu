import numpy as np
from numpy import dot
from numpy.linalg import norm

import tools

ORIGINAL_VECS = 'lang/vectors/ro_ft_300_allpos_datasets_ant.vec'

CF_VECS = 'lang/vectors/ro_ft_300_verbs_datasets_ant.vec'
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

    identical = True
    for k, v in new_vecs.items():
        try:
            old_values = original_vecs[k]
        except KeyError:
            print(f"{k} not foudn in original vecs")
            identical = False
            break
        if not np.array_equal(v, old_values):
            cos = cosine_sim(v, old_values)
            print('{} diff value -- cosine sim {}'.format(k, cos))
            identical = False
    print("Identical vector values" if identical else "Not identical vector values")


def main():
    compare_vecs(ORIGINAL_VECS, CF_VECS, VOCABULARY)


if __name__ == "__main__":
    main()