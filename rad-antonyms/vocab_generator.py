import tools


def compute_vocabulary_diac():
    vcb = tools.compute_vocabulary("datasets/fara_diacritice")
    tools.save_vocabulary(vcb, "lang/vocab_small_nodiac.txt")


def compute_vocabulary_nodiac():
    vcb = tools.compute_vocabulary("datasets/fara_diacritice")
    tools.save_vocabulary(vcb, "lang/vocab_small_nodiac.txt")


if __name__ == "__main__":
    compute_vocabulary_diac()
    compute_vocabulary_nodiac()
