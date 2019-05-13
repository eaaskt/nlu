import spacy

nlp = spacy.load('ro')
tokens = nlp('stinge aprinde inchide porneste')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
print()

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
