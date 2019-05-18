import spacy

nlp = spacy.load('ro')
tokens = nlp('stinge aprinde inchide porneste opreste mareste creste redu micsoreaza seteaza modifica')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
print()

for i in range(0, len(tokens)):
    for j in range(i, len(tokens)):
        token1 = tokens[i]
        token2 = tokens[j]
        print(token1.text, token2.text, token1.similarity(token2))
