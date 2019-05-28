import spacy

nlp = spacy.load('ro')
tokens = nlp('aprinde stinge scade creste seteaza pune ridica schimba opreste porneste '
             'inchide reduce diminueaza mica mare mareste ajusteaza modifica micsoreaza incepe incet tare treci')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
print()

for i in range(0, len(tokens)):
    for j in range(i, len(tokens)):
        token1 = tokens[i]
        token2 = tokens[j]
        print(token1.text, token2.text, token1.similarity(token2))


tokens2 = nlp('melodia simfonia lumina temperatura TV muzica televizor')
for i in range(0, len(tokens2)):
    for j in range(i, len(tokens2)):
        token1 = tokens2[i]
        token2 = tokens2[j]
        print(token1.text, token2.text, token1.similarity(token2))
