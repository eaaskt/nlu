# Natural Language Understanding

[Wit.ai HTTP API docs](https://wit.ai/docs/http/20170307)

[Rasa NLU documentation](https://rasa.com/docs/nlu/)

[SpaCy training language models](https://spacy.io/usage/training)


#Loading faster the fasttext models

There are 2 types of files you can download from fasttext 

**`.vec`** - file, imported like:
```
from gensim.models.keyedvectors import KeyedVectors

w2v = KeyedVectors.load_word2vec_format(file_name, binary=False)
```

**`.bin`** - file, imported like:
```
from gensim.models import FastText

w2v = FastText.load_fasttext_format(file_name)
w2v.wv # use it like this
```

The latter method loads the word vector model in ~ 130 - 150 seconds on my machine while the first method loads the vectors in around 10 mins.

 


