# Generating Antonym/Synonym Pairs from RWN, Counterfitting, RASA Pipeline.

## Vocabulary Generator
The vocab_generator.py script generates vocabularies from the folder containing datasets, which is specified as a parameter.

## Relations Generator - Parameters
The relations_generator.py script generates synonym and antonym pairs from RoWordNet and stores them, according to their part-of-speech. The subfolders corresponding to each of the four parts of speech, verb, adverb, noun, adjective, will be generated under the path specified by CONSTRAINTS_ROOT_PATH in parameters.cfg.
Once generated, the pairs can be used in the next components of the pipeline.

## Counterfitting - Parameters
Counterfitting runs can be parametrized from the ./parameters.cfg file. 
Before running counterfitting, please add the original vectors file in lang/vectors and set its correct path in ./parameters.cfg

The paths section includes paths relative to the project root where the input / output files of the run are stored.
As input, paths contani 
As output, a path to where the counterfit vectors are to be stored is provided.

The settings sections include what to be taken into account for the counterfit run:
The 'MODE' parameter indicates which of the 3 terms to be included in the cost function: AR, SA, VSP. Explicitly, it is a subset of [ant, syn, vsp], with each term being associated to its correspondent in the cost function. For instance, if MODE = [ant], the cost function will only contain the AR term, therfore only antonym pairs will be enhanced.
The 'VOCABULARY' parameter indicates which vocabulary to use for the counterfit run: The whole vocabulary, or a smaller vocabulary, composed of words from our datasets +/-. Its values can be either 'all', for the whole vocabulary,  or 'small', for the dataset vocabulary. Based on this, the script chooses from the two vocabulary paths provided above.
The 'DIACRITICS' parameter sets whether the vocabulary generated based on the datasets with diacritics or the ones without should be used. Values can be either True or False

The hyperparameters section includes hyperparameters for the counterfit run, as described in the paper.
Hyper_k1, hyper_k2, hyper_3 are the weights of the AR, SA, respectively VSP terms in the cost function.
Delta - the intuitive minimum distance between antonymous words.
Gamma - the ideal maximum distance between synonymous words.
Rho - the maximum difference between two words for them to be considered a VSP pair.
Sgd_iters - number of iterations of stochastic gradient descent performed for minimising the cost function.