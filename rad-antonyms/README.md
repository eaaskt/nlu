# Generating Antonym/Synonym Pairs from RWN, Counterfitting, RASA Pipeline.

## Relations Generator - Parameters
Relations generator runs can be parametrized from the ./parameters.cfg file.
The paths section includes path relative to the project root where the input / output files of the run are stored.

The settings section includes what to be taken into account when generating pairs:
The 'POS' parameter indicates pairs of what parts of speech should be included in the output pairs. As RWN is limited to 4 parts of speech, this parameter will be a subset of [noun, verb, adjective, adverb]. For example, if we set this parameter to [noun, verb], only nouns and verbs will be included in the generated pairs.

## Counterfitting - Parameters

Counterfitting runs can be parametrized from the ./parameters.cfg file. 

The paths section includes paths relative to the project root where the input / output files of the run are stored. 
As input to a counterfit run, paths to the pairs of synonyms / antonyms are provided, along with paths to the vectors and the vocabulary.
As output, a path to where the counterfit vectors are to be stored is provided (along with a path where the different vectors are stored [WIP]).

The settings sections include what to be taken into account for the counterfit run:
The 'MODE' parameter indicates which of the 3 terms to be included in the cost function: AR, SA, VSP. Explicitly, it is a subset of [ant, syn, vsp], with each term being associated to its correspondent in the cost function. For instance, if MODE = [ant], the cost function will only contain the AR term, therfore only antonym pairs will be enhanced.
The 'VOCABULARY' parameter indicates which vocabulary to use for the counterfit run: The whole vocabulary, or a smaller vocabulary, composed of words from our datasets +/-. Its values can be either 'all', for the whole vocabulary,  or 'small', for the dataset vocabulary. Based on this, the script chooses from the two vocabulary paths provided above.

The hyperparameters section includes hyperparameters for the counterfit run, as described in the paper.
Hyper_k1, hyper_k2, hyper_3 are the weights of the AR, SA, respectively VSP terms in the cost function.
Delta - the intuitive minimum distance between antonymous words.
Gamma - the ideal maximum distance between synonymous words.
Rho - the maximum difference between two words for them to be considered a VSP pair.
Sgd_iters - number of iterations of stochastic gradient descent performed for minimising the cost function.