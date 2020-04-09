import rowordnet as rwn
import io
import os
import gensim.models as gs
import re
from operator import itemgetter
from datetime import datetime

# region Paths

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
OUTPUT_PATH = ROOT_PATH + "\\out"

RAW_ANTONYM_PAIRS_PATH = OUTPUT_PATH + "\\rown_antonyms.txt"
PROCESSED_ANTONYM_PAIRS_PATH = OUTPUT_PATH + "\\rown_antonyms_processed.txt"

RAW_SYNONYM_PAIRS_PATH = OUTPUT_PATH + "\\rown_synonyms.txt"
PROCESSED_SYNONYMS_PAIRS_PATH = OUTPUT_PATH + "\\rown_synonyms_processed.txt"

W2V_PATH = ROOT_PATH + "\\cc.ro.300.vec"
W2V_SMALL_PATH = ROOT_PATH + "\\small_cc.ro.300.vec"


# endregion


def process_line(raw_line: str) -> str:
    # Remove reflexive forms, etc., basically anything that's in between [ ] or | |, infintive form "a ..."
    # re.sub("\[(.*?)\]", "", raw_line)
    # re.sub("|(.*?)|", "", raw_line)
    # Can't get the regex working, so just use the sad solution with multiple replace calls:
    to_replace = ["[se]", "|se|", "[-și]", "[o]", "|-și|", "|și|", "[-i]", "[i]", "[și]", "a "]
    for sub in to_replace:
        raw_line = raw_line.replace(sub, "")

    # Replace multiple spaces, strip beginning / ending spaces
    processed_line = re.sub('\s{2,}', ' ', raw_line).strip()

    words = processed_line.split(' ')

    # Return the pair as a string "word1 word2"
    # Or the empty string if the line contains only one word or the same word twice
    return "" if len(words) < 2 or words[1] in words[0] or words[0] in words[1] else " ".join(
        words)


def preprocess_pairs(input_path: str, output_path: str) -> None:
    print(f"Started preprocessing pairs @ {datetime.now()}")

    print(f"Started loading vocabulary model @ {datetime.now()}")
    model = gs.KeyedVectors.load_word2vec_format(W2V_PATH, binary=False, encoding='utf-8')
    print(f"Successfully loaded model @ {datetime.now()}")

    # Open the input file, which contains the raw pairs
    with io.open(file=input_path, mode="r", encoding="utf-8") as input_file:
        # Open the output file, which will contain the preprocessed pairs
        with io.open(file=output_path, mode="w", encoding="utf-8") as output_file:
            # Wipe the output file
            output_file.truncate(0)

            # Get the list of lines in the input file
            raw_lines = input_file.readlines()

            for raw_line in raw_lines:
                # Preprocess each line
                processed_line = process_line(raw_line)

                # If the processed line is not empty (meaning we have 2 different words separated by a space)
                if processed_line:
                    print(f"Preprocessed line result: {processed_line}")
                    # Split the words
                    w1, w2 = processed_line.split(" ")

                    # Check if both are in the dictionary

                    if w1 in model.wv.vocab and w2 in model.wv.vocab:
                        output_file.write(f"{w1} {w2}\n")
        output_file.close()
    input_file.close()
    print(f"Successfully finished preprocessing pairs")


def write_pairs_to_file(pairs: set, output_path: str) -> None:
    print(f"Writing pairs to file @ {datetime.now()}")
    with io.open(file=output_path, mode="w", encoding='utf-8') as out:
        # Wipe the file
        out.truncate(0)

        # Write each pair to the file
        for pair in pairs:
            out.write(f"{pair[0]} {pair[1]}\n")

        # Close the output file
        out.close()

    print(f"Successfully written pairs to file @ {datetime.now()}")


def generate_antonym_pairs() -> set:
    print(f"Generating initial antonym pairs from RoWordNet @ {datetime.now()}")
    wn = rwn.RoWordNet()

    # Create the output set of antonym pairs
    pairs = set()

    # Take each of the 4 RWN PoS' : [NOUN, VERB, ADJECTIVE, ADVERB]
    for part_of_speech in rwn.Synset.Pos:

        # Return all synsets corresponding to the PoS
        synset_ids = wn.synsets(pos=part_of_speech)

        # Iterate all the synsets for the current PoS
        for synset_id in synset_ids:

            # Get the synset object specified by synset_id
            synset = wn.synset(synset_id)

            # Get the outbound relations of type antonym from
            outbound_relations = filter(lambda x: x[1] == 'near_antonym', wn.outbound_relations(synset_id))

            # Get the literals
            current_literals = synset.literals

            # Iterate outbound relations
            for relation in outbound_relations:

                # Get the synset corresponding to the target of the outbound relation
                target_synset_id = relation[0]
                target_synset = wn.synset(target_synset_id)

                # Get the literals in the synset above
                target_literals = target_synset.literals

                # Get all the pairs, sort them by first word to keep set entries unique
                current_iteration_pairs = set(
                    [tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in current_literals for w2 in target_literals])

                # Add the current set of pairs
                for pair in current_iteration_pairs:
                    pairs.add(pair)

    # Return the whole set containing antonym pairs
    print(f"Successfully generated antonym paris @ {datetime.now()}")
    return pairs


def generate_synonym_pairs() -> set:
    print(f"Generating initial synonym pairs from RoWordNet @ {datetime.now()}")
    wn = rwn.RoWordNet()

    # Create the output set of antonym pairs
    pairs = set()

    # Take each of the 4 RWN PoS' : [NOUN, VERB, ADJECTIVE, ADVERB]
    for part_of_speech in rwn.Synset.Pos:

        # Return all synsets corresponding to the PoS
        synset_ids = wn.synsets(pos=part_of_speech)

        # Iterate all the synsets for the current PoS
        for synset_id in synset_ids:

            # Get the synset object specified by synset_id
            synset = wn.synset(synset_id)

            literals = synset.literals

            # Get all the pairs, sort them by first word to keep set entries unique
            current_iteration_pairs = set(
                [tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in literals for w2 in literals if not w1 == w2])

            # Append all pairs from the current PoS to the global set
            for pair in current_iteration_pairs:
                pairs.add(pair)

    print(f"Successfully generated synonym pairs {datetime.now()}")
    return pairs


def antonyms_pipeline() -> None:
    antonym_pairs = generate_antonym_pairs()
    write_pairs_to_file(antonym_pairs, RAW_ANTONYM_PAIRS_PATH)
    preprocess_pairs(RAW_ANTONYM_PAIRS_PATH, PROCESSED_ANTONYM_PAIRS_PATH)


def synonyms_pipeline() -> None:
    synonym_pairs = generate_synonym_pairs()
    write_pairs_to_file(synonym_pairs, RAW_SYNONYM_PAIRS_PATH)
    preprocess_pairs(RAW_SYNONYM_PAIRS_PATH, PROCESSED_SYNONYMS_PAIRS_PATH)


if __name__ == '__main__':
    # antonyms_pipeline()
    synonyms_pipeline()
