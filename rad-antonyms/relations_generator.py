import configparser
import io
import re
import sys
from datetime import datetime
from operator import itemgetter
from typing import Optional

import gensim.models as gs
import rowordnet as rwn


class SettingConfig:

    def __init__(self, config_path):

        # Read the config file
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        vec_path = self.config.get("paths", "VEC_PATH")

        pos = self.config.get("variables", "POS").replace("[", "").replace("]", "").replace(" ", "").split(",")

        mapping = {"verb": rwn.Synset.Pos.VERB,
                   "noun": rwn.Synset.Pos.NOUN,
                   "adverb": rwn.Synset.Pos.ADVERB,
                   "adjective": rwn.Synset.Pos.ADJECTIVE}

        # Value for each key if key in pos
        self.pos = [mapping.get(m) for m in mapping.keys() if m in pos]

        print(f"Initializing pair generator for the following parts of speech: {self.pos} ")

        self.raw_antonyms_path = self.config.get("paths", "RAW_ANTONYMS_PATH")
        self.raw_synonyms_path = self.config.get("paths", "RAW_SYNONYMS_PATH")

        self.synonyms_path = self.config.get("paths", "SYNONYMS_PATH")
        self.antonyms_path = self.config.get("paths", "ANTONYMS_PATH")

        print(f"Started loading vocabulary model @ {datetime.now()}")
        self.model = gs.KeyedVectors.load_word2vec_format(vec_path, binary=False, encoding='utf-8')
        print(f"Successfully loaded model @ {datetime.now()}")


def process_line(raw_line: str) -> Optional[str]:
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
    # Or the empty string if the words are the same or contain eachother
    # return "" if len(words) != 2 or words[1] in words[0] or words[0] in words[1] else " ".join(words)
    if len(words) != 2:
        return None
    if words[1] in words[0] or words[0] in words[1]:
        return None
    if words[1][0].isupper() or words[0][0].isupper():
        return None
    return " ".join(words)


def preprocess_pairs(input_path: str, output_path: str, config: SettingConfig) -> None:
    print(f"Started preprocessing pairs @ {datetime.now()}")
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
                    if w1 in config.model.wv.vocab and w2 in config.model.wv.vocab:
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


def generate_antonym_pairs(pos: list) -> set:
    print(f"Generating initial antonym pairs from RoWordNet @ {datetime.now()}")
    wn = rwn.RoWordNet()

    # Create the output set of antonym pairs
    pairs = set()

    # Take each of the 4 RWN PoS' : [NOUN, VERB, ADJECTIVE, ADVERB]
    for part_of_speech in pos:

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


def generate_synonym_pairs(pos: list) -> set:
    print(f"Generating initial synonym pairs from RoWordNet @ {datetime.now()}")
    wn = rwn.RoWordNet()

    # Create the output set of antonym pairs
    pairs = set()

    # Take each of the 4 RWN PoS' : [NOUN, VERB, ADJECTIVE, ADVERB]
    for part_of_speech in pos:

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


def antonyms_pipeline(config: SettingConfig) -> None:
    antonym_pairs = generate_antonym_pairs(config.pos)
    write_pairs_to_file(antonym_pairs, config.raw_antonyms_path)
    preprocess_pairs(config.raw_antonyms_path, config.antonyms_path, config)


def synonyms_pipeline(config: SettingConfig) -> None:
    synonym_pairs = generate_synonym_pairs(config.pos)
    write_pairs_to_file(synonym_pairs, config.raw_synonyms_path)
    preprocess_pairs(config.raw_synonyms_path, config.synonyms_path, config)


def main():
    try:
        config_filepath = sys.argv[1]
    except IndexError:
        print("\nUsing the default config file: parameteres.cfg")
        config_filepath = "parameters.cfg"
    config = SettingConfig(config_filepath)
    synonyms_pipeline(config)
    antonyms_pipeline(config)


if __name__ == '__main__':
    main()
