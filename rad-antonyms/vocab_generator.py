import configparser
import sys

import tools


class SettingConfig:
    def __init__(self, config_path):
        # Read the config file
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        self.dataset_root_diac_path = self.config.get("paths", "DATASET_DIAC_PATH")
        self.dataset_root_nodiac_path = self.config.get("paths", "DATASET_NODIAC_PATH")

        self.vocab_diac_path = self.config.get("paths", "VOCAB_PATH_DATASET_DIAC")
        self.vocab_nodiac_path = self.config.get("paths", "VOCAB_PATH_DATASET_NODIAC")


def compute_vocabulary_diac(config: SettingConfig):
    vcb = tools.compute_vocabulary(config.dataset_root_diac_path)
    tools.save_vocabulary(vcb, config.vocab_diac_path)


def compute_vocabulary_nodiac(config: SettingConfig):
    vcb = tools.compute_vocabulary(config.dataset_root_nodiac_path)
    tools.save_vocabulary(vcb, config.vocab_nodiac_path)


def main():
    try:
        config_filepath = sys.argv[1]
    except IndexError:
        print("\nUsing the default config file: parameters.cfg")
        config_filepath = "parameters.cfg"
    config = SettingConfig(config_filepath)
    compute_vocabulary_diac(config)
    compute_vocabulary_nodiac(config)


if __name__ == "__main__":
    main()
