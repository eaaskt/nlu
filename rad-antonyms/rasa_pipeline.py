import configparser
import copy
import io
import json
import os
import sys

import gspread
import subprocess

from shutil import copyfile
from shutil import rmtree
from statistics import stdev, mean
from typing import Optional

import yaml
from gspread import Worksheet
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.utils import resample
from zipfile import ZipFile

from tools import copy_path

SPREADSHEET_START_VERTICAL_OFFSET = 3


class SettingConfig:

    def __init__(self, config_path):

        # Read the config file
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        # Load Hyperparameters
        self.splits = self.config.getint("hyperparameters", "splits")
        self.sample_percent = self.config.getfloat("hyperparameters", "sample_percent")

        # Load datasets path
        self.diacritics = self.config.get("settings", "DIACRITICS")
        if self.diacritics == "True":
            self.datasets_path = self.config.get("paths", "DATASET_DIAC_PATH")
        else:
            self.datasets_path = self.config.get("paths", "DATASET_NODIAC_PATH")

        # Load script and config paths
        self.rasa_script_path = self.config.get("paths", "SCRIPT_PATH")
        self.rasa_config_path = self.config.get("paths", "RASA_CONFIG_PATH")

        # Load report paths
        self.intent_report_path = self.config.get("paths", "INTENT_REPORT_PATH")
        self.intent_errors_path = self.config.get("paths", "INTENT_ERRORS_PATH")
        self.slot_report_path = self.config.get("paths", "SLOT_REPORT_PATH")
        self.slot_errors_path = self.config.get("paths", "SLOT_ERRORS_PATH")
        self.conf_mat_path = self.config.get("paths", "CONFUSION_MATRIX_PATH")

        # Load merged reports path
        self.merged_reports_root = self.config.get("paths", "MERGED_REPORTS_ROOT")
        self.merged_intent_report_path = self.config.get("paths", "MERGED_INTENT_REPORT_PATH")
        self.merged_intent_errors_path = self.config.get("paths", "MERGED_INTENT_ERRORS_PATH")
        self.merged_slot_report_path = self.config.get("paths", "MERGED_SLOT_REPORT_PATH")
        self.merged_slot_errors_path = self.config.get("paths", "MERGED_SLOT_ERRORS_PATH")
        self.merged_matrices_path = self.config.get("paths", "MERGED_MATRICES_PATH")

        with io.open(self.rasa_config_path, "r") as rasa_config_file:
            rasa_config = yaml.load(rasa_config_file, Loader=yaml.FullLoader)
            self.language = rasa_config['language']

        self.identifier = f"{self.language}_{'diac' if self.diacritics else 'nodiac'}"


def create_single_split(base_path: str, split_identifier: int, sample_percent: float, train: bool) -> None:
    path = base_path + ("_train.json" if train else "_test.json")
    with io.open(path, "r", encoding="utf-8") as file:
        # Load the JSON content
        content = json.load(file)
        content_copy = copy.deepcopy(content)

        # Obtain the wrapped object
        data = content_copy["rasa_nlu_data"]

        # Obtain the list of sentences
        common_examples = data["common_examples"]

        sample_count = int(sample_percent * len(common_examples))
        resampled_examples = resample(common_examples, n_samples=sample_count)
        data["common_examples"] = resampled_examples

        # Create the new file
        write_path = base_path + ("_train_" if train else "_test_") + f"{split_identifier}" + ".json"
        file = io.open(write_path, "w", encoding="utf-8")

        # Dump JSON content into the file
        json.dump(content_copy, file, ensure_ascii=False)
        file.close()


def create_splits(config: SettingConfig) -> None:
    # Iterate every scenario folder - Name has to conform "scenario_$nr"
    for file in os.listdir(config.datasets_path):

        # Compute the path of to the scenario folder
        scenario_folder_path = os.path.join(config.datasets_path, file)

        # Only take into account scenario directories
        if os.path.isdir(scenario_folder_path):

            # Ensure the folder contains the scenarios
            if len(os.listdir(scenario_folder_path)) == 0:
                print('Empty Scenario Directory. Exiting.')
                break

            # Compute the file path - File path has to start with "scenario_$nr"
            file_path = os.path.join(scenario_folder_path, file)

            # For each split, create the dataset split and save it
            for split_id in range(config.splits):
                # Create splits for train / test datasets
                create_single_split(file_path, split_id, train=True, sample_percent=config.sample_percent)
                create_single_split(file_path, split_id, train=False, sample_percent=config.sample_percent)
    print("Finished creating fresh dataset splits")


def wipe_reports(config: SettingConfig) -> None:
    # Wipe the merged intent and slot errors and reports
    merged_reports_paths = [config.merged_intent_report_path, config.merged_intent_errors_path,
                            config.merged_slot_report_path, config.merged_slot_errors_path]
    for file in merged_reports_paths:
        io.open(file, "w", encoding="utf-8").close()

    # Wipe the current errors and reports
    reports_paths = [config.intent_report_path, config.intent_errors_path, config.slot_report_path,
                     config.slot_errors_path]
    for file in reports_paths:
        io.open(file, "w", encoding="utf-8").close()

    # Delete all report directories
    for scenario_folder in os.listdir(config.datasets_path):
        scenario_path = os.path.join(config.datasets_path, scenario_folder)
        if os.path.isdir(scenario_path):
            for file in os.listdir(scenario_path):
                file_path = os.path.join(scenario_path, file)
                if os.path.isdir(file_path):
                    rmtree(file_path)


def wipe_splits(config: SettingConfig) -> None:
    # Iterate every scenario folder - Name has to conform "scenario_$nr"
    for file in os.listdir(config.datasets_path):

        # Compute the path of to the scenario folder
        scenario_folder_path = os.path.join(config.datasets_path, file)

        # Only take into account scenario directories
        if os.path.isdir(scenario_folder_path):

            # Do not proceed if the scenario does only contain the train and test file
            if len(os.listdir(scenario_folder_path)) <= 2:
                print('Directory contains at most train and test splits. Breaking.')
                return

            # Compute the file path - File path has to start with "scenario_$nr"
            file_path = os.path.join(scenario_folder_path, file)

            # Compute the file name for the splits, remove them
            for split_id in range(config.splits):
                train_split = file_path + "_train_" + f"{split_id}" + ".json"
                test_split = file_path + "_test_" + f"{split_id}" + ".json"
                os.remove(train_split)
                os.remove(test_split)
        print("Finished wiping the splits")


def process_intent_result(identifier: str, scenario_report_path: str, config: SettingConfig) -> float:
    #  Move intent report result to the merged report
    with io.open(config.intent_report_path, "r", encoding="utf-8") as report_file:

        # Open the merged intent report file for writing
        with io.open(config.merged_intent_report_path, "a", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)
            weighted_average_f1 = content['weighted avg']['f1-score']

            # Write the weighted average for intent detection in the file
            output_file.write(
                "F1-Score - ID for "
                + identifier + f":{weighted_average_f1}\n")

            output_file.close()
        report_file.close()

    # Move intent errors to the merged errors
    with io.open(config.intent_errors_path, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(config.merged_intent_errors_path, "a", encoding="utf-8") as output_file:
            # Load the content
            if os.path.getsize(config.intent_errors_path) != 0:
                content = json.load(report_file)
                if content:
                    output_file.write(f"\nIntent Errors report for {identifier}")
                    content = sorted(content, key=lambda k: (k['intent_prediction']['name'], k['intent']))
                    # Copy all the intent errors in a human-readable form to the merged errors report
                    for entry in content:
                        output_file.write(
                            f"\n\t Predicted: {entry['intent_prediction']['name']}. Actual: {entry['intent']}. "
                            f"Text: {entry['text']}. "
                            f"Conf: {entry['intent_prediction']['confidence']}".replace('\"', ""))
                    output_file.close()
            report_file.close()

    # Also move the whole reports for backup
    report_identifier_path = os.path.join(scenario_report_path, "intent_reports", "report_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")
    errors_identifier_path = os.path.join(scenario_report_path, "intent_reports", "errors_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")
    # Open (create) the file
    io.open(report_identifier_path, "w+", encoding="utf-8")
    io.open(errors_identifier_path, "w+", encoding="utf-8")

    # Save the content from the current report to the intent report
    copy_path(config.intent_report_path, report_identifier_path)
    copy_path(config.intent_errors_path, errors_identifier_path)

    return weighted_average_f1


def process_slot_result(identifier: str, scenario_report_path: str, config: SettingConfig) -> float:
    # Move slot report result to the merged report
    with io.open(config.slot_report_path, "r", encoding="utf-8") as report_file:
        # Open the merged slot report file for writing
        with io.open(config.merged_slot_report_path, "a", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)

            # Write the weighted average for intent detection in the file
            weighted_average_f1 = content['weighted avg']['f1-score']
            output_file.write(
                "F1-Score - SF for "
                + identifier + f":{weighted_average_f1}\n")

            output_file.close()
        report_file.close()

    # Move slot errors to the merged errors
    with io.open(config.slot_errors_path, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(config.merged_slot_errors_path, "a", encoding="utf-8") as output_file:
            # Load the content
            if os.path.getsize(config.slot_errors_path) != 0:
                content = json.load(report_file)
                if content:
                    content = sorted(content, key=lambda k: k['entities'][0]['entity'] if k['entities'] else "")
                    # Copy all the slot errors in a human-readable form to the merged errors report
                    output_file.write(f"\nErrors report for {identifier}")
                    for entry in content:
                        output_file.write(
                            f"\n\t    Predicted: {[(e['entity'], e['value']) for e in entry['predicted_entities']]}. "
                            f"Actual: {[(e['entity'], e['value']) for e in entry['entities']]}. "
                            f"Text: {entry['text']}".replace("[", "").replace(")]", "").replace("(", "").replace("',",
                                                                                                                 ": ").replace(
                                "),",
                                ",").replace(
                                '  ', ' ').replace('\'', "").replace("].", " - ")
                        )
            output_file.close()
        report_file.close()

    # Compute the path for the current intent report and errors
    report_identifier_path = os.path.join(scenario_report_path, "slot_reports", "report_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")
    errors_identifier_path = os.path.join(scenario_report_path, "slot_reports", "errors_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")

    # Open (create) the file
    io.open(report_identifier_path, "w+", encoding="utf-8")
    io.open(errors_identifier_path, "w+", encoding="utf-8")

    # Save the content from the current report to the intent report
    copy_path(config.slot_report_path, report_identifier_path)
    copy_path(config.slot_errors_path, errors_identifier_path)
    return weighted_average_f1


def copy_confusion_matrix(identifier: str, config: SettingConfig) -> None:
    copyfile(config.conf_mat_path,
             os.path.join(config.merged_matrices_path,
                          identifier.replace(" ", "").replace(",", "").replace("_", "") + ".png"))


def process_datasets(config: SettingConfig, sheet: Worksheet) -> None:
    print("Started running RASA")

    # Compute and write the title of the spreadsheet based on the loaded configurations
    spreadsheet_title = [config.identifier]
    sheet.insert_row(spreadsheet_title, 1)

    # For each scenario folder
    spreadsheet_row_index = SPREADSHEET_START_VERTICAL_OFFSET
    for file in os.listdir(config.datasets_path):

        # Compute the path for the scenario folder
        folder_path = os.path.join(config.datasets_path, file)
        if os.path.isdir(folder_path):

            # Break if the directory does not contain the splits
            if len(os.listdir(folder_path)) <= 2:
                print("Directory only contains train and test files, but no splits. Breaking.")
                break

            # Compute the scenario file name
            file_path = os.path.join(folder_path, file)

            # Compute the reports path and create the directory
            scenario_reports_path = os.path.join(folder_path, 'scenario_reports')
            os.mkdir(scenario_reports_path)

            # Compute the intent and slot reports paths and create them
            intent_reports_path = os.path.join(scenario_reports_path, 'intent_reports')
            slot_reports_path = os.path.join(scenario_reports_path, 'slot_reports')

            os.mkdir(intent_reports_path)
            os.mkdir(slot_reports_path)

            scenario_slot_results = [f'Slot - {file}']
            scenario_intent_results = [f'Intent - {file}']

            for split_id in range(config.splits):
                # Compute the identifier, get the train split and test split
                identifier = f" {file}, split {split_id}"
                train_split = file_path + "_train_" + f"{split_id}" + ".json"
                test_split = file_path + "_test_" + f"{split_id}" + ".json"

                # Run the subprocess for RASA training and testing, and wait for its completion
                command = [config.rasa_script_path, train_split, test_split, config.rasa_config_path]
                subprocess.Popen(command, shell=True).wait()

                # Process the slot and intent errors & reports and save their return values
                intent_f1 = process_intent_result(identifier, scenario_reports_path, config)
                slot_f1 = process_slot_result(identifier, scenario_reports_path, config)

                # Move the confusion matrix to the results path
                copy_confusion_matrix(identifier, config)

                scenario_slot_results.append(float("{:0.4f}".format(slot_f1)))
                scenario_intent_results.append(float("{:0.4f}".format(intent_f1)))

                print(f"Finished processing split {identifier}")

            # Append the mean value to each list for the scenario
            scenario_intent_results.append(float("{:0.4f}".format(mean(scenario_intent_results[1:]))))
            scenario_slot_results.append(float("{:0.4f}".format(mean(scenario_slot_results[1:]))))

            # Append the standard deviation to each list for the scenario
            scenario_intent_results.append(
                float("{:0.3f}".format(stdev(scenario_intent_results[1:len(scenario_intent_results) - 2]))))
            scenario_slot_results.append(
                float("{:0.3f}".format(stdev(scenario_slot_results[1:len(scenario_slot_results) - 2]))))

            # Append the line in the google doc:
            sheet.insert_row(scenario_slot_results, spreadsheet_row_index)
            sheet.insert_row(scenario_intent_results, spreadsheet_row_index)
            spreadsheet_row_index += 3


def get_worksheet(name: str) -> Optional[Worksheet]:
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('thesis_secret.json', scope)
    client = gspread.authorize(credentials)
    return client.open(name).sheet1


def create_analysis_archive(config: SettingConfig):
    with ZipFile(f'{config.identifier}.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(config.merged_reports_root):
            for filename in filenames:
                # create complete filepath of file in directory
                filepath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filepath)
    zipObj.close()


def rasa_pipeline(config_path: str) -> None:
    config = SettingConfig(config_path)
    wipe_reports(config)
    wipe_splits(config)
    create_splits(config)
    worksheet = get_worksheet('Benchmark Counterfitting')
    process_datasets(config, worksheet)
    create_analysis_archive(config)


def main():
    try:
        config_filepath = sys.argv[1]
    except IndexError:
        print("\nUsing the default config file: parameters.cfg")
        config_filepath = "parameters.cfg"
    rasa_pipeline(config_filepath)


if __name__ == "__main__":
    main()
