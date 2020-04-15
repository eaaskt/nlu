import copy
import io
import json
import os
import subprocess
import warnings
from shutil import copyfile
from shutil import rmtree
from statistics import stdev, mean

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.utils import resample


warnings.simplefilter(action='ignore', category=FutureWarning)

# region Dictionary

# region Constants
SPLITS_NUMBER = 5

PROJECT_ROOT_PATH = 'C:\\Uni\\Thesis\\radandreicristian\\nlu\\rad-antonyms'
DATASET_ROOT_PATH = PROJECT_ROOT_PATH + '\\datasets\\fara_diacritice'
SCRIPT_PATH = PROJECT_ROOT_PATH + "\\" + "verify.bat"
RESULTS_PATH = PROJECT_ROOT_PATH + "\\" + "results"
CONFIG_PATH = PROJECT_ROOT_PATH + "\\config.yml"

# report files paths
INTENT_REPORT_PATH = RESULTS_PATH + "\\intent_report.json"
INTENT_ERRORS_PATH = RESULTS_PATH + "\\intent_errors.json"

SLOT_REPORT_PATH = RESULTS_PATH + "\\CRFEntityExtractor_report.json"
SLOT_ERRORS_PATH = RESULTS_PATH + "\\CRFEntityExtractor_errors.json"

CONFUSION_MATRIX_PATH = RESULTS_PATH + "\\confmat.png"

# output files paths
MERGED_INTENT_REPORT_PATH = RESULTS_PATH + "\\merged_reports\\" + "intent_report_merged.txt"
MERGED_INTENT_ERRORS_PATH = RESULTS_PATH + "\\merged_reports\\" + "intent_errors_merged.txt"

MERGED_SLOT_REPORT_PATH = RESULTS_PATH + "\\merged_reports\\" + "slot_report_merged.txt"
MERGED_SLOT_ERRORS_PATH = RESULTS_PATH + "\\merged_reports\\" + "slot_errors_merged.txt"

MERGED_MATRICES_PATH = RESULTS_PATH + "\\merged_reports\\confusion_matrices"

SAMPLE_PERCENT = 0.9

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('thesis_secret.json', scope)
client = gspread.authorize(credentials)

SPREADSHEET_START_VERTICAL_OFFSET = 3
spreadsheet = client.open('Benchmark Split Datasets').sheet1


# endregion


# region Helper functions

def strip_replace(src, strip_sequence):
    multi_replace(src, strip_sequence, "")


def multi_replace(src, to_replace, replacement):
    for char in to_replace:
        src.replace(char, replacement)
    return src


def copy_path(src_path, dst_path, append=True):
    with io.open(src_path, "r", encoding="utf-8") as src:
        with io.open(dst_path, "a" if append else "w", encoding="utf-8") as dst:
            dst.writelines([l for l in src.readlines()])


def copy_file(src_file, dst_file):
    dst_file.writelines([l for l in src_file.readlines()])


def assert_correct_file_size(path, new_path):
    path_json = json.load(io.open(path, "r", encoding='utf8'))
    new_path_json = json.load(io.open(new_path, "r", encoding='utf8'))
    assert (int(0.9 * len(path_json['rasa_nlu_data']['common_examples'])) == len(
        new_path_json['rasa_nlu_data']['common_examples']))


# endregion


def create_single_split(base_path: str, split_identifier: int, sample_percent: float, train=True) -> None:
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

        # Make sure the new file contains the correct number of entries
        # assert_correct_file_size(path, write_path)


def create_splits(base_path=DATASET_ROOT_PATH) -> None:
    # Iterate every scenario folder - Name has to conform "scenario_$nr"
    for file in os.listdir(base_path):

        # Compute the path of to the scenario folder
        scenario_folder_path = base_path + "\\" + file

        # Only take into account scenario directories
        if os.path.isdir(scenario_folder_path):

            # Ensure the folder contains the scenarios
            if len(os.listdir(scenario_folder_path)) == 0:
                print('Empty Scenario Directory. Exiting.')
                break

            # Compute the file path - File path has to start with "scenario_$nr"
            file_path = scenario_folder_path + "\\" + file

            # For each split, create the dataset split and save it
            for split_id in range(SPLITS_NUMBER):
                # Create splits for train / test datasets
                create_single_split(file_path, split_id, train=True, sample_percent=SAMPLE_PERCENT)
                create_single_split(file_path, split_id, train=False, sample_percent=SAMPLE_PERCENT)
    print("Finished creating fresh dataset splits")


def wipe_splits(base_path=DATASET_ROOT_PATH) -> None:
    # Iterate every scenario folder - Name has to conform "scenario_$nr"
    for file in os.listdir(base_path):

        # Compute the path of to the scenario folder
        scenario_folder_path = base_path + "\\" + file

        # Only take into account scenario directories
        if os.path.isdir(scenario_folder_path):

            # Do not proceed if the scenario does only contain the train and test file
            if len(os.listdir(scenario_folder_path)) <= 2:
                print('Directory contains at most train and test splits. Breaking.')
                break

            # Compute the file path - File path has to start with "scenario_$nr"
            file_path = scenario_folder_path + "\\" + file

            # Compute the file name for the splits, remove them
            for split_id in range(SPLITS_NUMBER):
                train_split = file_path + "_train_" + f"{split_id}" + ".json"
                test_split = file_path + "_test_" + f"{split_id}" + ".json"
                os.remove(train_split)
                os.remove(test_split)
        print("Finished wiping the splits")


def process_intent_result(identifier: str, scenario_report_path: str) -> float:
    # Open the current intent report file to be read from
    with io.open(INTENT_REPORT_PATH, "r", encoding="utf-8") as report_file:
        # Open the merged intent report file for writing
        with io.open(MERGED_INTENT_REPORT_PATH, "a", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)
            weighted_average_f1 = content['weighted avg']['f1-score']

            # Write the weighted average for intent detection in the file
            output_file.write(
                "F1-Score - ID for "
                + identifier + f":{weighted_average_f1}\n")
            output_file.close()

            # Compute the path for the current intent report and errors

            report_identifier_path = scenario_report_path \
                                     + "\\intent_reports\\" \
                                     + "report_" + identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                                        "") + ".txt"
            errors_identifier_path = scenario_report_path \
                                     + "\\intent_reports\\" \
                                     + "errors_" + identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                                        "") + ".txt"
            # Open (create) the file
            io.open(report_identifier_path, "w+", encoding="utf-8")
            io.open(errors_identifier_path, "w+", encoding="utf-8")

            # Save the content from the current report to the intent report
            copy_path(INTENT_REPORT_PATH, report_identifier_path)
            copy_path(INTENT_ERRORS_PATH, errors_identifier_path)
        output_file.close()
        report_file.close()

    # Open the current intent errors file to be read from
    with io.open(INTENT_ERRORS_PATH, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(MERGED_INTENT_ERRORS_PATH, "a", encoding="utf-8") as output_file:
            # Load the content
            if os.path.getsize(INTENT_ERRORS_PATH) != 0:
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
    return weighted_average_f1


def process_slot_result(identifier: str, scenario_report_path: str) -> float:
    # Open the current slot report file to be read from
    with io.open(SLOT_REPORT_PATH, "r", encoding="utf-8") as report_file:
        # Open the merged slot report file for writing
        with io.open(MERGED_SLOT_REPORT_PATH, "a", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)

            # Write the weighted average for intent detection in the file
            weighted_average_f1 = content['weighted avg']['f1-score']
            output_file.write(
                "F1-Score - SF for "
                + identifier + f":{weighted_average_f1}\n")

            # Compute the path for the current intent report and errors
            report_identifier_path = scenario_report_path \
                                     + "\\slot_reports\\" \
                                     + "report_" + identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                                        "") + ".txt"
            errors_identifier_path = scenario_report_path \
                                     + "\\slot_reports\\" \
                                     + "errors_" + identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                                        "") + ".txt"

            # Open (create) the file
            io.open(report_identifier_path, "w+", encoding="utf-8")
            io.open(errors_identifier_path, "w+", encoding="utf-8")

            # Save the content from the current report to the intent report
            copy_path(SLOT_REPORT_PATH, report_identifier_path)
            copy_path(SLOT_ERRORS_PATH, errors_identifier_path)
            output_file.close()
        report_file.close()

    # Open the current intent errors file to be read from
    with io.open(SLOT_ERRORS_PATH, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(MERGED_SLOT_ERRORS_PATH, "a", encoding="utf-8") as output_file:
            # Load the content
            if os.path.getsize(SLOT_ERRORS_PATH) != 0:
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
    return weighted_average_f1


def copy_confusion_matrix(identifier: str) -> None:
    copyfile(CONFUSION_MATRIX_PATH,
             MERGED_MATRICES_PATH + "\\" + identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                                "") + ".png")


def run_rasa_pipeline(base_path=DATASET_ROOT_PATH) -> None:
    print("Started running RASA")

    # For each scenario folder
    spreadsheet_row_index = SPREADSHEET_START_VERTICAL_OFFSET
    for file in os.listdir(base_path):

        # Compute the path for the scenario folder
        folder_path = base_path + "\\" + file
        if os.path.isdir(folder_path):

            # Break if the directory does not contain the splits
            if len(os.listdir(folder_path)) <= 2:
                print('Directory contains at most train and test splits. Breaking.')
                break

            # Compute the scenario file name
            file_path = folder_path + "\\" + file

            # Compute the reports path and create the directory
            scenario_reports_path = folder_path + "\\scenario_reports"
            os.mkdir(scenario_reports_path)

            # Compute the intent and slot reports paths and create them
            intent_reports_path = scenario_reports_path + "\\intent_reports"
            slot_reports_path = scenario_reports_path + "\\slot_reports"

            os.mkdir(intent_reports_path)
            os.mkdir(slot_reports_path)

            scenario_slot_results = [f'SF - {file}']
            scenario_intent_results = [f'ID - {file}']

            for split_id in range(SPLITS_NUMBER):
                # Compute the identifier, get the train split and test split
                identifier = f" {file}, split {split_id}"
                train_split = file_path + "_train_" + f"{split_id}" + ".json"
                test_split = file_path + "_test_" + f"{split_id}" + ".json"

                # Run the subprocess for RASA training and testing, and wait for its completion
                command = [SCRIPT_PATH, train_split, test_split, CONFIG_PATH]
                subprocess.Popen(command, shell=True).wait()

                # Process the slot and intent errors & reports and save their return values
                intent_f1 = process_intent_result(identifier, scenario_reports_path)
                slot_f1 = process_slot_result(identifier, scenario_reports_path)

                # Move the confusion matrix to the results path
                copy_confusion_matrix(identifier)

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
            spreadsheet.insert_row(scenario_slot_results, spreadsheet_row_index)
            spreadsheet.insert_row(scenario_intent_results, spreadsheet_row_index)
            spreadsheet_row_index += 3


def wipe_reports():
    # Wipe the merged intent and slot errors and reports
    io.open(MERGED_INTENT_REPORT_PATH, "w", encoding="utf-8").close()
    io.open(MERGED_INTENT_ERRORS_PATH, "w", encoding="utf-8").close()
    io.open(MERGED_SLOT_ERRORS_PATH, "w", encoding="utf-8").close()
    io.open(MERGED_SLOT_REPORT_PATH, "w", encoding="utf-8").close()

    # Wipe the current errors and reports
    io.open(INTENT_REPORT_PATH, "w", encoding="utf-8").close()
    io.open(INTENT_ERRORS_PATH, "w", encoding="utf-8").close()
    io.open(SLOT_ERRORS_PATH, "w", encoding="utf-8").close()
    io.open(SLOT_REPORT_PATH, "w", encoding="utf-8").close()

    # Delete all report directories
    for scenario_folder in os.listdir(DATASET_ROOT_PATH):
        scenario_path = DATASET_ROOT_PATH + f"\\{scenario_folder}"
        if os.path.isdir(scenario_path):
            for file in os.listdir(scenario_path):
                file_path = scenario_path + f"\\{file}"
                if os.path.isdir(file_path):
                    rmtree(file_path)


if __name__ == "__main__":
    wipe_reports()
    wipe_splits()
    # create_splits()
    # run_rasa_pipeline()
