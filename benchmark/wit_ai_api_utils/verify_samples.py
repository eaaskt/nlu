import json
import requests
import argparse
import configparser
import datetime

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import data_evaluation.eval_converter as eval_conv

url = ''
TOKEN = ''


def read_config_file(config_file, app_name):
    global url
    global TOKEN
    config = configparser.ConfigParser()
    config.read(config_file)
    url = config['URL']['get_message']
    if app_name == 'snips':
        TOKEN = config['TOKEN']['snips_token']
    elif app_name == 'atis':
        TOKEN = config['TOKEN']['atis_token']
    else:
        TOKEN = config['TOKEN']['test_token']


def get_messages(input_file, output_file, failed_file):
    global url
    now = datetime.datetime.now()
    url = url + str(now.year) + '{num:02d}'.format(num=now.month) + '{num:02d}'.format(num=now.day) + '&q='
    print(url)
    print(TOKEN)
    headers = {
        'Authorization': 'Bearer ' + TOKEN,
        'Content-Type': 'application/json'
    }

    with open(input_file) as inf:
        json_data = json.loads(inf.read())

    failed_list = []
    response = []
    for sample in json_data:
        sample_url = url + sample['text'] + "&verbose=True"
        try:
            r = requests.get(sample_url, headers=headers)
            print(r.content)
            if r.status_code != 200:
                failed_list.append(sample)
            else:
                # process response to save in file
                resp = json.loads(r.content.decode('utf-8'))
                resp = eval_conv.convert_iob_prediction(resp)
                resp['id'] = sample['id']
                response.append(resp)

        except Exception as e:
            print(e)

    with open(failed_file, "w") as ff:
        json.dump(failed_list, ff)

    with open(output_file, "w") as of:
        json.dump(response, of, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Get results from Wit.ai for validation set",
                usage="post_entities.py <config_file> <input_file> <output_file> <failed_samples_file> <app_name>")
    parser.add_argument('config_file', help='Config file')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('failed_file', help='failed file')
    parser.add_argument('app', help='App to post data to')
    args = parser.parse_args()
    read_config_file(args.config_file, args.app)
    get_messages(input_file=args.input_file, failed_file=args.failed_file, output_file=args.output_file)

