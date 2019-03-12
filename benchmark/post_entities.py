import json
import requests
import argparse
import configparser
import datetime

url = ''
TOKEN = ''


def read_config_file(config_file, app_name):
    global url
    global TOKEN
    config = configparser.ConfigParser()
    config.read(config_file)
    url = config['URL']['post_entities_url']
    if app_name == 'snips':
        TOKEN = config['TOKEN']['snips_token']
    else:
        TOKEN = config['TOKEN']['atis_token']


def post_entities(input_file):
    global url
    now = datetime.datetime.now()
    url = url + str(now.year) + '{num:02d}'.format(num=now.month) + '{num:02d}'.format(num=now.day)
    print(url)
    print(TOKEN)
    headers = {
        'Authorization': 'Bearer ' + TOKEN,
        'Content-Type': 'application/json'
    }

    with open(input_file) as inf:
        json_data = json.loads(inf.read())

    for entity in json_data:
        entities_data = {'doc': entity['entity'],
                         'id': entity['entity']}
        try:
            r = requests.post(url, data=json.dumps(entities_data), headers=headers)
            print(r.content)
        except Exception as ex:
            print(ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Extract entities from sample Wit.ai format",
                usage="entity_extractor.py <input_file> <output_file>")
    parser.add_argument('config_file', help='Config file')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('app', help='App to post data to')
    args = parser.parse_args()
    read_config_file(args.config_file, args.app)
    post_entities(input_file=args.input_file)

