import json
import requests
import argparse
import configparser
import datetime

url = ''
TOKEN = ''


def read_config_file(config_file):
    global url
    global TOKEN
    config = configparser.ConfigParser()
    config.read(config_file)
    url = config['URL']['post_entities_url']
    TOKEN = config['TOKEN']['token']


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
                description="Posts entities to Wit.ai",
                usage="post_entities.py <config_file> <input_file> <app_name")
    parser.add_argument('config_file', help='Config file')
    parser.add_argument('input_file', help='Input file')
    args = parser.parse_args()
    read_config_file(args.config_file)
    post_entities(input_file=args.input_file)

