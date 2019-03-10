import json
import requests

TOKEN = ''
url = 'https://api.wit.ai/entities?v=20170307'  # TODO: dynamic date
data = {'doc': 'A city that I like',
       'id': 'favorite_city'}    # Set POST fields here
headers = {
    'Authorization': 'Bearer ' + TOKEN,
    'Content-Type': 'application/json'
}
# r = requests.post(url, data=json.dumps(data), headers=headers)
# print(r.content)

url_samples = 'https://api.wit.ai/samples?v=20170307'  # TODO: dynamic date
sample_data = [{
        "text": "I like San Francisco",
        "entities": [
          {
            "entity": "intent",
            "value": "stating_fact"
          },
          {
            "entity": "favorite_city",
            "start": 7,
            "end": 19,
            "value": "San Francisco"
          }
        ]
      }]
r = requests.post(url_samples, data=json.dumps(sample_data), headers=headers)
print(r.content)