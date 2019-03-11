import json
import requests

TOKEN = ''
url = 'https://api.wit.ai/entities?v=20170307'  # TODO: dynamic date
entities1 = {'doc': 'Name',
       'id': 'entity_name'}
headers = {
    'Authorization': 'Bearer ' + TOKEN,
    'Content-Type': 'application/json'
}
# r = requests.post(url, data=json.dumps(entities1), headers=headers)
# print(r.content)

url_samples = 'https://api.wit.ai/samples?v=20170307'  # TODO: dynamic date
sample_data2 = [
    {
        "text": "add Stani, stani Ibar vodo songs in my playlist música libre",
        "entities": [
            {
                "entity": "intent",
                "value": "AddToPlaylist"
            },
            {
                "start": 4,
                "end": 26,
                "value": "Stani, stani Ibar vodo",
                "entity": "entity_name"
            },
            {
                "start": 36,
                "end": 38,
                "value": "my",
                "entity": "playlist_owner"
            },
            {
                "start": 48,
                "end": 61,
                "value": "música libre",
                "entity": "playlist"
            }
        ]
    },
    {
        "text": "add this album to my Blues playlist",
        "entities": [
            {
                "entity": "intent",
                "value": "AddToPlaylist"
            },
            {
                "start": 9,
                "end": 14,
                "value": "album",
                "entity": "music_item"
            },
            {
                "start": 18,
                "end": 20,
                "value": "my",
                "entity": "playlist_owner"
            },
            {
                "start": 21,
                "end": 26,
                "value": "Blues",
                "entity": "playlist"
            }
        ]
    }
    ]
r = requests.post(url_samples, data=json.dumps(sample_data2), headers=headers)
print(r.content)
