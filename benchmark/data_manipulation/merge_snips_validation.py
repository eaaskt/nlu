import json


def merge(file_list, output_path):
    merged_data = []
    id_count = 1
    for f_path in file_list:
        with open(f_path, errors='replace', encoding='utf-8') as f:
            data = json.load(f)

        for d in data:
            d['id'] = id_count
            id_count += 1
            merged_data.append(d)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    validation_file_list = ['..\\data\\snips\\add-to-playlist\\validate_AddToPlaylist_wit_iob.json',
                            '..\\data\\snips\\book-restaurant\\validate_BookRestaurant_wit_iob.json',
                            '..\\data\\snips\\get-weather\\validate_GetWeather_wit_iob.json',
                            '..\\data\\snips\\play-music\\validate_PlayMusic_wit_iob.json',
                            '..\\data\\snips\\rate-book\\validate_RateBook_wit_iob.json',
                            '..\\data\\snips\\search-creative-work\\validate_SearchCreativeWork_wit_iob.json',
                            '..\\data\\snips\\search-screening-event\\validate_SearchScreeningEvent_wit_iob.json']

    validation_resp_list = ['..\\data\\snips\\add-to-playlist\\validation_response.json',
                            '..\\data\\snips\\book-restaurant\\validation_response.json',
                            '..\\data\\snips\\get-weather\\validation_response.json',
                            '..\\data\\snips\\play-music\\validation_response.json',
                            '..\\data\\snips\\rate-book\\validation_response.json',
                            '..\\data\\snips\\search-creative-work\\validation_response.json',
                            '..\\data\\snips\\search-screening-event\\validation_response.json']

    rasa_validation_resp_list = ['..\\data\\snips\\add-to-playlist\\validation_response_rasa.json',
                                 '..\\data\\snips\\book-restaurant\\validation_response_rasa.json',
                                 '..\\data\\snips\\get-weather\\validation_response_rasa.json',
                                 '..\\data\\snips\\play-music\\validation_response_rasa.json',
                                 '..\\data\\snips\\rate-book\\validation_response_rasa.json',
                                 '..\\data\\snips\\search-creative-work\\validation_response_rasa.json',
                                 '..\\data\\snips\\search-screening-event\\validation_response_rasa.json']

    merge(validation_file_list, '..\\data\\snips\\all\\validate_wit_iob.json')
    merge(validation_resp_list, '..\\data\\snips\\all\\validation_response.json')
    merge(rasa_validation_resp_list, '..\\data\\snips\\all\\validation_response_rasa.json')

