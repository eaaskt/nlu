# Wit.ai benchmark

Evaluating the performance of Wit.ai on two datasets: ATIS and SnipsNLU

Library for sequence labeling evaluation: [seqeval](https://github.com/chakki-works/seqeval)

Intent detection evaluation:
   * error rate
   * confusion matrix
  
## SnipsNLU dataset
### Intent detection
![Alt text](pics/snips_confusionmat.PNG?raw=true "Confusion matrix for Snips")

### Slot filling
#### AddToPlaylist
* F1 score:   0.446337
* Accuracy:   0.562628
* Precision:  0.417197
* Recall:     0.479853

| |precision|    recall|  f1-score|   support|
|---------|----------|----------|----------|----------|
|playlist |      0.46|      0.48|      0.47|       100|
artist    |   0.51   |   0.48   |   0.49   |     46   |
playlist_owner|       0.50|      0.50|      0.50|        54|
music_item|       0.52|      0.55|      0.53|        55|
entity_name|       0.27|      0.22|      0.24|        18|
**avg / total**|       **0.48**|      **0.48**|      **0.48**|       **273**|

#### BookRestaurant
* F1 score: 0.602203
* Accuracy: 0.737553
* Precision: 0.495968
* Recall: 0.766355

| |precision|    recall|  f1-score|   support|
|---------|----------|----------|----------|----------|
party_size_description|       0.82|      0.69|      0.75|        13|
served_dish|       0.50|      0.80|      0.62|         5|
timeRange|       0.76|      0.74|      0.75|        34|
state|       0.21|      0.88|      0.34|        25|
restaurant_type|       0.74|      0.79|      0.77|        62|
party_size_number|       0.80|      0.82|      0.81|        57|
spatial_relation|       0.74|      0.74|      0.74|        19|
city|       0.58|      0.55|      0.56|        33|
restaurant_name|       0.79|      0.75|      0.77|        20|
country|       0.90|      0.90|      0.90|        20|
poi|       0.75|      0.50|      0.60|         6|
sort|       0.78|      0.78|      0.78|         9|
cuisine|       0.80|      0.73|      0.76|        11|
facility|       1.00|      1.00|      1.00|         7|             
**avg / total**|       **0.72**|      **0.77**|      **0.72**|       **321**|

#### GetWeather
* F1 score: 0.357268
* Accuracy: 0.581818
* Precision: 0.310030
* Recall: 0.421488

| |precision|    recall|  f1-score|   support|
|---------|----------|----------|----------|----------|
timeRange |      0.28|      0.30|      0.29|        64|
current_location |      0.41|      0.41|      0.41|        17|
state |      0.17|      0.62|      0.26|        26|
condition_temperature |      0.38|      0.38|      0.38|       21|
condition_description |      0.45|      0.45|      0.45|       22|
city |      0.41|      0.39|      0.40|        38|
geographic_poi |      0.43|      0.38|      0.40|        16|
country |      0.52|      0.54|      0.53|        24|
spatial_relation |      0.50|      0.57|      0.53|        14|
**avg / total**  |     **0.37**|      **0.42**|      **0.38**|       **242**|

#### PlayMusic
* F1 score: 0.269880
* Accuracy: 0.463687
* Precision: 0.267943
* Recall: 0.271845

| |precision|    recall|  f1-score|   support|
|---------|----------|----------|----------|----------|
      album|       0.00|      0.00|      0.00|        13|
    service|       0.46|      0.33|      0.39|        39|
     artist|       0.41|      0.25|      0.31|        63|
 music_item|       0.38|      0.35|      0.37|        31|
      track|       0.00|      0.00|      0.00|         6|
       year|       0.61|      0.44|      0.51|        25|
       sort|       0.25|      0.12|      0.16|        17|
   playlist|       0.08|      0.33|      0.12|         9|
      genre|       0.00|      0.00|      0.00|         3|
avg / total|      0.37 |     0.27 |     0.31 |      206|

#### RateBook
* F1 score: 0.081461
* Accuracy: 0.229529
* Precision: 0.084058
* Recall: 0.079019

| |precision|    recall|  f1-score|   support|
|---------|----------|----------|----------|----------|
object_name|       0.00|      0.00|      0.00|        51
object_type|       0.15|      0.12|      0.14|        40
object_select|       0.17|      0.16|      0.17|        49
rating_unit|       0.17|      0.16|      0.17|        61
object_part_of_series_type|       0.17|      0.13|      0.15|        15
rating_value|       0.11|      0.04|      0.06|       100
best_rating|       0.00|      0.00|      0.00|        51
avg / total|       0.10|      0.08|      0.09|       367

#### SearchCreativeWork
* F1 score: 0.089005
* Accuracy: 0.356886
* Precision: 0.081340
* Recall: 0.098266

| |precision|    recall|  f1-score|   support|
|---------|----------|----------|----------|----------|
object_name|       0.36 |     0.04|      0.07|       100|
object_type|       0.33 |     0.18|      0.23|        73|
avg / total|       0.35 |     0.10|      0.14|       173|

#### SearchScreeningEvent
* F1 score: 0.016807
* Accuracy: 0.231591
* Precision: 0.015152
* Recall: 0.018868

| |precision|    recall|  f1-score|   support|
|---------|----------|----------|----------|----------|
spatial_relation|       0.00 |     0.00|      0.00|        35|
movie_name|       0.14 |     0.02|      0.04|        49|
object_type|       0.03 |     0.05|      0.04|        43|
object_location_type|       0.00 |     0.00|      0.00|        20|
location_name|       0.00 |     0.00|      0.00|        29|
movie_type|       0.04 |     0.04|      0.04|        24|
timeRange |      0.00 |     0.00|      0.00|        12|
avg / total |      0.04 |     0.02|      0.02 |      212|
 
 
The overall scores for each intent are summarized in the table below:
        
| |accuracy| precision  | recall    | f1-score
|----------|----------|----------|----------|----------|
AddToPlaylist|0.56|0.42|0.48|0.45
BookRestaurant|0.74 |0.50 |0.77 | 0.60
GetWeather| 0.58|0.31 |0.42 | 0.36
PlayMusic| 0.46| 0.27| 0.27| 0.27
RateBook| 0.23| 0.08| 0.08| 0.08
SearchCreativeWork|0.36 | 0.08| 0.10| 0.09
SearchScreeningEvent| 0.23| 0.01| 0.02| 0.02
