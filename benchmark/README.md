# Wit.ai benchmark

Evaluating the performance of Wit.ai on two datasets: ATIS and SnipsNLU

Library for sequence labeling evaluation: [seqeval](https://github.com/chakki-works/seqeval)
This library was tested with the Conlleval Perl script.

Precision = # of reference slots correctly detected by SLU / # of total slots detected by SLU

Recall = # of reference slots correctly detected by SLU / # of total reference slots

F1 = 2 x Precision x Recall / (Precision + Recall)

Accuracy = # of tokens correctly detected by SLU / # of total tokens

## ATIS dataset 
### Intent detection
![Alt text](pics/atis_confusionmat.PNG?raw=true "Confustion matrix for ATIS")

| |f1|
|-----|-----|
flight_no           |0.000000
airfare             |0.866667
aircraft            |0.947368
meal                |0.500000
flight+airfare      |0.000000
flight              |0.959325
airline             |0.931507
quantity            |0.600000
capacity            |0.950000
day_name            |0.000000
ground_fare         |0.833333
flight_no+airline   |0.000000
abbreviation        |1.000000
airport             |0.909091
city                |0.500000
airfare+flight      |0.000000
flight_time         |1.000000
flight+airline      |0.000000
distance            |0.750000
ground_service      |0.986301

#### Overall F1 (micro-averaged)
**0.93**

### Slot filling
* F1 score: 0.160543
* Accuracy: 0.663467
* Precision: 0.158039
* Recall: 0.163128


## SnipsNLU dataset
### Intent detection
![Alt text](pics/snips_confusionmat.PNG?raw=true "Confusion matrix for Snips")

| |f1|
|---------|---------|
AddToPlaylist|0.985075
BookRestaurant|0.994975
GetWeather| 0.989899
PlayMusic| 0.979592
RateBook| 1.00
SearchCreativeWork|0.947867
SearchScreeningEvent| 0.942408

#### Overall F1 score
**0.97**

#### Incorrect intent predictions

* Can you add last of the ghetto astronauts to the playlist called black sabbath the dio years?
    - True: **AddToPlaylist**
    - Pred: **SearchCreativeWork**

* Lets eat near Oakfield 17 seconds from now at Ted Peters Famous Smoked Fish
    - True: **BookRestaurant**
    - Pred: **SearchScreeningEvent**

* overcast on State Holiday in Pawling Nature Reserve and neighboring places
    - True: **GetWeather**
    - Pred: **x**

* Where is Belgium located
    - True: **GetWeather**
    - Pred: **x**

* Live In L.aJoseph Meyer please
    - True: **PlayMusic**
    - Pred: **x**

* Please tune into Chieko Ochi's good music
    - True: **PlayMusic**
    - Pred: **AddToPlaylist**

* play the top-20 Nawang Khechog soundtrack
    - True: **PlayMusic**
    - Pred: **SearchCreativeWork**

* Put What Color Is Your Sky by Alana Davis on the stereo.
    - True: **PlayMusic**
    - Pred: **AddToPlaylist**

* i want to see JLA Adventures: Trapped In Time
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* Where can I see The Prime Ministers: The Pioneers
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* Can I see Ellis Island Revisited in 1 minute
    - True: **SearchScreeningEvent**
    - Pred: **x**

* show me the schedule for The Oblong Box
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* show the closest movie theatre that shows Boycott
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* I want to see Medal for the General
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* I want to see Fear Chamber.
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* I want to see Shattered Image.
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* I want to see Outcast.
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

* Can you check the timings for Super Sweet 16: The Movie?
    - True: **SearchScreeningEvent**
    - Pred: **SearchCreativeWork**

### Slot filling
#### AddToPlaylist
* F1 score:   0.446337
* Accuracy:   0.562628
* Precision:  0.417197
* Recall:     0.479853


#### BookRestaurant
* F1 score: 0.602203
* Accuracy: 0.737553
* Precision: 0.495968
* Recall: 0.766355


#### GetWeather
* F1 score: 0.357268
* Accuracy: 0.581818
* Precision: 0.310030
* Recall: 0.421488


#### PlayMusic
* F1 score: 0.269880
* Accuracy: 0.463687
* Precision: 0.267943
* Recall: 0.271845


#### RateBook
* F1 score: 0.081461
* Accuracy: 0.229529
* Precision: 0.084058
* Recall: 0.079019


#### SearchCreativeWork
* F1 score: 0.089005
* Accuracy: 0.356886
* Precision: 0.081340
* Recall: 0.098266

#### SearchScreeningEvent
* F1 score: 0.016807
* Accuracy: 0.231591
* Precision: 0.015152
* Recall: 0.018868
 
The overall scores for each intent are summarized in the table below:
 
| |accuracy| precision  | recall    | f1-score
|----------|----------|----------|----------|----------|
AddToPlaylist|0.90|0.79|0.91|0.85
BookRestaurant|0.81 |0.59 |0.91 | 0.72
GetWeather| 0.87|0.65 |0.88 | 0.75
PlayMusic| 0.69| 0.55| 0.56| 0.56
RateBook| 0.60| 0.55| 0.52| 0.53
SearchCreativeWork|0.48 | 0.20| 0.25| 0.22
SearchScreeningEvent| 0.69| 0.54| 0.67| 0.60
