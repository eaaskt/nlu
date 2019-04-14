[Wit.ai results](#wit.ai-benchmark)

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
**0.90**

### Slot filling
* F1 score: 0.160543
* Accuracy: 0.663467
* Precision: 0.158039
* Recall: 0.163128

### After balancing the dataset
We modified the dataset by removing the original 'flight' intent and grouping all 'x+something' intents
into the original x intent.
#### Intent distribution
![Alt text](pics/atis_no_flight_joint_intent_distribution1.png?raw=true "ATIS intent distribution")

#### Intent detection
![Alt text](pics/atis_no_flight_confusionmat.PNG?raw=true "Confustion matrix for ATIS (balanced)")

| |f1|
|-----|-----|
airline        | 0.974359
x              | 0.000000
capacity       | 0.975610
quantity       | 0.500000
flight_time    | 1.000000
airfare        | 0.849558
flight         | 0.142857
city           | 0.285714
aircraft       | 0.900000
ground_service | 0.972973
distance       | 0.888889
abbreviation   | 1.000000
flight_no      | 0.941176
ground_fare    | 0.833333
airport        | 1.000000
meal           | 0.800000
day_name       | 0.000000

##### Overall F1 (micro-averaged)
**0.88**

#### Slot filling
F1 score: 0.350785
Accuracy: 0.781904
Precision: 0.354497
Recall: 0.347150

## SnipsNLU dataset
### Intent detection
![Alt text](pics/snips_confusionmat.png?raw=true "Confusion matrix for Snips")

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
* F1 score: 0.848382
* Accuracy: 0.896304
* Precision: 0.792994
* Recall: 0.912088

#### BookRestaurant
* F1 score: 0.717258
* Accuracy: 0.807790
* Precision: 0.590726
* Recall: 0.912773


#### GetWeather
* F1 score: 0.749562
* Accuracy: 0.868902
* Precision: 0.650456
* Recall: 0.884298


#### PlayMusic
* F1 score: 0.559036
* Accuracy: 0.690476
* Precision: 0.555024
* Recall: 0.563107


#### RateBook
* F1 score: 0.533708
* Accuracy: 0.603727
* Precision: 0.550725
* Recall: 0.517711


#### SearchCreativeWork
* F1 score: 0.224543
* Accuracy: 0.482014
* Precision: 0.204762
* Recall: 0.248555

#### SearchScreeningEvent
* F1 score: 0.596639
* Accuracy: 0.689820
* Precision: 0.537879
* Recall: 0.669811
 
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


# Rasa benchmark

## ATIS dataset
### Intent detection

![Alt text](pics/rasa_atis_confusionmat.png?raw=true "Confustion matrix for ATIS (balanced)")

| |f1|
|-----|-----|
abbreviation        | 0.970588
aircraft            | 0.615385
airfare             | 0.604651
airfare+flight      | 0.000000
airline             | 0.931507
airport             | 0.882353
capacity            | 0.764706
city                | 0.000000
day_name            | 0.000000
distance            | 0.761905
flight              | 0.941538
flight+airfare      | 0.285714
flight+airline      | 0.000000
flight_no           | 0.769231
flight_no+airline   | 0.000000
flight_time         | 1.000000
ground_fare         | 0.833333
ground_service      | 0.972973
meal                | 0.500000
quantity            | 0.600000


#### Overall F1 score
**0.9857**

### Slot filling
* F1 score: 0.921440
* Accuracy: 0.963579
* Precision: 0.927195
* Recall: 0.915756

## SnipsNLU dataset
### Intent detection
![Alt text](pics/rasa_snips_confusionmat.png?raw=true "Confusion matrix for Snips")

| |f1|
|---------|---------|
AddToPlaylist|0.990099
BookRestaurant|0.995025
GetWeather| 1.00
PlayMusic| 0.989899
RateBook| 1.00
SearchCreativeWork| 0.961165
SearchScreeningEvent| 0.963731

#### Overall F1 score
**0.9857**


### Slot filling
#### AddToPlaylist
* F1 score: 0.899818
* Accuracy: 0.931211
* Precision: 0.894928
* Recall: 0.904762

#### BookRestaurant
* F1 score: 0.939347
* Accuracy: 0.966977
* Precision: 0.937888
* Recall: 0.940810


#### GetWeather
* F1 score: 0.933610
* Accuracy: 0.971545
* Precision: 0.937500
* Recall: 0.929752


#### PlayMusic
* F1 score: 0.820000
* Accuracy: 0.869748
* Precision: 0.845361
* Recall: 0.796117


#### RateBook
* F1 score: 0.968536
* Accuracy: 0.970186
* Precision: 0.972527
* Recall: 0.964578


#### SearchCreativeWork
* F1 score: 0.823881
* Accuracy: 0.911271
* Precision: 0.851852
* Recall: 0.797688

#### SearchScreeningEvent
* F1 score: 0.926366
* Accuracy: 0.938922
* Precision: 0.933014
* Recall: 0.919811

### Overall slot filling results
* F1 score: 0.912665
* Accuracy: 0.940572
* Precision: 0.919638
* Recall: 0.905797