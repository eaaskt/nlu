Version 1 contains the original datasets. For scenarios 1, 2, and 3 we keep the training 
and testing distributions separate.

Version 2 is the version in which we allow one test distribution sample per intent in the 
training set.



#### Updated datasets
We discovered an error for the schimbaIntensitateMuzica intent - the slot was 
missing from the dataset (i.e. it was not labelled) --> we modified the existing
datasets so that it contains the label. At the moment, only V1 is updated, only train
and test - the validation results are still the old ones.

##### TODOS
* At some point we should redo the benchmarks with the new modifications.
* Also update v2 of dataset
    * Remove invalid tv channels
    * slot for changeVolume
 
#### June update
Replaced the schimbaIntensitateMuzica intent with the two intents: cresteIntensitateMuzica and scadeIntensitateMuzica, to maintain consistency with the rest of the dataset. Currently this change is visible only in the diacritics dataset, in a separate folder.