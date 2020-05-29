Things to keep in mind:

* Need to modify the scripts so that absolute and relative slots are generated separately (currently they are marked as the same slot - e.g. [grade#relativ] and [grade#absolut] are both marked as [grade] in the generated dataset).
* In *intreabaEventCalendar* and *adaugaEventCalendar*, the Chatito scripts might generate sentences in which [ora_start] > [ora_final]. This can be fixed in the scripts themselves, but the reasoning system should at some point check for this edge case (even though the chances of a user making this mistake when giving a command are slim).
