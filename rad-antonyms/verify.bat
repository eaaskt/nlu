@ECHO off
echo %*
rasa train nlu -u %1 --config config.yml
rasa test nlu -u %2 --config config.yml
