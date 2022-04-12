# astromech-binary-droidspeak
Entirely offline Windows based keyword spotting and binary 'droid speak' beep language translation for your Astromech droid.

A new version of https://github.com/theneverworks/astromech-binary-droidspeak-aiml with authnetic sounds.

# WORK IN PROGRESS
I have a lot of clean up. This is not perfect. I want to add a better mechanism for managing droid profiles without code edits. Maybe command line switches for known droids.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/3LUnMmUf-UM/0.jpg)](http://www.youtube.com/watch?v=3LUnMmUf-UM "R4 Droid Speak Speech Recognition Demo 4")

# Purpose
I wanted to power a home built Star Wars inspired droid with their binary droid speak seen in the movies. I wanted a real experience with natural language understanding and keyword spotting. To achieve this, I employ Windows Speech Recognition and Speech Studio custom keywords to recognize when I’m talking to the droid, e.g., “R2 what is your name?” Once the keyword is detected, a recording of the sound for an adjustable duration is collected. The sound sample is submitted to Deep Speech and the text output is submitted to NLTK for natural language understanding and AIML for conversation if desired. The returned payload is parsed by the code for commands to execute locally and for sound output. I use the “pronouncing” module in python to break the returned text output into one (1) of thirty-nine (39) phonemes by breaking it into syllables and assigning each syllable a group of BleepBox sound files.

# Notes
This code adapts the Microsoft Speech Custom Keyword python sample available through the SDK.

https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/custom-keyword-basics?pivots=programming-language-python

# Prerequisites

## Python
Known to support Python 3.6, other versions not tested but may work.

## Install Pronouncing
https://pypi.org/project/pronouncing/

## Install Sounddevice
https://pypi.org/project/sounddevice/

## Install Playsound
https://pypi.org/project/playsound/

## Install Deepspeech 0.9.3
https://pypi.org/project/deepspeech/

## Download Deepspeech Model Files 0.9.3
https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm

https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

## Install Py-AIML
https://pypi.org/project/python-aiml/

## AIML Files for Personality
https://github.com/pandorabots/Free-AIML

## Download Sound Files
https://github.com/reeltwo/BleepBox/tree/master/assets

Grab all MP3s from the BleepBox repo you wish to use.

Sort them into a file structure as suggested. This is for future sentiment analysis.

conn - Connective Sounds

neg - Responsive Negative

neu - Responsive Neutral

pos - Responsive Positive

proneg - Proactive Negative

proneu - Proactive Neutral

propos - Proactive Positive

# Edits
## droid_speech.py

You must select which droid keyword you want. (R2, BB8, etc.)

Edit the table name to point to the included pretrained models. By default, the droid is called R4.

### Function speech_recognize_keyword_locally_from_microphone()

```
    # Creates an instance of a keyword recognition model. Update this to
    # point to the location of your keyword recognition model.
    model = speechsdk.KeywordRecognitionModel("r4.table")

    # The phrase your keyword recognition model triggers on.
    keyword = "R4"
 ```
 
You could/should adjust the filters that remove the keyword(s) from the payload before it is sent to Watson Assistant. This helps with accuracy but isn’t required. Some of these filters will emerge though reviewing the analytics in Watson Assistant.

```
str = str.replace('are four ','').replace('Are Four ', '').replace('are Four ', '').replace('are 4 ', '').replace('our four ', '').replace('Our Four ', '').replace('our 4 ', '').replace('r 4 ', '').replace('R. for ', '')
```

# Running
`python.exe main.py`
