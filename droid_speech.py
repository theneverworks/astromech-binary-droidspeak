#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# Contains code samples from Pronouncing Tutorialss
# https://pronouncing.readthedocs.io/en/latest/tutorial.html
"""
Speech recognition samples for the Microsoft Cognitive Services Speech SDK
"""

import glob
import sys
import platform
import time
import nltk
from nltk.tokenize import word_tokenize
import wave
import soundfile as sf
import re
import os
import urllib.request  as urllib2
import json
from os.path import join, dirname 
import sounddevice as sd
import re
import pronouncing
import deepspeech
import operator
import marshal
import aiml
import csv
import numpy as np
import serial
#from nltk.sentiment import SentimentIntensityAnalyzer
from playsound import playsound

model_file_path = 'C:\\ai\\R4\\deepspeech-0.9.3-models.pbmm'
dsmodel = deepspeech.Model(model_file_path)
scorer_file_path = 'C:\\ai\\R4\\deepspeech-0.9.3-models.scorer'
dsmodel.enableExternalScorer(scorer_file_path)
lm_alpha = 0.75
lm_beta = 1.85
dsmodel.setScorerAlphaBeta(lm_alpha, lm_beta)
beam_width = 500
dsmodel.setBeamWidth(beam_width)
#sia = SentimentIntensityAnalyzer()

connecteddevicecount = 0
try:
    ser0 = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
    connecteddevicecount += 1
except:
    ser0 = None

# AIML Directory
saiml = "C:\\ai\\R4\\aiml\\"

# brain
k = aiml.Kernel()

# setpreds() function
def setpreds():
    with open(saiml + 'preds.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            #print((row[0]), (row[1]))
            k.setBotPredicate((row[0]), (row[1]))
    plat = platform.machine()
    osys = os.name
    print("R2 for " + osys)
    print("System Architecture " + plat)
    #print "Memory " + psutil.virtual_memory()
    k.setBotPredicate("architecture", plat)
    k.setBotPredicate("os", osys)


# get_oldest_file() function
def get_oldest_file(files, _invert=False):
    """ Find and return the oldest file of input file names.
    Only one wins tie. Values based on time distance from present.
    Use of `_invert` inverts logic to make this a youngest routine,
    to be used more clearly via `get_youngest_file`.
    """
    gt = operator.lt if _invert else operator.gt
    # Check for empty list.
    if not files:
        return None
    # Raw epoch distance.
    now = time.time()
    # Select first as arbitrary sentinel file, storing name and age.
    oldest = files[0], now - os.path.getmtime(files[0])
    # Iterate over all remaining files.
    for f in files[1:]:
        age = now - os.path.getmtime(f)
        if gt(age, oldest[1]):
            # Set new oldest.
            oldest = f, age
    # Return just the name of oldest file.
    return oldest[0]


# get_youngest_file() function
def get_youngest_file(files):
    return get_oldest_file(files, _invert=True)


# learn() function
def learn(aimlfiles):
    if not aimlfiles:
        k.learn(saiml + "xfind.aiml")
    for f in aimlfiles[1:]:
        k.learn(f)


# brain() function
def brain():
    if os.path.isfile(saiml + "R2.brn"):
        brnfiles = glob.glob(saiml + "*.brn")
        aimlfiles = glob.glob(saiml + "*.aiml")
        brn = get_oldest_file(brnfiles)
        oldage = os.path.getmtime(brn)
        youngaiml = get_youngest_file(aimlfiles)
        youngage = os.path.getmtime(youngaiml)
        testfiles = [brn, youngaiml]
        oldfile = get_oldest_file(testfiles)
        if (oldfile != (saiml + "R2.brn")):
            k.bootstrap(brainFile=saiml + "R2.brn")
        else:
            learn(aimlfiles)
    else:
        aimlfiles = glob.glob(saiml + "*.aiml")
        learn(aimlfiles)
        setpreds()
    if os.path.isfile(saiml + "R2.ses"):
        sessionFile = file(saiml + "R2.ses", "rb")
        session = marshal.load(sessionFile)
        sessionFile.close()
        for pred,value in session.items():
            k.setPredicate(pred, value, "R2")
    else:
        setpreds()
    k.saveBrain(saiml + "R2.brn")

brain()

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for
    installation instructions.
    """)
    import sys
    sys.exit(1)

def playnote(n, emote):
    if n == 'AA':
        playsound('C:/ai/R4/sounds/neu/chat-1-0.wav')
        playsound('C:/ai/R4/sounds/neu/chat-1-1.wav')
    elif n == 'AE':
        playsound('C:/ai/R4/sounds/neu/chat-1-2.wav')
        playsound('C:/ai/R4/sounds/neu/chat-1-3.wav')
    elif n == 'AH':
        playsound('C:/ai/R4/sounds/neu/chat-1-4.wav')
        playsound('C:/ai/R4/sounds/neu/chat-1-5.wav')
    elif n == 'AO':
        playsound('C:/ai/R4/sounds/neu/chat-1-6.wav')
        playsound('C:/ai/R4/sounds/neu/chat-1-7.wav')
    elif n == 'AW':
        playsound('C:/ai/R4/sounds/neu/chat-1-8.wav')
        playsound('C:/ai/R4/sounds/neu/chat-1-9.wav')
    elif n == 'AY':
        playsound('C:/ai/R4/sounds/neu/chat-2-0.wav')
        playsound('C:/ai/R4/sounds/neu/chat-2-1.wav')
    elif n == 'B':
        playsound('C:/ai/R4/sounds/neu/chat-2-2.wav')
        playsound('C:/ai/R4/sounds/neu/chat-2-3.wav')
    elif n == 'CH':
        playsound('C:/ai/R4/sounds/neu/chat-2-4.wav')
        playsound('C:/ai/R4/sounds/neu/chat-2-5.wav')
    elif n == 'D':
        playsound('C:/ai/R4/sounds/neu/chat-2-6.wav')
        playsound('C:/ai/R4/sounds/neu/chat-2-7.wav')
    elif n == 'DH':
        playsound('C:/ai/R4/sounds/neu/chat-2-8.wav')
        playsound('C:/ai/R4/sounds/neu/chat-2-9.wav')
    elif n == 'EH':
        playsound('C:/ai/R4/sounds/neu/chat-2-10.wav')
        playsound('C:/ai/R4/sounds/neu/chat-3-0.wav')
    elif n == 'ER':
        playsound('C:/ai/R4/sounds/neu/chat-3-1.wav')
        playsound('C:/ai/R4/sounds/neu/chat-4-0.wav')
    elif n == 'EY':
        playsound('C:/ai/R4/sounds/neu/chat-4-1.wav')
        playsound('C:/ai/R4/sounds/neu/chat-4-2.wav')
    elif n == 'F':
        playsound('C:/ai/R4/sounds/neu/chat-4-3.wav')
        #playsound('C:/ai/R4/sounds/neu/chat-4-4.wav')
    elif n == 'G':
        playsound('C:/ai/R4/sounds/neu/chat-4-5.wav')
        playsound('C:/ai/R4/sounds/neu/chat-4-6.wav')
    elif n == 'HH':
        playsound('C:/ai/R4/sounds/neu/chat-5-0.wav')
        playsound('C:/ai/R4/sounds/neu/chat-5-1.wav')
    elif n == 'IH':
        playsound('C:/ai/R4/sounds/neu/chat-5-2.wav')
        playsound('C:/ai/R4/sounds/neu/chat-5-3.wav')
    elif n == 'IY':
        playsound('C:/ai/R4/sounds/neu/chat-5-4.wav')
        playsound('C:/ai/R4/sounds/neu/chat-5-5.wav')
    elif n == 'JH':
        playsound('C:/ai/R4/sounds/neu/chat-5-6.wav')
        playsound('C:/ai/R4/sounds/neu/chat-5-7.wav')
    elif n == 'K':
        playsound('C:/ai/R4/sounds/neu/chat-5-8.wav')
        playsound('C:/ai/R4/sounds/neu/chat-5-9.wav')
    elif n == 'L':
        playsound('C:/ai/R4/sounds/neu/chat-6-0.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-1.wav')
    elif n == 'M':
        playsound('C:/ai/R4/sounds/neu/chat-6-2.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-3.wav')
    elif n == 'N':
        playsound('C:/ai/R4/sounds/neu/chat-6-4.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-5.wav')
    elif n == 'NG':
        playsound('C:/ai/R4/sounds/neu/chat-6-6.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-7.wav')
    elif n == 'OW':
        playsound('C:/ai/R4/sounds/neu/chat-6-8.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-9.wav')
    elif n == 'OY':
        playsound('C:/ai/R4/sounds/neu/chat-6-10.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-11.wav')
    elif n == 'P':
        playsound('C:/ai/R4/sounds/neu/chat-6-12.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-13.wav')
    elif n == 'R':
        playsound('C:/ai/R4/sounds/neu/chat-6-14.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-15.wav')
    elif n == 'S':
        playsound('C:/ai/R4/sounds/neu/chat-6-16.wav')
        playsound('C:/ai/R4/sounds/neu/chat-6-17.wav')
    elif n == 'SH':
        playsound('C:/ai/R4/sounds/neu/chat-7-0.wav')
        playsound('C:/ai/R4/sounds/neu/chat-7-1.wav')
    elif n == 'T':
        playsound('C:/ai/R4/sounds/neu/chat-7-2.wav')
        playsound('C:/ai/R4/sounds/neu/chat-7-3.wav')
    elif n == 'TH':
        playsound('C:/ai/R4/sounds/neu/chat-7-4.wav')
        playsound('C:/ai/R4/sounds/neu/chat-7-5.wav')
    elif n == 'UH':
        playsound('C:/ai/R4/sounds/neu/chat-7-6.wav')
        playsound('C:/ai/R4/sounds/neu/chat-8-0.wav')
    elif n == 'UW':
        playsound('C:/ai/R4/sounds/neu/chat-8-1.wav')
        playsound('C:/ai/R4/sounds/neu/chat-8-2.wav')
    elif n == 'V':
        playsound('C:/ai/R4/sounds/neu/chat-8-3.wav')
        playsound('C:/ai/R4/sounds/neu/chat-8-4.wav')
    elif n == 'W':
        playsound('C:/ai/R4/sounds/neu/chat-8-5.wav')
        playsound('C:/ai/R4/sounds/neu/chat-8-6.wav')
    elif n == 'Y':
        playsound('C:/ai/R4/sounds/neu/chat-8-7.wav')
    elif n == 'Z':
        playsound('C:/ai/R4/sounds/neu/chat-9-0.wav')
        playsound('C:/ai/R4/sounds/neu/chat-9-1.wav')
    elif n == 'ZH':
        playsound('C:/ai/R4/sounds/neu/chat-9-2.wav')
        playsound('C:/ai/R4/sounds/neu/chat-10-0.wav')
    else:
        playsound('C:/ai/R4/sounds/neu/chat-10-1.wav')
        playsound('C:/ai/R4/sounds/neu/chat-10-2.wav')
    return

def mechsay(message):
    try:
        #sentscore = sia.polarity_scores(message)
        #compsentscore = sia.polarity_scores(message)["compound"]
        #possentscore = sia.polarity_scores(message)["pos"]
        #negsentscore = sia.polarity_scores(message)["neg"]
        #neusentscore = sia.polarity_scores(message)["neu"]
        #print(possentscore)
        #print(negsentscore)
        #neusentscore = (1 - (possentscore + negsentscore))
        #print(neusentscore)
        send = message or "Ok"
        response = "Ok"
        send = send.replace('\r\n','').replace('\n','').replace('\\n','').replace('\\r\\n','').replace('\r','')

        text = send
        words = text.split()
        WordToPhn=[]
        for word in words:
            try:
                pronunciation_list = pronouncing.phones_for_word(word)[0] # choose the first version of the phoneme
                WordToPhn.append(pronunciation_list)
            except IndexError:
                continue

        SentencePhn=' '.join(WordToPhn) 
        Output = re.sub(r'\d+', '', SentencePhn) #Remove the digits in phonemes

        #print(Output)

        soundlist = Output.split(' ')

        for x in soundlist:
            frequency = assignnote(x)
            duration = 100  # In ms, 1000 ms = 1 second
            try:
                winsound.Beep(frequency, duration)
            except:
                print("Beep failed.")
        return response
    except:
        print("Unable to beep back")
        response = "Unable to beep back"
        return response

def handle_input(testutterance):
    sent = nltk.word_tokenize(testutterance.lower())
    if ("light" in sent and "saber" in sent):
        ser0.write(b'M')
        response = "Ok"
    elif ("settle" in sent and "down" in sent) or ("put" in sent and "away" in sent):
        ser0.write(b'S')
        response = "Fine"
    else:
        response = k.respond(testutterance)
    mechsay(response)

# Set up the subscription info for the Speech Service:
# Replace with your own subscription key and service region (e.g., "westus").
speech_key, service_region = "none", "none"

# Specify the path to an audio file containing speech (mono WAV / PCM with a sampling rate of 16
# kHz).
weatherfilename = "whatstheweatherlike.wav"
weatherfilenamemp3 = "whatstheweatherlike.mp3"


def speech_recognize_once_from_mic():
    """performs one-shot speech recognition from the default microphone"""
    # <SpeechRecognitionWithMicrophone>
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Creates a speech recognizer using microphone as audio input.
    # The default language is "en-us".
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    # </SpeechRecognitionWithMicrophone>


def speech_recognize_once_from_file():
    """performs one-shot speech recognition with input from an audio file"""
    # <SpeechRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)
    # Creates a speech recognizer using a file as audio input, also specify the speech language
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, language="de-DE", audio_config=audio_config)

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    # </SpeechRecognitionWithFile>


def speech_recognize_once_compressed_input():
    """performs one-shot speech recognition with compressed input from an audio file"""
    # <SpeechRecognitionWithCompressedFile>
    class BinaryFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
        def __init__(self, filename: str):
            super().__init__()
            self._file_h = open(filename, "rb")

        def read(self, buffer: memoryview) -> int:
            try:
                size = buffer.nbytes
                frames = self._file_h.read(size)

                buffer[:len(frames)] = frames

                return len(frames)
            except Exception as ex:
                print('Exception in `read`: {}'.format(ex))
                raise

        def close(self) -> None:
            print('closing file')
            try:
                self._file_h.close()
            except Exception as ex:
                print('Exception in `close`: {}'.format(ex))
                raise
    # Creates an audio stream format. For an example we are using MP3 compressed file here
    compressed_format = speechsdk.audio.AudioStreamFormat(compressed_stream_format=speechsdk.AudioStreamContainerFormat.MP3)
    callback = BinaryFileReaderCallback(filename=weatherfilenamemp3)
    stream = speechsdk.audio.PullAudioInputStream(stream_format=compressed_format, pull_stream_callback=callback)

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # Creates a speech recognizer using a file as audio input, also specify the speech language
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    # </SpeechRecognitionWithCompressedFile>


def speech_recognize_once_from_file_with_customized_model():
    """performs one-shot speech recognition with input from an audio file, specifying a custom
    model"""
    # <SpeechRecognitionUsingCustomizedModel>
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Create source language configuration with the speech language and the endpoint ID of your customized model
    # Replace with your speech language and CRIS endpoint ID.
    source_language_config = speechsdk.languageconfig.SourceLanguageConfig("zh-CN", "YourEndpointId")

    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)
    # Creates a speech recognizer using a file as audio input and specify the source language config
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, source_language_config=source_language_config, audio_config=audio_config)

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    # </SpeechRecognitionUsingCustomizedModel>


def speech_recognize_once_from_file_with_custom_endpoint_parameters():
    """performs one-shot speech recognition with input from an audio file, specifying an
    endpoint with custom parameters"""
    initial_silence_timeout_ms = 15 * 1e3
    template = "wss://{}.stt.speech.microsoft.com/speech/recognition" \
            "/conversation/cognitiveservices/v1?initialSilenceTimeoutMs={:d}"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key,
            endpoint=template.format(service_region, int(initial_silence_timeout_ms)))
    print("Using endpoint", speech_config.get_property(speechsdk.PropertyId.SpeechServiceConnection_Endpoint))
    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)
    # Creates a speech recognizer using a file as audio input.
    # The default language is "en-us".
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def speech_recognize_async_from_file():
    """performs one-shot speech recognition asynchronously with input from an audio file"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)
    # Creates a speech recognizer using a file as audio input.
    # The default language is "en-us".
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Perform recognition. `recognize_async` does not block until recognition is complete,
    # so other tasks can be performed while recognition is running.
    # However, recognition stops when the first utterance has bee recognized.
    # For long-running recognition, use continuous recognitions instead.
    result_future = speech_recognizer.recognize_once_async()

    print('recognition is running....')
    # Other tasks can be performed here...

    # Retrieve the recognition result. This blocks until recognition is complete.
    result = result_future.get()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def speech_recognize_continuous_from_file():
    """performs continuous speech recognition with input from an audio file"""
    # <SpeechContinuousRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()
    # </SpeechContinuousRecognitionWithFile>

def speech_recognize_keyword_from_microphone():
    """performs keyword-triggered speech recognition with input microphone"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Creates an instance of a keyword recognition model. Update this to
    # point to the location of your keyword recognition model.
    model = speechsdk.KeywordRecognitionModel("r4.table")

    # The phrase your keyword recognition model triggers on.
    keyword = "R4"

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    done = False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    def recognizing_cb(evt):
        """callback for recognizing event"""
        if evt.result.reason == speechsdk.ResultReason.RecognizingKeyword:
            print('RECOGNIZING KEYWORD: {}'.format(evt))
        elif evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            print('RECOGNIZING: {}'.format(evt))

    def recognized_cb(evt):
        """callback for recognized event"""
        if evt.result.reason == speechsdk.ResultReason.RecognizedKeyword:
            print('RECOGNIZED KEYWORD: {}'.format(evt))
        elif evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print('RECOGNIZED: {}'.format(evt))
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('NOMATCH: {}'.format(evt))

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start keyword recognition
    speech_recognizer.start_keyword_recognition(model)
    print('Say something starting with "{}" followed by whatever you want...'.format(keyword))
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_keyword_recognition()

def speech_recognition_with_pull_stream():
    """gives an example how to use a pull audio stream to recognize speech from a custom audio
    source"""
    class WavFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
        """Example class that implements the Pull Audio Stream interface to recognize speech from
        an audio file"""
        def __init__(self, filename: str):
            super().__init__()
            self._file_h = wave.open(filename, mode=None)

            self.sample_width = self._file_h.getsampwidth()

            assert self._file_h.getnchannels() == 1
            assert self._file_h.getsampwidth() == 2
            assert self._file_h.getframerate() == 16000
            assert self._file_h.getcomptype() == 'NONE'

        def read(self, buffer: memoryview) -> int:
            """read callback function"""
            size = buffer.nbytes
            frames = self._file_h.readframes(size // self.sample_width)

            buffer[:len(frames)] = frames

            return len(frames)

        def close(self):
            """close callback function"""
            self._file_h.close()

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # specify the audio format
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16,
            channels=1)

    # setup the audio stream
    callback = WavFileReaderCallback(weatherfilename)
    stream = speechsdk.audio.PullAudioInputStream(callback, wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # instantiate the speech recognizer with pull stream input
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()


def speech_recognition_with_push_stream():
    """gives an example how to use a push audio stream to recognize speech from a custom audio
    source"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    # The number of bytes to push per buffer
    n_bytes = 3200
    wav_fh = wave.open(weatherfilename)

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    # start pushing data until all data has been read from the file
    try:
        while(True):
            frames = wav_fh.readframes(n_bytes // 2)
            print('read {} bytes'.format(len(frames)))
            if not frames:
                break

            stream.write(frames)
            time.sleep(.1)
    finally:
        # stop recognition and clean up
        wav_fh.close()
        stream.close()
        speech_recognizer.stop_continuous_recognition()

def speech_recognize_once_with_auto_language_detection_from_mic():
    """performs one-shot speech recognition from the default microphone with auto language detection"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # create the auto detection language configuration with the potential source language candidates
    auto_detect_source_language_config = \
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["de-DE", "en-US"])
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, auto_detect_source_language_config=auto_detect_source_language_config)
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(result)
        print("Recognized: {} in language {}".format(result.text, auto_detect_source_language_result.language))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

def speech_recognize_with_auto_language_detection_UsingCustomizedModel():
    """performs speech recognition from the audio file with auto language detection, using customized model"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)

    # Replace the languages with your languages in BCP-47 format, e.g. fr-FR.
    # Please see https://docs.microsoft.com/azure/cognitive-services/speech-service/language-support
    # for all supported languages
    en_language_config = speechsdk.languageconfig.SourceLanguageConfig("en-US")
    # Replace the languages with your languages in BCP-47 format, e.g. zh-CN.
    # Set the endpoint ID of your customized mode that will be used for fr-FR.
    # Replace with your own CRIS endpoint ID.
    fr_language_config = speechsdk.languageconfig.SourceLanguageConfig("fr-FR", "myendpointId")
    # create the auto detection language configuration with the source language configurations
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        sourceLanguageConfigs=[en_language_config, fr_language_config])

    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config)
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(result)
        print("Recognized: {} in language {}".format(result.text, auto_detect_source_language_result.language))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def speech_recognize_keyword_locally_from_microphone():
    """runs keyword spotting locally, with direct access to the result audio"""

    # Creates an instance of a keyword recognition model. Update this to
    # point to the location of your keyword recognition model.
    model = speechsdk.KeywordRecognitionModel("r4.table")

    # The phrase your keyword recognition model triggers on.
    keyword = "R4"

    # Create a local keyword recognizer with the default microphone device for input.
    keyword_recognizer = speechsdk.KeywordRecognizer()

    done = False

    def recognized_cb(evt):
        # Only a keyword phrase is recognized. The result cannot be 'NoMatch'
        # and there is no timeout. The recognizer runs until a keyword phrase
        # is detected or recognition is canceled (by stop_recognition_async()
        # or due to the end of an input file or stream).
        result = evt.result
        # if result.reason == speechsdk.ResultReason.RecognizedKeyword:
        #     print("RECOGNIZED KEYWORD: {}".format(result.text))
        try:
            os.system("CLS")
            os.system("COLOR 47")
        except:
            print("Unable to tweak commandline output.")
        nonlocal done
        done = True

    def canceled_cb(evt):
        result = evt.result
        if result.reason == speechsdk.ResultReason.Canceled:
            print('CANCELED: {}'.format(result.cancellation_details.reason))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the keyword recognizer.
    keyword_recognizer.recognized.connect(recognized_cb)
    keyword_recognizer.canceled.connect(canceled_cb)

    # Start keyword recognition.
    result_future = keyword_recognizer.recognize_once_async(model)
    try:
        os.system("CLS")
        os.system("COLOR 17")
    except:
        print("Unable to tweak commandline output.")
    print('Say something starting with "{}" followed by whatever you want...'.format(keyword))
    result = result_future.get()

    # Read result audio (incl. the keyword).
    if result.reason == speechsdk.ResultReason.RecognizedKeyword:
        time.sleep(3) # give some time so the stream is filled
        result_stream = speechsdk.AudioDataStream(result)
        result_stream.detach_input() # stop any more data from input getting to the stream

        save_future = result_stream.save_to_wav_file_async("AudioFromRecognizedKeyword.wav")
        #print('Saving and sending to Watson...')
        saved = save_future.get()
    
    filename = 'AudioFromRecognizedKeyword.wav'
    w = wave.open(filename, 'r')
    rate = w.getframerate()
    frames = w.getnframes()
    buffer = w.readframes(frames)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    lines = dsmodel.stt(data16)
    sayback = str(lines)
    
    # Stores the transcribed text 
    utterance = "" 
    utterance = sayback.replace('are four ','').replace('Are Four ', '').replace('are Four ', '').replace('are 4 ', '').replace('our four ', '').replace('Our Four ', '').replace('our 4 ', '').replace('r 4 ', '').replace('R. for ', '').replace('or for ', '')
    
    handle_input(utterance)

def pronunciation_assessment_from_microphone():
    """pronunciation assessment."""

    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    # Note: The pronunciation assessment feature is currently only available on westus, eastasia and centralindia regions.
    # And this feature is currently only available on en-US language.
    config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    reference_text = ""
    # create pronunciation assessment config, set grading system, granularity and if enable miscue based on your requirement.
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(reference_text=reference_text,
                                                                   grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                                                                   granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                                                                   enable_miscue=True)

    recognizer = speechsdk.SpeechRecognizer(speech_config=config)
    while True:
        # Receives reference text from console input.
        print('Enter reference text you want to assess, or enter empty text to exit.')
        print('> ')

        try:
            reference_text = input()
        except EOFError:
            break

        pronunciation_config.reference_text = reference_text
        pronunciation_config.apply_to(recognizer)

        # Starts recognizing.
        print('Read out "{}" for pronunciation assessment ...'.format(reference_text))

        result = recognizer.recognize_once_async().get()

        # Check the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print('Recognized: {}'.format(result.text))
            print('  Pronunciation Assessment Result:')

            pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
            print('    Accuracy score: {}, Pronunciation score: {}, Completeness score : {}, FluencyScore: {}'.format(
                pronunciation_result.accuracy_score, pronunciation_result.pronunciation_score,
                pronunciation_result.completeness_score, pronunciation_result.fluency_score
            ))
            print('  Word-level details:')
            for idx, word in enumerate(pronunciation_result.words):
                print('    {}: word: {}, accuracy score: {}, error type: {};'.format(
                    idx + 1, word.word, word.accuracy_score, word.error_type
                ))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
