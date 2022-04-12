#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import droid_speech
#import intent_sample
#import translation_sample
#import speech_synthesis_sample

from collections import OrderedDict
import platform

#eofkey = 'Ctrl-Z' if "Windows" == platform.system() else 'Ctrl-D'

droidspeechfunctions = OrderedDict([
    (droid_speech, [
        droid_speech.speech_recognize_once_from_mic,
        droid_speech.speech_recognize_once_from_file,
        droid_speech.speech_recognize_once_compressed_input,
        droid_speech.speech_recognize_once_from_file_with_customized_model,
        droid_speech.speech_recognize_once_from_file_with_custom_endpoint_parameters,
        droid_speech.speech_recognize_async_from_file,
        droid_speech.speech_recognize_continuous_from_file,
        droid_speech.speech_recognition_with_pull_stream,
        droid_speech.speech_recognition_with_push_stream,
        droid_speech.speech_recognize_keyword_from_microphone,
        droid_speech.speech_recognize_keyword_locally_from_microphone,
        droid_speech.pronunciation_assessment_from_microphone,
    ])
])


def select():
    modules = list(droidspeechfunctions.keys())

    try:
        selected_module = modules[0]
    except EOFError:
        raise
    except Exception as e:
        print(e)
        return

    try:
        selected_function = droidspeechfunctions[selected_module][10]
    except EOFError:
        raise
    except Exception as e:
        print(e)
        return

    try:
        selected_function()
    except Exception as e:
        print('Error running droid funtion: {}'.format(e))

    print()


while True:
    try:
        select()
    except EOFError:
        break
