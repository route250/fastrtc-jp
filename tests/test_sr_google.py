import sys,os,asyncio
sys.path.insert(0,'./')
from typing import Any
import numpy as np
import speech_recognition as sr

import wave
import importlib.resources as resources
from fastrtc_jp.speech_to_text.sr_google import GoogleSTT
from fastrtc_jp.text_to_speech.gtts import GTTSModel

def test_recognize_google():

    test_wave_file = resources.files('fastrtc.speech_to_text').joinpath('test_file.wav')
    print(f"test {test_wave_file}")
    with wave.open(str(test_wave_file), 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_i16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audio_data = sr.AudioData( audio_i16.tobytes(), sample_rate, 2 )
    r = sr.Recognizer()
    stt_result = r.recognize_google(audio_data, language='en-US')
    print(f"stt: {stt_result}")

def test_english():
    test_wave_file = resources.files('fastrtc.speech_to_text').joinpath('test_file.wav')
    print(f"test {test_wave_file}")
    with wave.open(str(test_wave_file), 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_i16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    stt = GoogleSTT()
    audio_in = (sample_rate, audio_i16.reshape(1,-1))
    stt_result = stt.stt(audio_in)
    print(f"stt: {stt_result}")

def test_japanese():
    test_wave_file = 'tests/testData/voice_mosimosi.wav'
    print(f"test {test_wave_file}")
    with wave.open(str(test_wave_file), 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_i16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    stt = GoogleSTT()
    audio_in = (sample_rate, audio_i16.reshape(1,-1))
    stt_result = stt.stt(audio_in)
    print(f"stt: {stt_result}")

def test_with_gtts():
    tts = GTTSModel()
    tts_text = "こんにちは、今日も良い天気ですね。"
    sample_rate,audio = tts.tts(tts_text)
    stt = GoogleSTT()
    stt_text = stt.stt((sample_rate,audio.reshape(1,-1)))
    print(f"stt: {stt_text}")

if __name__ == "__main__":
    #test1()
    test_english()
    test_japanese()