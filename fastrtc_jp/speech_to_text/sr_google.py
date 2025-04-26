import sys,os
from typing import Any
from fastrtc.speech_to_text.stt_ import STTModel
import numpy as np
from numpy.typing import NDArray
import speech_recognition as sr

class GoogleSTT(STTModel):
    """GoogleのSTTModelを実装したクラス"""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_lang_index = 0
        self._langs = ['ja-JP', 'en-US']

    def stt(self, audiodata: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sample_rate,audio_streo = audiodata
        print(f"Received audio: sr:{sample_rate} {audio_streo.shape} {audio_streo.dtype}")
        audio_mono:NDArray = audio_streo[0]
        # audiof = audio_stereo[0].astype(np.float32) / 32768.0
        # Convert mono audio to AudioData object for speech_recognition
        if audio_mono.dtype == np.float32:
            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
            audio_mono = (audio_mono * 32767).astype(np.int16)
        elif audio_mono.dtype != np.int16:
            # Handle any other data type
            audio_mono = audio_mono.astype(np.int16)
        audio_bin = sr.AudioData( audio_mono.tobytes(), sample_rate, 2 )

        lang_index = self.last_lang_index
        for i in range(len(self._langs)):
            try:
                text = self.recognizer.recognize_google(audio_bin, language=self._langs[lang_index])
                self.last_lang_index = lang_index
                print(f"stt: {text}")
                return str(text)
            except sr.UnknownValueError:
                print(f"Google Speech Recognition could not understand the audio {self._langs[lang_index]}")
            lang_index = (lang_index+1) % 2
        return ""

