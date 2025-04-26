import sys,os
from typing import Any
from fastrtc.speech_to_text.stt_ import STTModel
import numpy as np
from numpy.typing import NDArray
import vosk
import json

class VoskSTT(STTModel):
    """fastrtcのSTTModelを実装したクラス"""
    def __init__(self):
        vosk.SetLogLevel(-1)
        #self.model = vosk.Model(model_name="vosk-model-small-ja-0.22")
        self.model = vosk.Model(model_name="vosk-model-ja-0.22")

    def stt(self, audiodata: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sample_rate,audio_streo = audiodata
        print(f"Received audio: sr:{sample_rate} {audio_streo.shape} {audio_streo.dtype}")
        audio_mono:NDArray = audio_streo[0]
        # audiof = audio_stereo[0].astype(np.float32) / 32768.0
        audio_bin = audio_mono.tobytes() # struct.pack('<' + 'h'*len(audio_mono), *audio_mono)

        rec = vosk.KaldiRecognizer(self.model, sample_rate)
        rec.AcceptWaveform(audio_bin)
        result_str = rec.FinalResult()
        result = json.loads(result_str)
        text = result.get('text', '')
        print(f"stt: {text}")
        return text


