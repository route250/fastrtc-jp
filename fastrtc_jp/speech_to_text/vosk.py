import sys,os
import json
from typing import Any
import numpy as np
from numpy.typing import NDArray
from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.utils import audio_to_float32, audio_to_int16
from fastrtc_jp.speech_to_text.util import resample_audio
import vosk

# 参考までにモデルの保存先
# MODEL_DIRS = [os.getenv("VOSK_MODEL_PATH"), Path("/usr/share/vosk"),
#        Path.home() / "AppData/Local/vosk", Path.home() / ".cache/vosk"]

class VoskSTT(STTModel):
    """fastrtcのSTTModelを実装したクラス"""
    def __init__(self,model_name:str|None=None,lang:str|None=None):
        vosk.SetLogLevel(-1)
        #self.model = vosk.Model(model_name="vosk-model-small-ja-0.22")
        if not model_name and not lang:
            model_name = "vosk-model-ja-0.22"
        self.model = vosk.Model(model_name=model_name,lang=lang)

    def stt(self, audiodata: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sample_rate,audio_streo = audiodata
        print(f"Received audio: sr:{sample_rate} {audio_streo.shape} {audio_streo.dtype}")
        audio_mono = audio_streo[0]
        # if sample_rate != 160000:
        #     audio_mono = audio_to_float32(audio_mono)
        #     audio_mono = resample_audio(sample_rate, audio_mono, 16000)
        audio_mono = audio_to_int16(audio_mono)
        audio_bin = audio_mono.tobytes()
        rec = vosk.KaldiRecognizer(self.model, sample_rate)
        rec.AcceptWaveform(audio_bin)
        result_str = rec.FinalResult()
        result = json.loads(result_str)
        text = result.get('text', '')
        print(f"stt: {text}")
        return text


