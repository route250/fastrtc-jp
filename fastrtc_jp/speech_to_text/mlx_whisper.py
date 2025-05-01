import sys,os
from typing import Any
import mlx_whisper.whisper
import numpy as np
from numpy.typing import NDArray

import mlx_whisper

from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.utils import audio_to_float32, audio_to_int16
from fastrtc_jp.speech_to_text.util import resample_audio
from fastrtc_jp.utils.hf_util import download_hf_hub

    # model_size = 'mlx-community/whisper-tiny-mlx'
    # model_size = 'mlx-community/whisper-tiny-mlx-fp32'
    # model_size = 'mlx-community/whisper-tiny-mlx-q4'
    # model_size = 'mlx-community/whisper-tiny-mlx-8bit'

    # model_size = 'mlx-community/whisper-base-mlx'
    # model_size = 'mlx-community/whisper-base-mlx-fp32'
    # model_size = 'mlx-community/whisper-base-mlx-q4'
    # model_size = 'mlx-community/whisper-base-mlx-8bit'
    # model_size = 'mlx-community/whisper-base-mlx-4bit'
    # model_size = 'mlx-community/whisper-base-mlx-2bit'

    # model_size = 'mlx-community/whisper-small-mlx'
    # model_size = 'mlx-community/whisper-small-mlx-fp32'
    # model_size = 'mlx-community/whisper-small-mlx-q4' # 197M
    # model_size = 'mlx-community/whisper-small-mlx-8bit'
    # model_size = 'mlx-community/whisper-small-mlx-4bit'

    # model_size = 'mlx-community/whisper-medium-mlx' # 1.5GB
    # model_size = 'mlx-community/whisper-medium-mlx-fp32' # 3GB
    # model_size = 'mlx-community/whisper-medium-mlx-q4' # 0.5GB
    # model_size = 'mlx-community/whisper-medium-mlx-8bit' # 865M

    # model_size = 'mlx-community/whisper-large-v3-mlx'
    # mlx-community/whisper-large-v3-turbo # 1.6GB

class MlxWhisper(STTModel):
    """fastrtcのSTTModelを実装したクラス"""
    def __init__(self,*,path_or_hf_repo:str|None=None,**kwargs:Any):
        self.path_or_hf_repo = path_or_hf_repo or 'mlx-community/whisper-medium-mlx-q4'
        self.model_path = download_hf_hub( repo_id=self.path_or_hf_repo )

    # def load_model(self):
    #     pass

    # def check_model(self):
    #     check_audio:AudioF32 = sin_signal()
    #     res:TranscribRes = self.transcrib(check_audio)

    def stt(self, audiodata:tuple[int, NDArray[np.int16 | np.float32]]) ->str:
        sample_rate,audio_streo = audiodata
        print(f"Received audio: sr:{sample_rate} {audio_streo.shape} {audio_streo.dtype}")
        audio_mono:NDArray = audio_to_float32(audio_streo[0])
        audio_mono = resample_audio(sample_rate, audio_mono, 16000)
        transcribe_res = mlx_whisper.transcribe( audio_mono,no_speech_threshold=0.01,
                            language=None, word_timestamps=False,
                            fp16=False, path_or_hf_repo=self.model_path)
        text = transcribe_res.get('text', '')
        print(f"stt: {text}")
        if isinstance(text,str):
            return text
        return ""
