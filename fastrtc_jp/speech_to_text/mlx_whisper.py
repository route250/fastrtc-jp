import sys,os
from typing import Any
import mlx_whisper.whisper
import numpy as np
from numpy.typing import NDArray

import mlx_whisper
from mlx_whisper.tokenizer import LANGUAGES
from mlx_whisper.decoding import DecodingOptions

from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.utils import audio_to_float32, audio_to_int16
from fastrtc_jp.speech_to_text.util import resample_audio
from fastrtc_jp.utils.hf_util import download_hf_hub
import zlib

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
    def __init__(self,*,
            path_or_hf_repo:str|None=None,
            initial_prompt:str|None=None,
            condition_on_previous_text:bool=False,
            no_speech_threshold:float=0.01,
            compression_ratio_threshold:float=2.4,
            language:str|None=None,temperature:float=0.0,
            **kwargs:Any):
        if language is not None and language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES.keys()}")
        self.path_or_hf_repo = path_or_hf_repo or 'mlx-community/whisper-medium-mlx-q4'
        self.model_path = download_hf_hub( repo_id=self.path_or_hf_repo )
        self.condition_on_previous_text:bool = condition_on_previous_text
        self.no_speech_threshold:float = no_speech_threshold
        self.compression_ratio_threshold:float = compression_ratio_threshold
        self.initial_prompt:str|None = initial_prompt
        self.language:str|None = language
        self.temperature:float = temperature
    # def load_model(self):
    #     pass

    # def check_model(self):
    #     check_audio:AudioF32 = sin_signal()
    #     res:TranscribRes = self.transcrib(check_audio)

    def set_initial_prompt(self, prompt:str|None):
        """初期プロンプトを設定する"""
        self.initial_prompt = prompt

    def stt(self, audiodata:tuple[int, NDArray[np.int16 | np.float32]]) ->str:
        sample_rate,audio_streo = audiodata
        # print(f"Received audio: sr:{sample_rate} {audio_streo.shape} {audio_streo.dtype}")
        audio_mono:NDArray = audio_to_float32(audio_streo[0])
        audio_mono = resample_audio(sample_rate, audio_mono, 16000)
        transcribe_res = mlx_whisper.transcribe(
            audio_mono,
            path_or_hf_repo=self.model_path,
            language=self.language,
            temperature=self.temperature,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=self.initial_prompt,
            no_speech_threshold=self.no_speech_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
            word_timestamps=False,
            fp16=False,
        )
        text = transcribe_res.get('text', '')
        # print(f"stt: {text}")
        if not isinstance(text,str) or len(text) == 0:
            return ""
        # encoded_text = text.encode()
        # compressed = zlib.compress(encoded_text)
        # compression_ratio = len(encoded_text) / len(compressed)
        # if compression_ratio > 5:  # Very high compression ratio indicates repetitive text
        #     print(f"High compression ratio: {compression_ratio}")
        #     return ""
        words = ' '.join(text.split())
        unique_words = set(words)
        words_rate = len(unique_words)/len(words)
        if words_rate <0.3:
            print(f"unique words count: {words_rate} {len(unique_words)}/{len(words)}")
            return ""
        return text
