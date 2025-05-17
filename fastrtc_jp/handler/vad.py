
import asyncio
from dataclasses import dataclass
import inspect
from typing import Protocol, Type, Callable, Any, Literal, AsyncGenerator
from logging import getLogger

import numpy as np
from numpy.typing import NDArray
import librosa

from fastrtc.utils import audio_to_float32, audio_to_int16

from fastrtc_jp.handler.voice import SttAudio
from fastrtc_jp.speech_to_text.util import resample_audio

logger = getLogger(__name__)

def get_warmupdata(*,sample_rate:int=16000,duration:float=0.4,frequency:float=440.0,ch:int=2) ->tuple[int, NDArray[np.int16 | np.float32]]:
    """音声データを生成する"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    if ch > 1:
        audio_data = np.stack([audio_data] * ch, axis=0)  # 指定されたチャンネル数に複製
    else:
        audio_data = audio_data[np.newaxis, :]  # モノラルの場合、次元を追加
    return sample_rate, audio_data

def audio_to_vad(sampling_rate:int, audio:np.ndarray) ->tuple[int,np.ndarray]:
    try:
        audio_f32 = audio_to_float32(audio)
        sr = 16000
        if sr != sampling_rate:
            audio_f32 = librosa.resample(audio_f32, orig_sr=sampling_rate, target_sr=sr)
    except Exception as ex:
        logger.error(ex)
        print(ex)
        audio_f32 = np.zeros((0),dtype=np.float32)
    return (sr,audio_f32)

@dataclass
class VadOptions:
    """Algorithm options."""
    audio_chunk_duration: float = 0.6
    started_talking_threshold: float = 0.2
    speech_threshold: float = 0.1

class VadHandler:

    logger = getLogger(f"{__name__}.{__qualname__}")
    frame_rate:int = 16000
    def __init__(
            self,
            vad_fn:Callable[[tuple[int,NDArray[np.float32]]],float],
            algo_options: VadOptions|None=None,
        ):
        self.receive_rate = VadHandler.frame_rate
        self.vad_fn:Callable[[tuple[int,NDArray[np.float32]]],float] = vad_fn
        self.algo_options: VadOptions = algo_options or VadOptions()

        self.in_talking:bool = False
        self.rec_count:int = 0
        self.buffer:NDArray[np.int16] = np.zeros( (self.receive_rate*30), dtype=np.int16)
        self.start_idx:int = 0
        self.end_idx:int = 0

        max_duration = 0.3
        self.max_duration_length:int = int( self.receive_rate * max_duration )

    def _reconfigure(self, rate:int):
        """再設定"""
        if rate == self.receive_rate:
            return
        self.logger.debug(f"reconfigure: {self.receive_rate}Hz")
        self.receive_rate = rate
        self.in_talking = False
        self.rec_count = 0
        self.buffer = np.zeros( (self.receive_rate*30), dtype=np.int16)
        self.max_duration_length = int( self.receive_rate * 0.3 )
        self.start_idx = 0
        self.end_idx = 0

    def reset(self):
        self.in_talking = False
        self.rec_count = 0
        self.start_idx = 0
        self.end_idx = 0

    def shutdown(self):
        pass

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) ->SttAudio|None:
        stt_audio:SttAudio|None = None
        #----------------------
        # vadの長さになるまで
        #----------------------
        # frameを分解
        frame_rate, frame_audio = frame
        if frame_rate != VadHandler.frame_rate or frame_audio.shape[0]!=1:
            # 16khz mono 以外なら無視する
            return None
        self._reconfigure(frame_rate)
        frame_audio = np.squeeze(frame_audio) # shapeを変換 (ch,length) -> (length,)
        self.rec_count += len(frame_audio)
        # add to buffer
        end_pos = self.end_idx + len(frame_audio)
        while end_pos > len(self.buffer):
            self.buffer = np.concatenate([self.buffer, np.zeros(self.receive_rate * 10, dtype=np.int16)])
        self.buffer[self.end_idx:end_pos] = frame_audio
        self.end_idx = end_pos

        duration = (self.end_idx-self.start_idx) / self.receive_rate
        if duration < self.algo_options.audio_chunk_duration:
            # 規定の長さ以下
            return None
        
        # vad 判定 (内部で 16Khz float32に変換される)
        vad_frame = audio_to_vad( self.receive_rate, self.buffer[self.start_idx:self.end_idx] )
        dur_vad = self.vad_fn( vad_frame )
        # if dur_vad<0.0 or 1.0<dur_vad:
        #     self.logger.error("Invalid VAD result: %s. VAD result must be between 0 and 1.", dur_vad)
        #     dur_vad = 0.0

        if dur_vad > self.algo_options.started_talking_threshold and not self.in_talking:
            self.in_talking = True
            self.logger.debug(f"<vad> started talking vad:{dur_vad}")

        if not self.in_talking:
            # 無声部分の場合、最後のところだけ残しておく
            over:int = self.end_idx-self.max_duration_length
            if over>0:
                self.buffer[0:self.max_duration_length] = self.buffer[over:self.end_idx]
                self.start_idx = 0
                self.end_idx = self.max_duration_length

        else:
            if dur_vad < self.algo_options.speech_threshold:
                self.logger.debug(f"<vad> stop talking vad:{dur_vad}")
                segment = np.copy(self.buffer[0:self.end_idx]).reshape(1, -1)
                stt_audio = SttAudio(self.rec_count-self.end_idx, self.rec_count, self.receive_rate, segment)
                self.buffer[0:self.max_duration_length] = self.buffer[self.end_idx-self.max_duration_length:self.end_idx]
                self.start_idx = 0
                self.end_idx = self.max_duration_length
                self.in_talking = False

        return stt_audio