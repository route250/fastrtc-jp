import asyncio
from typing import Any, AsyncGenerator, Generator, Literal, Protocol
from dataclasses import dataclass
from io import BytesIO
import numpy as np
from numpy.typing import NDArray
from fastrtc.text_to_speech.tts import TTSOptions, TTSModel
from fastrtc.utils import audio_to_float32
from fastrtc_jp.text_to_speech.util import split_to_talk_segments
from fastrtc_jp.speech_to_text.util import resample_audio, time_stretch1
from gtts import gTTS
import av
from av.container import InputContainer
from av import logging as av_logging
av_logging.set_level(av_logging.FATAL)  # Set logging level to FATAL to suppress warnings

def mp3_to_pcm( mp3_data: bytes) -> tuple[int, NDArray[np.float32]]:
    # MP3をPCMに変換
    container:InputContainer = av.open(BytesIO(mp3_data),mode='r')
    audio = container.streams.audio[0]
    sample_rate = audio.rate
    
    samples:list[NDArray] = []
    for frame in container.decode(audio=0):
        samples.append(frame.to_ndarray().flatten())
    
    pcm_data:NDArray[np.float32] = audio_to_float32(np.concatenate(samples))
    first_non_silent = np.nonzero(pcm_data)[0]
    if len(first_non_silent) > 0:
        pcm_data2 = pcm_data[first_non_silent[0]:]
    else:
        pcm_data2 = pcm_data
    return sample_rate, pcm_data2

@dataclass
class GTTSOptions(TTSOptions):
    split:bool = False
    lang: str = "ja"       # 言語
    tld: str = "jp"        # トップレベルドメイン
    speed: float = 1.0     # 速度（現在のgTTSは未対応だが将来のために）

class GTTSModel(TTSModel):
    def __init__(self):
        self._sample_rate = 24000

    def tts(self, text: str, options: GTTSOptions | None = None) -> tuple[int, NDArray[np.float32]]:
        options = options or GTTSOptions()
        # tts実行
        tts = gTTS(text=text, lang=options.lang, tld=options.tld)
        # mp3からpcmに変換
        mp3_data = BytesIO()
        tts.write_to_fp(mp3_data)
        mp3_data.seek(0)
        sr,audio = mp3_to_pcm(mp3_data.read())
        # 速度調整
        audio = time_stretch1(audio, sr, round( 1.3 * options.speed, 1))
        # サンプリングレート変換
        audio = resample_audio( sr, audio, self._sample_rate )
        return (self._sample_rate, audio)

    async def stream_tts(self, text: str, options: GTTSOptions | None = None) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or GTTSOptions()
        segments = split_to_talk_segments(text) if options.split else [text]
        for seg in segments:
            sample_rate, audio = self.tts(seg, options)
            yield sample_rate, audio

    def stream_tts_sync(self, text: str, options: GTTSOptions | None = None) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break
