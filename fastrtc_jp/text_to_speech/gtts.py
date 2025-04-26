import asyncio
from typing import Any, AsyncGenerator, Generator, Literal, Protocol
from io import BytesIO

import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass
from fastrtc.text_to_speech.tts import TTSOptions, TTSModel
from fastrtc_jp.text_to_speech.util import split_to_talk_segments
from gtts import gTTS
import av
from av.container import InputContainer
import librosa

def mp3_to_pcm( mp3_data: bytes) -> tuple[int, NDArray[np.float32]]:
    # MP3をPCMに変換
    container:InputContainer = av.open(BytesIO(mp3_data),mode='r')
    audio = container.streams.audio[0]
    sample_rate = audio.rate
    
    samples = []
    for frame in container.decode(audio=0):
        samples.append(frame.to_ndarray().flatten())
    
    pcm_data = np.concatenate(samples)
    first_non_silent = np.nonzero(pcm_data)[0]
    if len(first_non_silent) > 0:
        pcm_data2 = pcm_data[first_non_silent[0]:]
    else:
        pcm_data2 = pcm_data
    pcm_data3 = time_stretch1(pcm_data2, sample_rate, 1.3)
    
    return sample_rate, pcm_data3

def time_stretch1(audio: NDArray[np.float32], sample_rate:int, rate: float) -> NDArray[np.float32]:
    return librosa.effects.time_stretch(audio, rate=rate)

def time_stretch2(audio: NDArray[np.float32], sample_rate:int, rate: float) -> NDArray[np.float32]:
    x = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-8)
    b = sample_rate // rate
    return librosa.resample(x, orig_sr=sample_rate, target_sr=b)

@dataclass
class GTTSOptions(TTSOptions):
    lang: str = "ja"       # 言語
    tld: str = "jp"        # トップレベルドメイン
    speed: float = 1.0     # 速度（現在のgTTSは未対応だが将来のために）

class GTTSModel(TTSModel):
    def __init__(self):
        self._sample_rate = 24000  # gTTSのデフォルトサンプリングレート

    def tts(self, text: str, options: GTTSOptions | None = None) -> tuple[int, NDArray[np.float32]]:
        options = options or GTTSOptions()
        tts = gTTS(text=text, lang=options.lang, tld=options.tld)
        mp3_data = BytesIO()
        tts.write_to_fp(mp3_data)
        mp3_data.seek(0)
        return mp3_to_pcm(mp3_data.read())

    async def stream_tts(self, text: str, options: GTTSOptions | None = None) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or GTTSOptions()
        segments = split_to_talk_segments(text)
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


