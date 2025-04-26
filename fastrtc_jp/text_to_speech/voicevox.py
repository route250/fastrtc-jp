import sys,os
import asyncio
import re
from typing import AsyncGenerator, Generator, Literal, Protocol
import wave
from io import BytesIO
import httpx

import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass
from fastrtc.text_to_speech.tts import TTSOptions, TTSModel

from fastrtc_jp.utils.util import get_availavle_url
from fastrtc_jp.text_to_speech.util import split_to_talk_segments

_EMPTY_DATA = np.zeros((1,), dtype=np.float32)

@dataclass
class VoicevoxTTSOptions(TTSOptions):
    url: str|None = None
    speaker: int = 8 # ひびき
    speedScale: float = 1.1
    pitchOffset: float = 0.0
    lang: str = "ja-jp"

async def voicevox_api( text:str, options:VoicevoxTTSOptions) -> tuple[int, NDArray[np.float32]]:
    try:
        sv_url = options.url
        if sv_url is None:
            print("Voicevox server not found")
            return (0, _EMPTY_DATA)

        xtimeout = httpx.Timeout( 5.0, connect=5.0, read=180.0)
        timeout = (5.0,180.0)
        params = {'text': text, 'speaker': options.speaker, 'timeout': timeout }

        async with  httpx.AsyncClient(timeout=xtimeout) as client:
            # step1: 音声合成のためのテキスト解析
            res1 = await client.post( f'{sv_url}/audio_query', params=params)
            if res1.status_code != 200:
                return (0, _EMPTY_DATA)
            res1_json = res1.json()
            # step2: パラメータ調整
            ss:float = res1_json.get('speedScale',1.0)
            res1_json['speedScale'] = ss * options.speedScale
            ps:float = res1_json.get('pitchScale',0.0)
            res1_json['pitchScale'] = ps + options.pitchOffset
            # step3: 音声合成
            params = {'speaker': options.speaker, 'timeout': timeout }
            headers = {'content-type': 'application/json'}
            res2 = await client.post( f'{sv_url}/synthesis',
                json=res1_json,
                params=params,
                headers=headers
            )
            if res2.status_code != 200:
                return (0, _EMPTY_DATA)
            # step4: 音声データ変換
            with wave.open(BytesIO(res2.content), 'rb') as wf:
                sample_rate = wf.getframerate()
                audio_i16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            audio_f32: NDArray[np.float32] = audio_i16.astype(np.float32) / 32768.0
            # step5: 音声データを返す
            return (sample_rate, audio_f32)
    except Exception as e:
        print(f"atts: {e}")
        return (0, _EMPTY_DATA)

class VoicevoxTTSModel(TTSModel):
    def __init__(self, hostlist:str|None=None):
        self._voicevox_url:str|None = None
        self._voicevox_list:str = hostlist if hostlist else 'http://127.0.0.1:50021'
        self._sample_rate = 48000

    async def _aget_voicevox_url( self ) ->str|None:
        if self._voicevox_url is None:
            self._voicevox_url = await get_availavle_url(self._voicevox_list)
        return self._voicevox_url

    def tts(self, text: str, options:VoicevoxTTSOptions|None=None) -> tuple[int, NDArray[np.float32]]:
            options = options or VoicevoxTTSOptions()
            if options.url is None:
                options.url = asyncio.run( self._aget_voicevox_url() )
            ret = asyncio.run( voicevox_api(text, options) )
            if isinstance(ret, tuple) and ret[0]>0:
                self._sample_rate = ret[0]
            return ret

    async def stream_tts(self, text: str, options: VoicevoxTTSOptions | None = None) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or VoicevoxTTSOptions()
        if options.url is None:
            options.url = await self._aget_voicevox_url()
        segments = split_to_talk_segments(text)
        for seg in segments:
            res = await voicevox_api(seg, options)
            if isinstance(res, tuple) and res[0]>0:
                self._sample_rate = res[0]
            yield res

    def stream_tts_sync( self, text: str, options: VoicevoxTTSOptions | None = None ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()
        # Use the new loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break
