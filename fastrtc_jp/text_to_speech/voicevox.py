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
import json
import pathlib
from functools import lru_cache

_EMPTY_DATA = np.zeros((1,), dtype=np.float32)


@dataclass
class VoicevoxCharacterInfo:
    """Character information for Voicevox"""
    id: int
    name: str
    key: str
    rubyName: str
    voiceFeature: str
    description: str
    labelInfos: list[dict]
    policyUrl: str
    detailUrl: str
    me: list[str]
    you: list[str]
    callNameInfo: dict
    samples: list[str]


@dataclass
class VoicevoxStyleInfo:
    """Style information for Voicevox"""
    speaker_id: int
    speaker_uuid: str
    speaker_name: str
    speaker_style: str
    char_info: VoicevoxCharacterInfo|None = None


@dataclass
class VoicevoxTTSOptions(TTSOptions):
    split:bool = False
    url: str|None = None
    speaker_name: str|None = None
    speaker_uuid: str|None = None
    speaker_style: str|None = None
    speaker_id: int = 8 # ひびき
    speedScale: float = 1.1
    pitchOffset: float = 0.0
    lang: str = "ja-jp"


@lru_cache(maxsize=1)
def load_voicevox_charinfo() -> dict[str,VoicevoxCharacterInfo]:
    """Load character information from voicevox_charinfo.json"""

    json_path = pathlib.Path(__file__).parent / "voicevox_charinfo.json"

    try:
        with open(json_path, encoding='utf-8') as f:
            charinfo_json = json.load(f)
        ret = {}
        for info in charinfo_json:
            impl = VoicevoxCharacterInfo(**{k: v for k, v in info.items() if k in VoicevoxCharacterInfo.__annotations__})
            if impl.key:
                ret[impl.key] = impl
            if impl.name:
                ret[impl.name] = impl
            if impl.id:
                ret[impl.id] = impl
        return ret
    except Exception as e:
        print(f"Failed to load voicevox_charinfo.json: {e}")
        return {}


async def voicevox_api_get(url: str, relpath: str, params: dict | None = None) -> dict|list|None:
    """
    Fetch data from the Voicevox API with the given relative path and parameters.

    Args:
        options (VoicevoxTTSOptions): Options containing the Voicevox server URL.
        relpath (str): The relative path to the API endpoint.
        params (dict | None): Query parameters for the API request.

    Returns:
        dict | None: The JSON response from the API, or None if an error occurs.
    """
    #url = options.url
    if not url:
        print("Voicevox server URL is not provided.")
        return None

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=5.0, read=180.0)) as client:
            response = await client.get(f"{url}/{relpath}", params=params)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        print(f"Request error while accessing {relpath}: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code} while accessing {relpath}: {e.response.text}")
    except Exception as e:
        print(f"Unexpected error while accessing {relpath}: {e}")

    return None


async def load_voicevox_style_dict(url:str) ->dict[str,VoicevoxStyleInfo]:
    try:
        infos = await voicevox_api_get(url, 'speakers')
        if not isinstance(infos,list):
            print("Failed to get voicevox speakers")
        else:
            cdict = load_voicevox_charinfo()
            ret: dict[str,VoicevoxStyleInfo] = {}
            for style_info in infos:
                speaker_name = style_info.get('name')
                char_info = cdict.get(speaker_name)
                speaker_uuid = style_info.get('speaker_uuid')
                styles = style_info.get('styles',[])
                for style in styles:
                    speaker_id = style.get('id')
                    speaker_style = style.get('name')
                    style_info = VoicevoxStyleInfo(
                        speaker_id=speaker_id,
                        speaker_uuid=speaker_uuid,
                        speaker_name=speaker_name,
                        speaker_style=speaker_style,
                        char_info=char_info
                    )
                    ret[str(speaker_id)] = style_info
                    ret[f"{speaker_name}||{speaker_style}"] = style_info
                    if speaker_style == 'ノーマル':
                        ret[speaker_name] = style_info
            return ret
    except Exception as ex:
        raise ex
    return {}


async def voicevox_api_synthesis( text:str, options:VoicevoxTTSOptions) -> tuple[int, NDArray[np.float32]]:
    try:
        sv_url = options.url
        if sv_url is None:
            print("Voicevox server not found")
            return (0, _EMPTY_DATA)

        xtimeout = httpx.Timeout( 5.0, connect=5.0, read=180.0)
        timeout = (5.0,180.0)
        params = {'text': text, 'speaker': options.speaker_id, 'timeout': timeout }

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
            params = {'speaker': options.speaker_id, 'timeout': timeout }
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


    async def update_options( self, options: VoicevoxTTSOptions|None ) ->VoicevoxTTSOptions:
        options = options or VoicevoxTTSOptions()
        if options.url is None:
            options.url = await self._aget_voicevox_url()
        sv_url = options.url
        if sv_url is None:
            print("Voicevox server not found")
            return options
        if not options.speaker_name or not options.speaker_style:
            if options.speaker_id:
                style_dict = await load_voicevox_style_dict(sv_url)
                style_info = style_dict.get(str(options.speaker_id))
                if style_info:
                    options.speaker_name = style_info.speaker_name
                    options.speaker_style = style_info.speaker_style
                    options.speaker_uuid = style_info.speaker_uuid
        elif not options.speaker_id:
            if options.speaker_name:
                if options.speaker_style:
                    key = f"{options.speaker_name}||{options.speaker_style}"
                else:
                    key = options.speaker_name
                style_dict = await load_voicevox_style_dict(sv_url)
                style_info = style_dict.get(str(key))
                if style_info:
                    options.speaker_id = style_info.speaker_id
                    options.speaker_style = style_info.speaker_style
                    options.speaker_uuid = style_info.speaker_uuid
        return options


    def tts(self, text: str, options:VoicevoxTTSOptions|None=None) -> tuple[int, NDArray[np.float32]]:
            options = asyncio.run( self.update_options(options) )
            ret = asyncio.run( voicevox_api_synthesis(text, options) )
            if isinstance(ret, tuple) and ret[0]>0:
                self._sample_rate = ret[0]
            return ret


    async def stream_tts(self, text: str, options: VoicevoxTTSOptions | None = None) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = await self.update_options(options)
        segments = split_to_talk_segments(text) if options.split else [text]
        for seg in segments:
            res = await voicevox_api_synthesis(seg, options)
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
