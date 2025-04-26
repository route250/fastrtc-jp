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

_EMPTY_DATA = np.zeros((1,), dtype=np.float32)

async def get_availavle_url( hostlist:str, timeout: float = 0.5 ) -> str | None:
    """
    音声合成サーバーのURLを取得する
    :param hostlist: ホスト名のリスト
    :return: 音声合成サーバーのURL
    """
    url_list:list[str] = list(set(hostlist.split(',')))
    if len(url_list)>0:
        async with httpx.AsyncClient(timeout=httpx.Timeout( timeout, connect=timeout, read=timeout)) as client:
            for url in url_list:
                    try:
                        response = await client.get(url)
                        if response.status_code == 200 or response.status_code == 404:
                            return url
                    except (httpx.ConnectError, httpx.TimeoutException):
                        continue
    return None


