import sys,os
from pathlib import Path
import asyncio
from typing import Any, AsyncGenerator, Generator, Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fastrtc.text_to_speech.tts import TTSOptions, TTSModel

from fastrtc_jp.text_to_speech.util import split_to_talk_segments

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel as SBV2_TTSModel

# style-vert-vits2のログを設定
import loguru
loguru.logger.remove()  # 既存のログ設定を削除
loguru.logger.add(sys.stderr, level="ERROR")  # ERRORレベルのログのみを表示

@dataclass
class StyleBertVits2Options(TTSOptions):
    model_path: Path
    config_path: Union[Path, HyperParameters]
    style_vec_path: Union[Path, NDArray[Any]]
    device: str = "cpu"
    split:bool = False
    speedScale: float = 1.1
    pitchOffset: float = 0.0
    lang: str = "ja-jp"

class StyleBertVits2(TTSModel):
    def __init__(self):
        self.model:SBV2_TTSModel|None = None
        pass

    async def _load(self,options:StyleBertVits2Options) -> SBV2_TTSModel:
        if self.model is None:
            # Bertモデルをロード
 
            bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
            bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

            self.model = SBV2_TTSModel( device=options.device,
                model_path= options.model_path,
                config_path= options.config_path,
                style_vec_path= options.style_vec_path,
            )
            print("load-2")
            self.model.load()
            print("load-9")
        return self.model

    async def _run(self, text:str, options:StyleBertVits2Options) -> tuple[int, NDArray[np.float32]]:
        model:SBV2_TTSModel = await self._load(options)
        frame = model.infer(text)
        return frame

    def tts(self, text: str, options:StyleBertVits2Options) -> tuple[int, NDArray[np.float32]]:
        ret = asyncio.run( self._run(text, options) )
        if isinstance(ret, tuple) and ret[0]>0:
            self._sample_rate = ret[0]
        return ret

    async def stream_tts(self, text: str, options: StyleBertVits2Options) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        segments = split_to_talk_segments(text) if options.split else [text]
        for seg in segments:
            res = await self._run(seg, options)
            if isinstance(res, tuple) and res[0]>0:
                self._sample_rate = res[0]
            yield res

    def stream_tts_sync( self, text: str, options: StyleBertVits2Options ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()
        # Use the new loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break
