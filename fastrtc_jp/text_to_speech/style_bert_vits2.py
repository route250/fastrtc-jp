import sys,os
from pathlib import Path
import asyncio
from typing import Any, AsyncGenerator, Generator, Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fastrtc.text_to_speech.tts import TTSOptions, TTSModel

from fastrtc_jp.text_to_speech.util import split_to_talk_segments

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError
from style_bert_vits2.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel as SBV2_TTSModel

# style-vert-vits2のログを設定
import loguru
loguru.logger.remove()  # 既存のログ設定を削除
loguru.logger.add(sys.stderr, level="ERROR")  # ERRORレベルのログのみを表示

# 言語ごとのデフォルトの BERT トークナイザーのhugginfaceのパス
# .cache/huggingface/hub/に保存されるはず
HF_TOKENIZER_PATHS = {
    Languages.JP: f"ku-nlp/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.JP].name}",
    Languages.EN: f"microsoft/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.EN].name}",
    Languages.ZH: f"hfl/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.ZH].name}",
}
MODELS = {
    'jvnv-F1-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F1-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F1-jp/style_vectors.npy'
            },
    },
    'jvnv-F2-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F2-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F2-jp/style_vectors.npy'
            },
    },
    'jvnv-M1-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M1-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M1-jp/style_vectors.npy'
            },
    },
    'jvnv-M2-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M2-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M2-jp/style_vectors.npy'
            },
    },
    'rinne': {
        'model':{
            'repo_id':'RinneAi/Rinne_Style-Bert-VITS2',
            'path':'model_assets/Rinne/Rinne.safetensors'
            },
        'config': {
            'repo_id':'RinneAi/Rinne_Style-Bert-VITS2',
            'path': 'model_assets/Rinne/config.json'
            },
        'style_vec': {
            'repo_id':'RinneAi/Rinne_Style-Bert-VITS2',
            'path': 'model_assets/Rinne/style_vectors.npy'
            },
    },
    'girl': {
        'model':{
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'NotAnimeJPManySpeaker_e120_s22200.safetensors'
            },
        'config': {
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'style_vectors.npy'
            },
    },
    'tsukuyomi-chan': {
        'model':{
            'repo_id': 'ayousanz/tsukuyomi-chan-style-bert-vits2-model',
            'path': 'tsukuyomi-chan_e116_s3000.safetensors'
            },
        'config': {
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'style_vectors.npy'
            },
    }
}

def download_hf_hub(repo_id: str, path: str, *, subfolder:str|None=None, cache_dir: str|Path|None=None) -> Path:
    # Hugging Face Hubからモデルをダウンロード
    for b in (True, False):
        try:
            model_path = hf_hub_download(
                repo_id=repo_id, filename=path,
                subfolder=subfolder, cache_dir=cache_dir,
                local_files_only=b,
            )
            return Path(model_path)
        except LocalEntryNotFoundError as e:
            pass
        except Exception as e:
            raise e
    raise RepositoryNotFoundError(f"{repo_id} {subfolder} {path} not found")

def load_model(arg) -> Path:
    if isinstance(arg, dict):
        return download_hf_hub(**arg)
    else:
        return Path(arg)

@dataclass
class StyleBertVits2Options(TTSOptions):
    model:int|str|None = None
    model_path: str|Path|None = None
    config_path: str|Path|None = None
    style_vec_path: str|Path|None = None
    device: str = "cpu"
    split:bool = False
    speedScale: float = 1.1
    pitchOffset: float = 0.0
    lang: str = "ja-jp"

class StyleBertVits2(TTSModel):
    def __init__(self):
        self.model:SBV2_TTSModel|None = None
        pass

    async def _load(self,options:StyleBertVits2Options|None=None) -> SBV2_TTSModel:
        if self.model is None:
            # Bertモデルをロード
            if options and options.lang and "en" in options.lang.lower():
                language = Languages.EN
            elif options and options.lang and "zh" in options.lang.lower():
                language = Languages.ZH
            else:
                language = Languages.JP
            bert_models.load_model(language, HF_TOKENIZER_PATHS[language])
            bert_models.load_tokenizer(language, HF_TOKENIZER_PATHS[language])

            device = options.device if options else "cpu"
            model_path = options.model_path if options else None
            config_path = options.config_path if options else None
            style_vec_path = options.style_vec_path if options else None
            #
            if model_path is None and config_path is None and style_vec_path is None:
                if options and isinstance(options.model,str):
                    model_dict = MODELS.get(options.model)
                else:
                    model_dict = MODELS.get('jvnv-M1-jp')
                if model_dict:
                    model_path = model_dict.get('model')
                    config_path = model_dict.get('config')
                    style_vec_path = model_dict.get('style_vec')
            #
            a_model_path = load_model(model_path)
            a_config_path = load_model(config_path)
            a_style_vec_path = load_model(style_vec_path)
            #
            self.model = SBV2_TTSModel( device=device,
                model_path= a_model_path,
                config_path= a_config_path,
                style_vec_path= a_style_vec_path,
            )
            print("load-2")
            self.model.load()
            print("load-9")
        return self.model

    async def _run(self, text:str, options:StyleBertVits2Options|None=None) -> tuple[int, NDArray[np.float32]]:
        model:SBV2_TTSModel = await self._load(options)
        frame = model.infer(text)
        return frame

    def tts(self, text: str, options:StyleBertVits2Options|None=None) -> tuple[int, NDArray[np.float32]]:
        ret = asyncio.run( self._run(text, options) )
        if isinstance(ret, tuple) and ret[0]>0:
            self._sample_rate = ret[0]
        return ret

    async def stream_tts(self, text: str, options: StyleBertVits2Options|None) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        segments = split_to_talk_segments(text) if options and options.split else [text]
        for seg in segments:
            res = await self._run(seg, options)
            if isinstance(res, tuple) and res[0]>0:
                self._sample_rate = res[0]
            yield res

    def stream_tts_sync( self, text: str, options: StyleBertVits2Options|None ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()
        # Use the new loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break
