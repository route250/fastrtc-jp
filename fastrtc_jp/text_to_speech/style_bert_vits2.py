import sys,os
from pathlib import Path
import asyncio
from typing import Any, AsyncGenerator, Generator, Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fastrtc.text_to_speech.tts import TTSOptions, TTSModel

from fastrtc_jp.text_to_speech.util import split_to_talk_segments

from style_bert_vits2.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages, DEFAULT_STYLE
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel as SBV2_TTSModel

from fastrtc_jp.utils.hf_util import download_hf_hub

# style-vert-vits2のログを設定
import loguru
loguru.logger.remove()  # 既存のログ設定を削除
loguru.logger.add(sys.stderr, level="ERROR")  # ERRORレベルのログのみを表示

# 言語ごとのデフォルトの BERT トークナイザーのhugginfaceのパス
# .cache/huggingface/hub/に保存されるはず
SBV2_TOKENIZER_PATHS = {
    Languages.JP: f"ku-nlp/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.JP].name}",
    Languages.EN: f"microsoft/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.EN].name}",
    Languages.ZH: f"hfl/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.ZH].name}",
}
SBV2_MODELS = {
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
        'language': 'jp',
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
        'language': 'jp',
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
        'language': 'jp',
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
        'language': 'jp',
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
        'language': 'jp',
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
        'language': 'jp',
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
        'language': 'jp',
    },
    'AbeShinzo': {
        'model':{
            'repo_id': 'AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
            'path': 'AbeShinzo20240210_e300_s43800.safetensors'
            },
        'config': {
            'repo_id': 'AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
            'path': 'style_vectors.npy'
            },
        'language': 'jp',
    },
    'sakura-miko': {
        'model':{
            'repo_id': 'Lycoris53/style-bert-vits2-sakura-miko',
            'path': 'sakuramiko_e89_s23000.safetensors'
            },
        'config': {
            'repo_id': 'Lycoris53/style-bert-vits2-sakura-miko',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'Lycoris53/style-bert-vits2-sakura-miko',
            'path': 'style_vectors.npy'
            },
        'language': 'jp',
    }
}

def download_sbv2_model(arg) -> Path:
    if isinstance(arg, dict):
        return Path(download_hf_hub(**arg))
    else:
        return Path(arg)

def to_language(lang: str|None) -> Languages:
    if lang and "en" in lang.lower():
        return Languages.EN
    elif lang and "zh" in lang.lower():
        return Languages.ZH
    else:
        return Languages.JP

@dataclass
class StyleBertVits2Options(TTSOptions):
    speaker_id: int|None = None
    speaker_style: str|None = None
    speaker_name: str|None = None
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

            model_path = options.model_path if options else None
            config_path = options.config_path if options else None
            style_vec_path = options.style_vec_path if options else None
            language = options.lang if options else "ja-jp"
            #
            if options and options.model_path and options.config_path and options.style_vec_path:
                model=options.model or options.model_path
                model_path = options.model_path
                config_path = options.config_path
                style_vec_path = options.style_vec_path
                language = to_language(options.lang)
            else:
                model = str(options.model) if options and str(options.model) in SBV2_MODELS else "girl"
                model_dict = SBV2_MODELS[model]
                model_path = model_dict.get('model')
                config_path = model_dict.get('config')
                style_vec_path = model_dict.get('style_vec')
                language = to_language(model_dict.get('language'))
            device = options.device if options and options.device else "cpu"

            bert_models.load_model(language, SBV2_TOKENIZER_PATHS[language])
            bert_models.load_tokenizer(language, SBV2_TOKENIZER_PATHS[language])

            a_model_path = download_sbv2_model(model_path)
            a_config_path = download_sbv2_model(config_path)
            a_style_vec_path = download_sbv2_model(style_vec_path)
            #
            self.model = SBV2_TTSModel( device=device,
                model_path= a_model_path,
                config_path= a_config_path,
                style_vec_path= a_style_vec_path,
            )
            self.model.load()
        return self.model

    async def _run(self, text:str, options:StyleBertVits2Options|None=None) -> tuple[int, NDArray[np.float32]]:
        model:SBV2_TTSModel = await self._load(options)

        speaker_id:int = 0 if 0 in model.id2spk else list(model.id2spk.keys())[0]
        speaker_style:str = DEFAULT_STYLE if DEFAULT_STYLE in model.style2id else list(model.style2id.keys())[0]
        if options:
            if options.speaker_id in model.id2spk:
                speaker_id = options.speaker_id
                options.speaker_name = model.id2spk[speaker_id]
            elif options.speaker_name in model.spk2id:
                speaker_id = model.spk2id[options.speaker_name]
                options.speaker_id = speaker_id
            else:
                options.speaker_id = speaker_id
                options.speaker_name = model.id2spk[speaker_id]

            if options.speaker_style in model.style2id:
                speaker_style = options.speaker_style
            else:
                options.speaker_style = speaker_style

        frame = model.infer(text,speaker_id=speaker_id,style=speaker_style)
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
