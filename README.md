[Japanese](README.md)/[English](README_en.md)

# fastrtc-jp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/fastrtc-jp.svg)](https://pypi.org/project/fastrtc-jp/)

fastrtc用の日本語TTSとSTT追加キット（日本語音声合成・認識モジュール）

## 概要

[fastrtc](https://fastrtc.org/)は高性能なリアルタイム通信フレームワークですが、現状では日本語の音声合成(TTS)および音声認識(STT)機能が不十分です。このプロジェクトは、fastrtcに日本語対応の音声処理機能を追加するための拡張パッケージです。

主な機能：
- 日本語に特化した音声合成モデル（VOICEVOX、Style-Bert-VITS2, gTTS）
- 日本語に対応した音声認識モデル（MlxWhisper, Vosk、Google Speech Recognition）

fastrtcの詳細な使い方については、[fastrtc公式ドキュメント](https://fastrtc.org/)を参照してください。

## 提供モデル一覧

| クラス | 説明 | 追加パッケージ | 特徴 |
|-------|------|--------------|------|
| VoicevoxTTSModel | VOICEVOXのapiで音声合成するクラス | - | 高品質な日本語音声、多様な話者、感情表現が可能 |
| StyleBertVits2 | Style-Bert-VITS2で音声合成するクラス | style-bert-vits2 / pyopenjtalk | 高品質な音声合成が可能、事前学習済みモデルに対応 |
| GTTSModel | Google Text-to-Speechで音声合成するクラス | gtts | インターネット接続が必要、自然な発話 |
| VoskSTT | Voskエンジンで音声認識するクラス | vosk | オフライン動作可能、軽量 |
| MlxWhisper | mlx-whisperで音声認識するクラス | mlx-whisper | 高精度な音声認識、Apple Silicon最適化 |
| GoogleSTT | SpeechRecognizerのGoogleエンジンで音声認識するクラス | SpeechRecognition==3.10.0 | 高精度、インターネット接続が必要 |

## システム要件

- Python 3.12以上
- Ubuntu 24.04 LTS, macOS Sonoma 15.4 にて動作確認
- 各モデルの追加要件は以下の「インストール」セクションを参照

## インストール

基本的な使い方として、Python仮想環境を作成してからpipでインストールすることをお勧めします。

```bash
# 仮想環境の作成と有効化（オプション）
python -m venv .venv
source .venv/bin/activate
# pipをアップデート
.venv/bin/python3 -m pip install -U pip setuptools
# 基本パッケージのインストール
pip install fastrtc-jp
```

### 追加モジュール

必要に応じて、以下の追加パッケージをインストールしてください：

#### 音声合成（TTS）

##### VOICEVOXを使用する場合:

[VOICEVOX](https://voicevox.hiroshiba.jp/)は、無料で使える高品質な日本語音声合成エンジンです。
VOICEVOXと音声モデルの使用条件については公式ドキュメントなどで確認して下さい。
[VOICEVOX公式](https://voicevox.hiroshiba.jp/)を参照し、APIでアクセスできる環境を準備してください。

- VoicevoxTTSModelクラス

環境変数VOICEVOX_HOSTLISTにVOICEVOXサーバのアドレスとポートを設定して下さい。
.envファイルもしくはconfig.envファイルに以下のように記述して下さい。
記述がない場合は下記のデフォルトが設定されます。

```text:config.env
VOICEVOX_HOSTLIST=http://127.0.0.1:50021
```

- VoicevoxTTSOptionsクラス
  - `speaker_name`: キャラクターの名前
  - `speaker_style`: 声のスタイル
  - `speaker_id`: 話者ID（整数）
  - `speedScale`: 話速スケール（デフォルト: 1.0）

  [http://127.0.0.1:50021/speakers](http://127.0.0.1:50021/speakers)などで、キャラクターの名前、スタイル、idを調べて、話者IDもしくは、名前とスタイルを設定して下さい。

##### Style-Bert-VITS2を使用する場合:

[Style-Bert-Vits2](https://github.com/litagin02/Style-Bert-VITS2)は、高品質な音声合成が可能なTTSモデルです。事前学習済みモデルを使って、自然な日本語音声を生成できます。
Style-Bert-Vits2と音声モデルの使用条件については公式ドキュメントなどで確認して下さい。
詳細は[style-bert-vits2リポジトリ](https://github.com/litagin02/Style-Bert-VITS2)を参照して下さい。
pythonのstyle-bert-vits2パッケージをpythonコードから利用するには、APIサーバを起動してAPIをコールする方法と、直接実行する方式があります。このパッケージでは、直接実行する方式を実装しています。

```bash
pip install fastrtc-jp[sbv2]
```

- StyleBertVits2クラス

標準音声モデルもしくは、別途、音声モデルを使用することができます。

- StyleBertVits2Optionsクラス

  - `model`: モデルプリセット名
  - `model_path`: モデルファイルのパス
  - `config_path`: 設定ファイルのパス
  - `style_vec_path`: スタイルベクトルファイルのパス
  - `device`: 使用するデバイス（デフォルト: "cpu"）
  - `speaker_id`: 話者ID (各モデルのconfig.jsonで確認して下さい)
  - `speaker_style`: スタイル(各モデルのconfig.jsonで確認して下さい)

  モデルプリセット名には以下のプリセット名を指定できます。プリセットを設定する場合、model_path,config_path,style_vec_pathは設定不要です。プリセット以外のモデルは、別途ダウンロードして、model_path,config_path,style_vec_pathを指定して下さい。
  モデルの利用条件は、配布元にて確認して下さい。
  
  |プリセット名|配布元|
  |---|---|
  |jvn-F1-jp|https://huggingface.co/litagin/style_bert_vits2_jvnv|
  |jvn-F2-jp|https://huggingface.co/litagin/style_bert_vits2_jvnv|
  |jvn-M1-jp|https://huggingface.co/litagin/style_bert_vits2_jvnv|
  |jvn-M2-jp|https://huggingface.co/litagin/style_bert_vits2_jvnv|
  |rinne|https://huggingface.co/RinneAi/Rinne_Style-Bert-VITS2|
  |girl|https://huggingface.co/Mofa-Xingche/girl-style-bert-vits2-JPExtra-models|
  |tsukuyomi-chan|https://huggingface.co/ayousanz/tsukuyomi-chan-style-bert-vits2-model|
  |AbeShinzo|https://huggingface.co/AbeShinzo0708/AbeShinzo_Style_Bert_VITS2|
  |sakura-miko|https://huggingface.co/Lycoris53/style-bert-vits2-sakura-miko|

   それぞれのファイルは、初回にダウンロードされます。ダウンロード先は、$HOME/.cache/huggingface/hubです。

```python:参考
StyleBertVits2Options( device="cpu",
    model_file = "model/jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors",
    config_file = "model/jvnv-M1-jp/config.json",
    style_file = "model/jvnv-M1-jp/style_vectors.npy",
)
```

##### GTTSModelを使用する場合:

詳細は[gTTSリポジトリ](https://github.com/pndurette/gTTS)を参照して下さい。
gTTSはgoogleのapiを使用しますので、インターネット接続が必要です。

```bash
pip install fastrtc-jp[gtts]
```

- GTTSModelクラス

Google Text-to-Speechを使用した音声合成モデルです。

- GTTSOptionsクラス

  - `speed`: 話す速度(デフォルト1.0)


#### 音声認識（STT）

##### VoskSTTを使用する場合:

[Vosk](https://alphacephei.com/vosk/)は、オフラインで動作する音声認識エンジンです。
詳細は[Voskの公式](https://alphacephei.com/vosk)を参照してください。
一応、日本語のモデルを自動でダウンロードするようにしています。ダウンロード先は、$HOME/.cache/voskです。

```bash
pip install fastrtc-jp[vosk]
```
- VoskSTTクラス

現在は、'vosk-model-ja-0.22'に固定です。

##### mlx-whisperを使用する場合:

[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)はApple Silicon向けに最適化されたWhisperモデルの実装です。
詳細は、[MLX Examplesリポジトリ](https://github.com/ml-explore/mlx-examples/tree/main/whisper)を参照してください。

```bash
pip install fastrtc-jp[mlx]
```

- MlxWhisperクラス

現在は、'mlx-community/whisper-medium-mlx-q4'に固定です。
mlx-communityから、mlx対応のモデルを自動的にダウンロードするようにしています。ダウンロード先は、$HOME/.cache/huggingface/hubです。

##### GoogleSTTを使用する場合:
- 詳細は[speech_recognitionリポジトリ](https://github.com/Uberi/speech_recognition)を参照してください。

```bash
pip install fastrtc-jp[sr]
```

- GoogleSTTクラス
Googleの音声認識エンジンを使用します。インターネット接続が必要です。

## 使用例

### 基本的なエコーバックサンプル

マイクの音声をそのままスピーカーにエコーバックするシンプルな例です。

```python
import sys, os
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

"""
マイクの音声をそのままスピーカーにエコーバックするだけのサンプル
"""
def echoback(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]}秒の音声が入力されました。")
    yield audio

def example_echoback():
    algo_options = AlgoOptions(
        audio_chunk_duration=0.6,
        started_talking_threshold=0.5,
        speech_threshold=0.1,
    )
    stream = Stream(
        handler=ReplyOnPause(
            echoback,
            algo_options=algo_options,
            input_sample_rate=16000,
            output_sample_rate=16000,
        ),
        modality="audio", 
        mode="send-receive",
    )

    stream.ui.launch()

if __name__ == "__main__":
    example_echoback()
```

### VOICEVOXで音声合成するサンプル

```python
import sys, os
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel, VoicevoxTTSOptions

"""
VOICEVOXで音声合成するだけのサンプル
"""

tts_model = VoicevoxTTSModel()  # デフォルトはlocalhostの50021ポートに接続
voicevox_opt = VoicevoxTTSOptions(
    speaker=8,  # つむぎ
    speedScale=1.0,  # 話速（1.0が標準）
)

def voicevox(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]}秒の音声が入力されました。")
    response = "やっほー、今日も元気だ。やきとり食べよう。"
    for audio_chunk in tts_model.stream_tts_sync(response, voicevox_opt):
        print("Sending audio")
        yield audio_chunk

def example_voicevox():
    algo_options = AlgoOptions(
        audio_chunk_duration=0.6,
        started_talking_threshold=0.5,
        speech_threshold=0.1,
    )
    stream = Stream(
        handler=ReplyOnPause(
            voicevox,
            algo_options=algo_options,
            input_sample_rate=16000,
            output_sample_rate=16000,
        ),
        modality="audio", 
        mode="send-receive",
    )

    stream.ui.launch()

if __name__ == "__main__":
    example_voicevox()
```

### 音声認識と音声合成を組み合わせたサンプル

```python
import sys, os
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

from fastrtc_jp.speech_to_text.sr_google import GoogleSTT
# from fastrtc_jp.speech_to_text.vosk import VoskSTT
from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel, VoicevoxTTSOptions
# from fastrtc_jp.text_to_speech.gtts import GTTSModel, GTTSOptions

"""
マイクの音声をSTT->TTSしてエコーバックするサンプル
"""

# 音声認識モデルの初期化
stt_model = GoogleSTT()
# stt_model = VoskSTT()  # Voskを使用する場合

# 音声合成モデルの初期化
tts_model = VoicevoxTTSModel()
voicevox_opt = VoicevoxTTSOptions(
    speaker_id=8,  # つむぎ
    speedScale=1.0,
)
# tts_model = GTTSModel()  # gTTSを使用する場合

def echoback(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]}秒の音声が入力されました。")
    # 音声認識
    user_input = stt_model.stt(audio)
    print(f"音声認識結果: {user_input}")
    
    # 認識した文章をそのまま音声合成してエコーバック
    response = user_input
    for audio_chunk in tts_model.stream_tts_sync(response, voicevox_opt):
        print("Sending audio")
        yield audio_chunk

def example_echoback():
    algo_options = AlgoOptions(
        audio_chunk_duration=0.6,
        started_talking_threshold=0.5,
        speech_threshold=0.1,
    )
    stream = Stream(
        handler=ReplyOnPause(
            echoback,
            algo_options=algo_options,
            input_sample_rate=16000,
            output_sample_rate=16000,
        ),
        modality="audio", 
        mode="send-receive",
    )

    stream.ui.launch()

if __name__ == "__main__":
    example_echoback()
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

---

fastrtcの詳細な使い方については、[fastrtc公式ドキュメント](https://fastrtc.org/)を参照してください。

