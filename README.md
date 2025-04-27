# fastrtc-jp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

fastrtc用の日本語TTSとSTT追加キット（日本語音声合成・認識モジュール）

## 概要

[fastrtc](https://fastrtc.org/)は高性能なリアルタイム通信フレームワークですが、現状では日本語の音声合成(TTS)および音声認識(STT)機能が十分に対応していません。このプロジェクトは、fastrtcに日本語対応の音声処理機能を追加するための拡張パッケージです。

主な機能：
- 日本語に特化した音声合成モデル（Voicevox、gTTS）
- 日本語に対応した音声認識モデル（Vosk、Google Speech Recognition）
- シンプルで使いやすいAPI
- fastrtcとのシームレスな統合

## システム要件

- Python 3.11以上
- fastrtc（インストール方法は[fastrtc公式サイト](https://fastrtc.org/)を参照）
- 各モデルの追加要件は以下の「インストール」セクションを参照

## インストール

基本的な使い方として、Python仮想環境を作成してからpipでインストールすることをお勧めします。

```bash
# 仮想環境の作成と有効化（オプション）
python -m venv fastrtc-env
source fastrtc-env/bin/activate  # Linuxの場合
# または
.\fastrtc-env\Scripts\activate  # Windowsの場合

# 基本パッケージのインストール
pip install fastrtc-jp
```

### 追加モジュール

必要に応じて、以下の追加パッケージをインストールしてください：

#### 音声合成（TTS）

**GTTSModelを使用する場合**:
```bash
pip install fastrtc-jp[gtts]
```

**VoicevoxTTSModelを使用する場合**:
Voicevoxエンジンが必要です。[Voicevox公式サイト](https://voicevox.hiroshiba.jp/)からダウンロードし、ローカルで実行するか、APIサーバーにアクセスできる環境を準備してください。

#### 音声認識（STT）

**VoskSTTを使用する場合**:
```bash
pip install fastrtc-jp[vosk]
```
さらに、日本語モデルをダウンロードする必要があります。詳細は[Voskの公式ドキュメント](https://alphacephei.com/vosk/models)を参照してください。

**GoogleSTTを使用する場合**:
```bash
pip install fastrtc-jp[sr]
```
インターネット接続が必要です。

## 提供モデル一覧

| クラス | 説明 | 追加パッケージ | 特徴 |
|-------|------|--------------|------|
| VoicevoxTTSModel | Voicevoxのapiで音声合成するクラス | - | 高品質な日本語音声、多様な話者、感情表現が可能 |
| GTTSModel | Google Text-to-Speechで音声合成するクラス | gtts | インターネット接続が必要、自然な発話 |
| VoskSTT | Voskエンジンで音声認識するクラス | vosk | オフライン動作可能、軽量 |
| GoogleSTT | SpeechRecognizerのGoogleエンジンで音声認識するクラス | sr | 高精度、インターネット接続が必要 |

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

### Voicevoxで音声合成するサンプル

```python
import sys, os
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel, VoicevoxTTSOptions

"""
voicevoxで音声合成するだけのサンプル
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
    speaker=8,  # つむぎ
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

## モデル詳細

### VoicevoxTTSModel

[VOICEVOX](https://voicevox.hiroshiba.jp/)は、無料で使える高品質な日本語音声合成エンジンです。

**初期化オプション:**
- `host`: Voicevox APIのホスト（デフォルト: "localhost"）
- `port`: Voicevox APIのポート（デフォルト: 50021）

**VoicevoxTTSOptions:**
- `speaker`: 話者ID（整数）
- `speedScale`: 話速スケール（デフォルト: 1.0）
- その他のパラメータについては[VOICEVOX APIリファレンス](https://voicevox.github.io/voicevox_engine/api/)を参照

### GTTSModel

Google Text-to-Speechを使用した音声合成モデルです。

**GTTSOptions:**
- `lang`: 言語コード（デフォルト: "ja"）
- `slow`: ゆっくり話すかどうか（デフォルト: False）

### VoskSTT

[Vosk](https://alphacephei.com/vosk/)は、オフラインで動作する音声認識エンジンです。

**初期化オプション:**
- `model_path`: Voskモデルのパス

### GoogleSTT

Googleの音声認識エンジンを使用します。インターネット接続が必要です。

## トラブルシューティング

### Voicevoxとの接続エラー

1. Voicevoxエンジンが正常に起動しているか確認してください
2. デフォルトでは`localhost:50021`に接続します。別のアドレスを使用する場合は初期化時に指定してください
3. ファイアウォール設定を確認してください

### 音声認識の精度が低い場合

1. VoskSTTを使用している場合、より大きなモデルを試してください
2. GoogleSTTの場合、安定したインターネット接続があることを確認してください
3. ノイズの少ない環境で録音してください

### インストールエラー

依存関係のインストールに問題がある場合は、以下を試してください：

```bash
pip install --upgrade pip
pip install --upgrade setuptools wheel
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 貢献

バグレポート、機能リクエスト、プルリクエストなど、あらゆる形での貢献を歓迎します。

## 謝辞

- [fastrtc](https://fastrtc.org/)チーム
- [VOICEVOX](https://voicevox.hiroshiba.jp/)プロジェクト
- [Vosk](https://alphacephei.com/vosk/)プロジェクト
- その他、このプロジェクトに貢献してくれた全ての方々

---

fastrtcの詳細な使い方については、[fastrtc公式ドキュメント](https://fastrtc.org/)を参照してください。

