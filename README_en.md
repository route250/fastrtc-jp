[Japanese](README.md)/[English](README_en.md)

# fastrtc-jp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/fastrtc-jp.svg)](https://pypi.org/project/fastrtc-jp/)

Japanese TTS and STT add-on kit for fastrtc (Japanese speech synthesis and recognition module)

## Overview

[fastrtc](https://fastrtc.org/) is a high-performance real-time communication framework, but currently, the Japanese speech synthesis (TTS) and speech recognition (STT) functions are insufficient. This project is an extension package to add Japanese-compatible speech processing functions to fastrtc.

Main features:
- Speech synthesis models specialized for Japanese (VOICEVOX, Style-Bert-VITS2, gTTS)
- Speech recognition models compatible with Japanese (MlxWhisper, Vosk, Google Speech Recognition)

For detailed usage of fastrtc, please refer to the [official fastrtc documentation](https://fastrtc.org/).

## List of Provided Models

| Class | Description | Additional Package | Features |
|-------|-------------|--------------------|----------|
| VoicevoxTTSModel | Class for speech synthesis using VOICEVOX API | - | High-quality Japanese speech, multiple speakers, emotional expression possible |
| StyleBertVits2 | Class for speech synthesis using Style-Bert-VITS2 | style-bert-vits2 / pyopenjtalk | High-quality speech synthesis, supports pretrained models |
| GTTSModel | Class for speech synthesis using Google Text-to-Speech | gtts | Requires internet connection, natural speech |
| VoskSTT | Class for speech recognition using Vosk engine | vosk | Offline operation possible, lightweight |
| MlxWhisper | Class for speech recognition using mlx-whisper | mlx-whisper | High accuracy speech recognition, optimized for Apple Silicon |
| GoogleSTT | Class for speech recognition using SpeechRecognizer's Google engine | SpeechRecognition==3.10.0 | High accuracy, requires internet connection |

## System Requirements

- Python 3.12 or higher
- Confirmed operation on Ubuntu 24.04 LTS, macOS Sonoma 15.4
- Additional requirements for each model are described in the "Installation" section below

## Installation

As a basic usage, it is recommended to create a Python virtual environment and then install via pip.

```bash
# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate
# Upgrade pip
.venv/bin/python3 -m pip install -U pip setuptools
# Install basic package
pip install fastrtc-jp
```

### Additional Modules

Install the following additional packages as needed:

#### Speech Synthesis (TTS)

##### Using VOICEVOX:

[VOICEVOX](https://voicevox.hiroshiba.jp/) is a free, high-quality Japanese speech synthesis engine.  
Please check the official documentation for the usage conditions of VOICEVOX and the voice models.  
Refer to [VOICEVOX Official](https://voicevox.hiroshiba.jp/) and prepare an environment accessible via API.

- VoicevoxTTSModel class

Set the VOICEVOX server address and port in the environment variable VOICEVOX_HOSTLIST.  
In the .env file or config.env file, write as follows.  
If not specified, the default below is set.

```text:config.env
VOICEVOX_HOSTLIST=http://127.0.0.1:50021
```

- VoicevoxTTSOptions class
  - `speaker_name`: Character name
  - `speaker_style`: Voice style
  - `speaker_id`: Speaker ID (integer)
  - `speedScale`: Speech speed scale (default: 1.0)

Check the character name, style, and ID at [http://127.0.0.1:50021/speakers](http://127.0.0.1:50021/speakers), and set either the speaker ID or the name and style.

##### Using Style-Bert-VITS2:

[Style-Bert-Vits2](https://github.com/litagin02/Style-Bert-VITS2) is a TTS model capable of high-quality speech synthesis. You can generate natural Japanese speech using pretrained models.  
Please check the official documentation for usage conditions of Style-Bert-VITS2 and voice models.  
Refer to the [style-bert-vits2 repository](https://github.com/litagin02/Style-Bert-VITS2) for details.  
To use the python style-bert-vits2 package from python code, there are two methods: starting an API server and calling the API, or direct execution. This package implements the direct execution method.

```bash
pip install fastrtc-jp[sbv2]
```

- StyleBertVits2 class

You can use either the standard voice model or another voice model.

- StyleBertVits2Options class

  - `model`: Model preset name
  - `model_path`: Path to the model file
  - `config_path`: Path to the configuration file
  - `style_vec_path`: Path to the style vector file
  - `device`: Device to use (default: "cpu")
  - `speaker_id`: Speaker ID (check each model's config.json)
  - `speaker_style`: Style (check each model's config.json)

You can specify the following preset names as model presets. When using presets, you do not need to set model_path, config_path, or style_vec_path. For models other than presets, download separately and specify model_path, config_path, and style_vec_path.  
Please check the distribution source for usage conditions of the models.

| Preset Name | Distribution Source |
|-------------|---------------------|
| jvn-F1-jp   | https://huggingface.co/litagin/style_bert_vits2_jvnv |
| jvn-F2-jp   | https://huggingface.co/litagin/style_bert_vits2_jvnv |
| jvn-M1-jp   | https://huggingface.co/litagin/style_bert_vits2_jvnv |
| jvn-M2-jp   | https://huggingface.co/litagin/style_bert_vits2_jvnv |
| rinne       | https://huggingface.co/RinneAi/Rinne_Style-Bert-VITS2 |
| girl        | https://huggingface.co/Mofa-Xingche/girl-style-bert-vits2-JPExtra-models |
| tsukuyomi-chan | https://huggingface.co/ayousanz/tsukuyomi-chan-style-bert-vits2-model |
| AbeShinzo   | https://huggingface.co/AbeShinzo0708/AbeShinzo_Style_Bert_VITS2 |
| sakura-miko | https://huggingface.co/Lycoris53/style-bert-vits2-sakura-miko |

Each file will be downloaded on first use. The download location is $HOME/.cache/huggingface/hub.

```python:example
StyleBertVits2Options(device="cpu",
    model_file="model/jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors",
    config_file="model/jvnv-M1-jp/config.json",
    style_file="model/jvnv-M1-jp/style_vectors.npy",
)
```

##### Using GTTSModel:

Refer to the [gTTS repository](https://github.com/pndurette/gTTS) for details.  
gTTS uses Google's API, so an internet connection is required.

```bash
pip install fastrtc-jp[gtts]
```

- GTTSModel class

Speech synthesis model using Google Text-to-Speech.

- GTTSOptions class

  - `speed`: Speaking speed (default 1.0)


#### Speech Recognition (STT)

##### Using VoskSTT:

[Vosk](https://alphacephei.com/vosk/) is an offline speech recognition engine.  
Refer to [Vosk official](https://alphacephei.com/vosk) for details.  
Japanese model is automatically downloaded as well. The download location is $HOME/.cache/vosk.

```bash
pip install fastrtc-jp[vosk]
```
- VoskSTT class

Currently fixed to 'vosk-model-ja-0.22'.

##### Using mlx-whisper:

[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) is an implementation of the Whisper model optimized for Apple Silicon.  
Refer to the [MLX Examples repository](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for details.

```bash
pip install fastrtc-jp[mlx]
```

- MlxWhisper class

Currently fixed to 'mlx-community/whisper-medium-mlx-q4'.  
Models compatible with mlx are automatically downloaded from mlx-community. The download location is $HOME/.cache/huggingface/hub.

##### Using GoogleSTT:
- Refer to the [speech_recognition repository](https://github.com/Uberi/speech_recognition) for details.

```bash
pip install fastrtc-jp[sr]
```

- GoogleSTT class  
Uses Google's speech recognition engine. Requires internet connection.

## Usage Examples

### Basic Echo Back Sample

A simple example that echoes microphone audio directly to the speaker.

```python
import sys, os
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

"""
A sample that simply echoes microphone audio to the speaker
"""
def echoback(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]} seconds of audio input received.")
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

### Sample of Speech Synthesis with VOICEVOX

```python
import sys, os
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel, VoicevoxTTSOptions

"""
Sample that simply synthesizes speech with VOICEVOX
"""

tts_model = VoicevoxTTSModel()  # Default connects to localhost port 50021
voicevox_opt = VoicevoxTTSOptions(
    speaker=8,  # Tsumugi
    speedScale=1.0,  # Speech speed (1.0 is standard)
)

def voicevox(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]} seconds of audio input received.")
    response = "Hey there, I'm feeling good today. Let's eat yakitori."
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

### Sample Combining Speech Recognition and Speech Synthesis

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
Sample that performs STT->TTS on microphone audio and echoes back
"""

# Initialize speech recognition model
stt_model = GoogleSTT()
# stt_model = VoskSTT()  # Use Vosk if preferred

# Initialize speech synthesis model
tts_model = VoicevoxTTSModel()
voicevox_opt = VoicevoxTTSOptions(
    speaker_id=8,  # Tsumugi
    speedScale=1.0,
)
# tts_model = GTTSModel()  # Use gTTS if preferred

def echoback(audio: tuple[int, np.ndarray]):
    print(f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]} seconds of audio input received.")
    # Speech recognition
    user_input = stt_model.stt(audio)
    print(f"Speech recognition result: {user_input}")
    
    # Synthesize recognized text and echo back
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

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For detailed usage of fastrtc, please refer to the [official fastrtc documentation](https://fastrtc.org/).
