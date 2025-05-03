import sys,os
sys.path.insert(0, '.')
import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel, VoicevoxTTSOptions

"""
voicevoxで音声合成するだけのサンプル
"""

tts_model = VoicevoxTTSModel()
voicevox_opt=VoicevoxTTSOptions(
    speaker_id=8, # つむぎ
    speedScale=1.0,
)

def voicevox(audio: tuple[int, np.ndarray]):
    print( f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]}秒の音声が入力されました。" )
    response="やっほー、今日も元気だ。やきとり食べよう。"
    for audio_chunk in tts_model.stream_tts_sync(response,voicevox_opt):
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
