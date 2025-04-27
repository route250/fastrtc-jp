import sys,os
sys.path.insert(0, '.')

import numpy as np

from fastrtc import ReplyOnPause
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.stream import Stream

"""
マイクの音声をそのままスピーカーにエコーバックするだけのサンプル
"""
def echoback(audio: tuple[int, np.ndarray]):
    print( f"shape:{audio[1].shape} dtype:{audio[1].dtype} {audio[0]}Hz {audio[1].shape[1]/audio[0]}秒の音声が入力されました。" )
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
