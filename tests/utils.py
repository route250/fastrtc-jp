import numpy as np
from numpy.typing import NDArray

import sounddevice as sd
def play_audio( audio: NDArray[np.float32], sample_rate:int ):
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        pass