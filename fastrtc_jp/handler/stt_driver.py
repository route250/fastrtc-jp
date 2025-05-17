
from logging import getLogger

import numpy as np
from numpy.typing import NDArray

from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.speech_to_text.stt_ import STTModel
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class SttDriver(ABC):
    """
    STT Driver Interface
    """

    @abstractmethod
    def copy(self) -> "SttDriver":
        """
        Copy the driver instance
        """
        return self

    @abstractmethod
    async def start_up(self) -> None:
        """
        Start up the driver
        """
        pass

    @abstractmethod
    def get_vad(self,state:bool, sr:int, audio:NDArray[np.int16]|NDArray[np.float32], algo_options:AlgoOptions, options ) -> bool:
        pass

    @abstractmethod
    def get_stt_model(self) -> STTModel:
        pass