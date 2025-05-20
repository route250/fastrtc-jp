
from logging import getLogger

import numpy as np
from numpy.typing import NDArray

from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.speech_to_text.stt_ import STTModel
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class SttHandler(ABC):
    """
    STT Driver Interface
    """

    @abstractmethod
    def copy(self) -> "SttHandler":
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
    def get_stt_model(self) -> STTModel:
        pass

    @abstractmethod
    def is_wakeup(self,contents:list[str]) -> bool:
        pass