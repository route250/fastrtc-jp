import sys
import asyncio
from typing import Any, Callable
from multiprocessing import Process, Value, Queue
from queue import Empty
from multiprocessing.sharedctypes import Synchronized
from logging import getLogger

import numpy as np
from numpy.typing import NDArray

from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.text_to_speech.tts import TTSModel


NOT_STARTED = 0  # Process has not started yet
INITIALIZING = 1  # Process is initializing
READY = 2 # Process is waiting
RUNNING = 3  # Process is running
STOPPING = 4  # Process is in the process of stopping
STOPPED = 5  # Process has stopped


class ServiceProcess:
    """
    Base class for service processes using multiprocessing and async communication.
    Manages process lifecycle, inter-process communication, and state.
    """
    logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(self):
        """
        Initialize shared state, command flag, input/output queues, and process handle.
        """
        self.stat:Synchronized = Value('i',NOT_STARTED)
        self.cmd:Synchronized = Value('i',0)
        self.inp = Queue()
        self.out = Queue()
        self.process:Process|None = None

    def start_up(self):
        """
        Start the process if it has not been started yet.
        If already stopped, raise an exception.
        """
        if self.process is not None:
            if self.stat.value == NOT_STARTED:
                self.stat.value = INITIALIZING
                self.process.start()
            elif not self.process.is_alive():
                self.stat.value = STOPPED
        elif self.is_stopped():
            raise Exception("stopped")

    def is_stopped(self) ->bool:
        """
        Check if the process is stopping or has stopped.
        """
        return self.stat.value == STOPPING or self.stat.value == STOPPED

    def shutdown(self):
        """
        Gracefully shutdown the process and clean up resources.
        Sends shutdown command, waits for process to stop, and closes queues.
        """
        try:
            self.cmd.value=1
            self.inp.put_nowait(None)
            n = 0
            # Wait for the process to stop, with a timeout
            while n<10 and self.stat.value != STOPPED:
                n+=1
                if self.process is not None:
                    self.process.join(timeout=0.1)
            try:
                if self.process is not None:
                    self.process.kill()  # Force kill if not stopped
            except:
                pass
        except:
            self.logger.exception("Exception during shutdown command or join.")
        try:
            self.inp.close()
            self.inp.join_thread()
            self.out.close()
            self.out.join_thread()
        except:
            self.logger.exception("Exception during queue cleanup.")

    async def _put_and_get(self,data):
        """
        Put data into the input queue and asynchronously wait for a result from the output queue.
        """
        if self.is_stopped():
            raise EOFError("Service is closed")
        try:
            self.start_up()
            self.inp.put_nowait(data)
            while not self.is_stopped():
                try:
                    ret = self.out.get_nowait()
                    if ret is None:
                        raise EOFError("Service is None")
                    return ret
                except Empty:
                    pass
                await asyncio.sleep(0.1)
        except EOFError as ex:
            raise ex
        except Exception as ex:
            self.logger.exception("Exception during put and get.")
            raise EOFError("Exception during put and get.") from ex


class STTService(ServiceProcess):
    """
    Speech-to-Text (STT) service process wrapper.
    Handles audio input and returns recognized text using a separate process.
    """
    def __init__(self, get_stt_model_fn:Callable[[],STTModel]):
        """
        Initialize the STT service process with a model factory function.
        """
        super().__init__()
        self.process = Process(
                target=STTService._service_task,
                args=(
                    self.stat,self.cmd,self.inp,self.out,get_stt_model_fn
                )
            )

    @staticmethod
    def _service_task(
            stat:Synchronized,cmd:Synchronized, inp:Queue, out:Queue,
            get_stt_model_fn: Callable[[],STTModel]
        ):
        """
        Main loop for the STT service process.
        Waits for audio input, performs speech recognition, and returns text.
        """
        try:
            stat.value = INITIALIZING
            model:STTModel = get_stt_model_fn()
            while cmd.value==0:
                stat.value = READY
                audio = inp.get()
                if audio is not None:
                    try:
                        stat.value = RUNNING
                        text:str = model.stt(audio)
                        out.put_nowait(text)
                    except Exception as ex:
                        out.put_nowait(ex)
        except:
            ServiceProcess.logger.exception("Exception in STTService process.")
        finally:
            stat.value = STOPPING
            try:
                pass
            except:
                ServiceProcess.logger.exception("Exception during STTService cleanup.")

            stat.value = STOPPED

    async def stt(self, audio:tuple[int, NDArray[np.float32] | NDArray[np.int16]]) ->str:
        """
        Asynchronously send audio data for speech recognition and return the recognized text.
        """
        ret = await self._put_and_get( audio )
        if isinstance(ret,str):
            return ret
        elif isinstance(ret,Exception):
            raise ret
        else:
            raise ValueError(f"not str {type(ret)}")

class TTSService(ServiceProcess):
    """
    Text-to-Speech (TTS) service process wrapper.
    Handles text input and returns synthesized audio using a separate process.
    """
    def __init__(self, get_tts_model_fn:Callable[[],TTSModel]):
        """
        Initialize the TTS service process with a model factory function.
        """
        super().__init__()
        self.process = Process(
                target=TTSService._service_task,
                args=(
                    self.stat,self.cmd,self.inp,self.out,get_tts_model_fn
                )
            )

    @staticmethod
    def _service_task(
            stat:Synchronized,cmd:Synchronized, inp:Queue, out:Queue,
            get_tts_model_fn: Callable[[],TTSModel]
        ):
        """
        Main loop for the TTS service process.
        Waits for text input, performs speech synthesis, and returns audio data.
        """
        try:
            stat.value = INITIALIZING
            model:TTSModel = get_tts_model_fn()
            while cmd.value==0:
                stat.value = READY
                input_data = inp.get()
                if input_data is not None:
                    try:
                        stat.value = RUNNING
                        text,options = input_data
                        audio:tuple[int, NDArray[np.float32] | NDArray[np.int16]] = model.tts(text,options)
                        out.put_nowait(audio)
                    except Exception as ex:
                        out.put_nowait(ex)
        except:
            ServiceProcess.logger.exception("Exception in TTSService process.")
        finally:
            stat.value = STOPPING
            try:
                pass
            except:
                ServiceProcess.logger.exception("Exception during TTSService cleanup.")

            stat.value = STOPPED

    async def tts(self, text:str, options:Any) ->tuple[int, NDArray[np.float32] | NDArray[np.int16]]:
        """
        Asynchronously send text and options for speech synthesis and return the generated audio data.
        """
        ret = await self._put_and_get( (text,options) )
        if isinstance(ret,tuple):
            return ret
        elif isinstance(ret,Exception):
            raise ret
        else:
            raise ValueError(f"not audio {type(ret)}")
