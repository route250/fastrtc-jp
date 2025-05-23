import asyncio
import inspect
import math
import traceback
from types import CoroutineType
from typing import Protocol, Type, Callable, Any, Literal, AsyncGenerator
from dataclasses import dataclass
from logging import getLogger
from enum import Enum
import time
import numpy as np
from numpy.typing import NDArray

from fastrtc import AsyncStreamHandler, AdditionalOutputs, wait_for_item
from fastrtc.tracks import EmitType

from fastrtc_jp.handler.agent_handler import AgentHandler
from fastrtc_jp.handler.service import STTService, TTSService
from fastrtc_jp.handler.voice import SttAudio, SttAudioBuffer, TtsAudio
from fastrtc_jp.speech_to_text.util import resample_audio

from fastrtc_jp.handler.vad import VadOptions, VadHandler
from fastrtc_jp.handler.stt_handler import SttHandler
from fastrtc_jp.handler.agent_task import AgentTask
from fastrtc_jp.handler.emit import EmitManager
from fastrtc_jp.handler.session import AgentMessage, AgentSession

def clear_queue(q:asyncio.Queue):
    try:
        while q.qsize()>0:
            q.get_nowait()
    except:
        pass


def cancel_task(task:asyncio.Task|None):
    try:
        if task is not None and not task.done:
            if not task.cancelled() or task.cancelling()==0:
                task.cancel()
    except:
        pass


class HdrStat(Enum):
    NotStarted = "NotStarted"
    Init = "Init"
    Idle = "run"
    Listen = "Listen"
    Thinking = "Thinking"
    Talking = "Talking"
    Error = "Error"
    Shutdown = "Shutdown"
    Stopped = "Stopped"

class AsyncVoiceStreamHandler(AsyncStreamHandler):
    logger = getLogger(f"{__name__}.{__qualname__}")
    def __init__(self,
        stt_hdr: SttHandler,
        driver: AgentHandler,
        *,
        #vad_fn:Callable[[bool,int,NDArray[np.int16]|NDArray[np.float32],AlgoOptions,Any],bool],
        get_tts_model_fn,
        vad_hdr:VadHandler|None=None,
        vad_options:VadOptions|None = None,
        # wakeup_words:list[str]|None=None
    ):
        """初期化"""
        super().__init__(
            expected_layout = 'mono',
            output_sample_rate = 24000,
            output_frame_size = None,
            input_sample_rate = 16000,
        )
        self._stat:HdrStat = HdrStat.Init
        self._stat_time:float = time.time()
        self.stt_hdr: SttHandler = stt_hdr
        self.driver:AgentHandler = driver
        if vad_hdr is None:
            self.vad_options:VadOptions = vad_options or VadOptions()
            self.vad_hdr = VadHandler(self.vad_options)
        else:
            self.vad_hdr = vad_hdr
            self.vad_options = vad_options or vad_hdr.vad_options
        self.emit_manager: EmitManager = EmitManager()
        self._before_in_talking:bool = False

        self.stt_queue:asyncio.Queue[SttAudio] = asyncio.Queue()
        self.agent_queue:asyncio.Queue[AgentTask] = asyncio.Queue()
        self.tts_queue:asyncio.Queue[TtsAudio] = asyncio.Queue()

        self.session:AgentSession = AgentSession("","","")

        self._stt_service:STTService = STTService(stt_hdr.get_stt_model)

        self.get_tts_model_fn = get_tts_model_fn
        self._tts_service:TTSService = TTSService(get_tts_model_fn)

        self._task_list:list[asyncio.Task] = []

        #self.wakeup_words: list[str] = [w for w in (wakeup_words or []) if w]
        self.wakeup_time:float = time.time()
        self._last_emit_time:float = time.time()

    def get_stat(self) -> HdrStat:
        return self._stat

    def set_stat(self, stat:HdrStat) -> None:
        if self._stat != stat:
            self._stat = stat
            self._stat_time = time.time()

    def is_running(self) -> bool:
        return self._stat in (HdrStat.Idle, HdrStat.Listen, HdrStat.Thinking, HdrStat.Talking)

    # @property
    # def _needs_additional_inputs(self) -> bool:
    #     """Checks if the reply function `fn` expects additional arguments."""
    #     return len(inspect.signature(self.fn).parameters) > 1

    async def request_args(self):
        if not self.phone_mode: 
            if self.channel:
                self.args_set.clear()
                await self.fetch_args()
        else:
            self.latest_args = [None]
            self.args_set.set()

    #Override
    def copy(self):
        try:
            return AsyncVoiceStreamHandler(
                self.stt_hdr.copy(),
                self.driver.copy(),
                vad_options = self.vad_options,
                get_tts_model_fn=self.get_tts_model_fn,
            )
        except:
            self.logger.exception("can not copy instance")


    def _stop_task(self):
        try:
            clear_queue(self.stt_queue)
            self.vad_hdr.reset()
            clear_queue(self.agent_queue)
            clear_queue(self.tts_queue)
            while len(self._task_list)>0:
                cancel_task( self._task_list.pop())
        except:
            self.logger.exception("can not reset")


    #Override
    async def start_up(self) -> None:
        try:
            await self.stt_hdr.start_up()
            await self.driver.start_up()
            await super().start_up()
            # 非同期タスクを開始
            self.set_stat(HdrStat.Init)
            self._task_list.append( asyncio.create_task(self._fn_task_stt()) )
            self._task_list.append( asyncio.create_task(self._fn_task_agent()) )
            self._task_list.append( asyncio.create_task(self._fn_task_tts()) )
            self._tts_service.start_up()
            self._stt_service.start_up()
            self._task_list.append( asyncio.create_task(self._fn_task_timer()) )
            self.set_stat(HdrStat.Idle)
        except:
            self.logger.exception("can not start_up")
            self.set_stat(HdrStat.Error)

    #Override
    def shutdown(self):
        err:bool = self.get_stat()==HdrStat.Error
        try:
            self.set_stat(HdrStat.Shutdown)
            super().shutdown()
            self._stop_task()
            self.driver.shutdown()
            self._tts_service.shutdown()
            self._stt_service.shutdown()
        except:
            self.logger.exception("can not shutdown")
        finally:
            if not err:
                self.set_stat(HdrStat.Stopped)
            else:
                self.set_stat(HdrStat.Error)


    #Override
    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        try:
            if not self.is_running():
                return
            stt_audio = await self.vad_hdr.receive(frame)
            if self.vad_hdr.in_talking:
                self.emit_manager.set_pause(True)
            if stt_audio:
                self.stt_queue.put_nowait(stt_audio)

            if self._before_in_talking != self.vad_hdr.in_talking:
                self._before_in_talking = self.vad_hdr.in_talking
                await asyncio.sleep(0.001)

        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit) as ex:
            self.logger.debug(f"receive cancelled {ex}")
        except:
            self.set_stat(HdrStat.Error)
            traceback.print_exc()
            self.logger.exception("error in receive")


    #Override
    async def emit(self) -> EmitType:
        try:
            if not self.is_running():
                return
            segment = await self.emit_manager.get_emit_segment()
            if segment is None:
                await asyncio.sleep(0.1)
            elif isinstance(segment,tuple) and len(segment)>=2 and isinstance(segment[1],np.ndarray):
                self._keep_talking()
            elif isinstance(segment, AdditionalOutputs):
                self.logger.debug(f"Emitting AdditionalOutputs: {segment}")
            return segment
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit) as ex:
            self.logger.debug(f"emit cancelled {ex}")
        except Exception as ex:
            self.set_stat(HdrStat.Error)
            traceback.print_exc()
            self.logger.exception(f"Error in emit: {ex}")


    def _keep_talking(self):
        if self.get_stat()==HdrStat.Listen or self.get_stat()==HdrStat.Talking:
            self._last_emit_time = time.time()


    async def _fn_task_timer(self):
        try:
            while self.is_running():
                await asyncio.sleep(1.0)
                if self.get_stat()==HdrStat.Talking:
                    aa = time.time() - self._last_emit_time
                    if aa>self.vad_options.grace_period_duration:
                        print(f"<stt> timeout {aa} COOLDOWN")
                        self.set_stat(HdrStat.Listen)
                elif self.get_stat()==HdrStat.Listen:
                    aa = time.time() - self._last_emit_time
                    if aa>self.vad_options.listen_mode_duration:
                        print(f"<stt> timeout {aa} IDLE")
                        self.set_stat(HdrStat.Idle)
                        await self.driver.end_session(self.session)
                        self.session = AgentSession(self.session.agent_id, self.session.session_id, self.session.user_id)
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit) as ex:
            self.logger.debug(f"timer cancelled {ex}")
        except Exception as ex:
            self.set_stat(HdrStat.Error)
            traceback.print_exc()
            self.logger.exception(f"Error in timer: {ex}")

    async def _fn_task_stt(self):
        before_task:AgentTask|None = None
        buffer_data: SttAudioBuffer = SttAudioBuffer()
        while self.is_running():
            try:
                # queueからデータを非同期に取得
                nx_stt_audio:SttAudio|None = await wait_for_item(self.stt_queue)
                if nx_stt_audio is not None:
                    asyncio.create_task( self.request_args() )
                    # 非同期でttsを実行
                    stt_result: str|None = await self._stt_service.stt( (nx_stt_audio.rate, nx_stt_audio.audio) )
                    if stt_result:
                        self._keep_talking()
                        nx_stt_audio.user_input = stt_result
                        if before_task is not None:
                            before_task.cancel()
                            if before_task.accepted<=0:
                                # 前回の入力がまったく処理されなかったら引き継ぐ
                                for s in before_task.stt:
                                    buffer_data.append(s)
                            before_task = None
                            if self.get_stat()==HdrStat.Thinking or self.get_stat()==HdrStat.Talking:
                                self.set_stat(HdrStat.Listen)
                        buffer_data.append(nx_stt_audio)

                        messages = self.session.get_messages()
                        messages += buffer_data.to_messages()
                        await self.emit_manager.ads( AdditionalOutputs([],messages))
                        # listen mode switch
                        if self.get_stat()==HdrStat.Idle:
                            # if not self.wakeup_words or any(w in stt_result for w in self.wakeup_words):
                            if self.stt_hdr.is_wakeup([stt_result]):
                                self.set_stat(HdrStat.Listen)
                                self._keep_talking()

                if self.get_stat()!=HdrStat.Idle and not self.vad_hdr.in_talking and self.stt_queue.qsize()==0:
                    self.emit_manager.set_pause(False)
                    if len(buffer_data)>0:
                        before_task = AgentTask(self.session, self.driver, buffer_data.copy_to_list() )
                        buffer_data.reset()
                        # 処理したデータをq1に送る
                        print(f"<stt> put to agent_queue {before_task.stt[-1].user_input}")
                        self.agent_queue.put_nowait(before_task)
                        await asyncio.sleep(0.001)

                # タスク完了を通知
                if nx_stt_audio is not None:
                    self.stt_queue.task_done()

            except (asyncio.CancelledError, asyncio.TimeoutError, EOFError, KeyboardInterrupt, SystemExit) as ex:
                self.logger.debug(f"task stt cancelled {ex}")
                break
            except Exception as e:
                self.logger.exception(f"Error in process_stt: {e}")


    async def _fn_task_agent(self):
        while self.is_running():
            try:
                agent_task:AgentTask = await self.agent_queue.get()
                args = self.latest_args
                print(f"<agent> args {args}")
                print(f"<agent> get from tts_quque")
                self._keep_talking()
                no:int = 0
                async for words in agent_task.execute():
                    tts_audio = TtsAudio(agent_task, no, words )
                    print(f"<agent> put to tts_queue {no} {tts_audio.ai_response}")
                    self.tts_queue.put_nowait(tts_audio)
                    self._keep_talking()
                    no+=1
                    await asyncio.sleep(0.05)
                self.agent_queue.task_done()
                print(f"<agent> done agent_task")
            except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit) as ex:
                self.logger.debug(f"task agent cancelled {ex}")
                break
            except Exception as e:
                self.logger.exception(f"Error in agent_task: {e}")
        print(f"<agent> end agent_task")


    async def _fn_task_tts(self):
        while self.is_running():
            try:
                # q2からデータを非同期に取得
                tts_data:TtsAudio = await self.tts_queue.get()
                # 非同期でttsを実行
                if not tts_data.is_canceled():
                    print(f"<tts> start {tts_data.ai_response}")
                    result = await self._tts_service.tts(tts_data.ai_response,None)
                    print(f" tts result {type(result)}")
                    tts_data.set_audio(result)
                    # 処理したデータをq1に送る
                    await self.emit_manager.put(tts_data)
                    self._keep_talking()
                    await asyncio.sleep(0.05)
                # タスク完了を通知
                self.tts_queue.task_done()
            except (asyncio.CancelledError, asyncio.TimeoutError, EOFError, KeyboardInterrupt, SystemExit) as ex:
                self.logger.debug(f"task tts cancelled {ex}")
                break
            except Exception as e:
                self.logger.exception(f"Error in process_tts: {e}")
                continue