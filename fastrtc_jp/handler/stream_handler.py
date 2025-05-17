import asyncio
import inspect
import traceback
from types import CoroutineType
from typing import Protocol, Type, Callable, Any, Literal, AsyncGenerator
from logging import getLogger
from enum import Enum
import time
import numpy as np
from numpy.typing import NDArray

from fastrtc import AsyncStreamHandler, AdditionalOutputs, wait_for_item
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.tracks import EmitType

from fastrtc_jp.handler.agent_driver import AgentDriver
from fastrtc_jp.handler.service import STTService, TTSService
from fastrtc_jp.handler.voice import SttAudio, SttAudioBuffer, TtsAudio
from fastrtc_jp.speech_to_text.util import resample_audio

from fastrtc_jp.handler.vad import VadHandler
from fastrtc_jp.handler.stt_driver import SttDriver
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
    Init = 1
    Running = 2
    Shutdown = 3

class Listen(Enum):
    IDLE = 0
    TALKING=4
    COOLDOWN=6

class AsyncVoiceStreamHandler(AsyncStreamHandler):
    logger = getLogger(f"{__name__}.{__qualname__}")
    def __init__(self,
        stt_driver: SttDriver,
        driver: AgentDriver,
        *,
        #vad_fn:Callable[[bool,int,NDArray[np.int16]|NDArray[np.float32],AlgoOptions,Any],bool],
        get_tts_model_fn,
        algo_options: AlgoOptions|None=None,
        vad_options = None,
        wakeup_words:list[str]|None=None
    ):
        """初期化"""
        super().__init__(
            expected_layout = 'mono',
            output_sample_rate = 24000,
            output_frame_size = None,
            input_sample_rate = 16000,
        )
        self.stat:HdrStat = HdrStat.Init
        self.stt_driver: SttDriver = stt_driver
        self.driver:AgentDriver = driver
        #self.vad_fn = vad_fn
        self.algo_options = algo_options
        self.vad_options = vad_options
        self.vad_hdr = VadHandler(self.stt_driver.get_vad, algo_options, vad_options)
        self.emit_manager: EmitManager = EmitManager()
        self._before_in_talking:bool = False

        self.stt_queue:asyncio.Queue[SttAudio] = asyncio.Queue()
        self.agent_queue:asyncio.Queue[AgentTask] = asyncio.Queue()
        self.tts_queue:asyncio.Queue[TtsAudio] = asyncio.Queue()

        self.session = AgentSession("","","")

        self._stt_service:STTService = STTService(stt_driver.get_stt_model)

        self.get_tts_model_fn = get_tts_model_fn
        self._tts_service:TTSService = TTSService(get_tts_model_fn)

        self._task_list:list[asyncio.Task] = []

        self.wakeup_words: list[str] = [w for w in (wakeup_words or []) if w]
        self.wakeup_status:Listen = Listen.IDLE
        self.wakeup_time:float = time.time()
        self._last_emit_time:float = time.time()

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
                self.stt_driver.copy(),
                self.driver.copy(),
                algo_options = self.algo_options,
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


    # #Override
    # def reset(self):
    #     try:
    #         self.stat = HdrStat.Init
    #         super().reset()
    #         self._stop_task()
    #         self.driver.reset()
    #     except:
    #         self.logger.exception("can not reset")


    #Override
    async def start_up(self) -> None:
        try:
            await self.driver.start_up()
            await super().start_up()
            # 非同期タスクを開始
            self.stat = HdrStat.Running
            self._task_list.append( asyncio.create_task(self._fn_task_stt()) )
            self._task_list.append( asyncio.create_task(self._fn_task_agent()) )
            self._task_list.append( asyncio.create_task(self._fn_task_tts()) )
            self._tts_service.start_up()
            self._stt_service.start_up()
            self._task_list.append( asyncio.create_task(self._fn_task_timer()) )
        except:
            self.logger.exception("can not start_up")


    #Override
    def shutdown(self):
        try:
            self.stat = HdrStat.Shutdown
            super().shutdown()
            self._stop_task()
            self.driver.shutdown()
            self._tts_service.shutdown()
            self._stt_service.shutdown()
        except:
            self.logger.exception("can not shutdown")


    #Override
    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        try:
            stt_audio = await self.vad_hdr.receive(frame)
            if self.vad_hdr.in_talking:
                self.emit_manager.set_pause(True)
            if stt_audio:
                self.stt_queue.put_nowait(stt_audio)
                self.emit_manager.set_pause(False)

            if self._before_in_talking != self.vad_hdr.in_talking:
                self._before_in_talking = self.vad_hdr.in_talking
                await asyncio.sleep(0.001)

        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit):
            pass
        except:
            traceback.print_exc()
            self.logger.exception("error in receive")


    #Override
    async def emit(self) -> EmitType:
        try:
            segment = await self.emit_manager.get_emit_segment()
            if segment is None:
                await asyncio.sleep(0.1)
            elif isinstance(segment,tuple) and len(segment)>=2 and isinstance(segment[1],np.ndarray):
                self._last_emit_time = time.time()
            return segment
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit):
            pass
        except Exception as ex:
            self.logger.exception(f"Error in emit: {ex}")
    
    async def _fn_task_timer(self):
        try:
            while self.stat == HdrStat.Running:
                await asyncio.sleep(1.0)
                if self.wakeup_status == Listen.TALKING:
                    aa = time.time() - self._last_emit_time
                    if aa>5.0:
                        self.wakeup_status = Listen.COOLDOWN
                elif self.wakeup_status == Listen.COOLDOWN:
                    aa = time.time() - self._last_emit_time
                    if aa>15.0:
                        self.wakeup_status = Listen.IDLE
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            self.logger.exception(f"Error in process_stt: {e}")

    async def _fn_task_stt(self):
        before_task:AgentTask|None = None
        buffer_data: SttAudioBuffer = SttAudioBuffer()
        while self.stat == HdrStat.Running:
            try:
                # queueからデータを非同期に取得
                nx_stt_audio:SttAudio|None = await wait_for_item(self.stt_queue)
                if nx_stt_audio is not None:
                    asyncio.create_task( self.request_args() )
                    # 非同期でttsを実行
                    stt_result: str|None = await self._stt_service.stt( (nx_stt_audio.rate, nx_stt_audio.audio) )
                    if stt_result:
                        nx_stt_audio.user_input = stt_result
                        if before_task is not None:
                            before_task.cancel()
                            if before_task.accepted<=0:
                                # 前回の入力がまったく処理されなかったら引き継ぐ
                                for s in before_task.stt:
                                    buffer_data.append(s)
                            before_task = None
                        buffer_data.append(nx_stt_audio)

                        messages = self.session.get_messages()
                        messages += buffer_data.to_messages()
                        await self.emit_manager.ads( AdditionalOutputs([],messages))
                        # listen mode switch
                        if self.wakeup_status==Listen.IDLE or self.wakeup_status==Listen.COOLDOWN:
                            if not self.wakeup_words or any(w in stt_result for w in self.wakeup_words):
                                self.wakeup_status = Listen.TALKING
                                self.wakeup_time = time.time()

                if not self.vad_hdr.in_talking and self.stt_queue.qsize()==0:
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

            except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                self.logger.exception(f"Error in process_stt: {e}")


    async def _fn_task_agent(self):
        while self.stat == HdrStat.Running:
            try:
                agent_task:AgentTask = await self.agent_queue.get()
                args = self.latest_args
                print(f"<agent> args {args}")
                print(f"<agent> get from tts_quque")
                no:int = 0
                async for words in agent_task.execute():
                    tts_audio = TtsAudio(agent_task, no, words )
                    print(f"<agent> put to tts_queue {no} {tts_audio.ai_response}")
                    self.tts_queue.put_nowait(tts_audio)
                    no+=1
                    await asyncio.sleep(0.05)
                self.agent_queue.task_done()
                print(f"<agent> done agent_task")
            except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit):
                print(f"<agent> cancelled")
                break
            except Exception as e:
                self.logger.exception(f"Error in agent_task: {e}")
        print(f"<agent> end agent_task")


    async def _fn_task_tts(self):
        while self.stat == HdrStat.Running:
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
                    await asyncio.sleep(0.05)
                # タスク完了を通知
                self.tts_queue.task_done()
            except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                self.logger.exception(f"Error in process_tts: {e}")
                continue