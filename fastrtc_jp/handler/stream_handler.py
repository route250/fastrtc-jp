import asyncio
import inspect
import math
import traceback
from types import CoroutineType
from typing import Protocol, Type, Callable, Any, Literal, AsyncGenerator,  TypeVar, Generic
from dataclasses import dataclass
from logging import getLogger
from enum import Enum
import time
import numpy as np
from numpy.typing import NDArray

from fastrtc import AsyncStreamHandler, AdditionalOutputs
from fastrtc.tracks import EmitType

from fastrtc_jp.handler.agent_handler import AgentHandler
from fastrtc_jp.handler.service import STTService, TTSService
from fastrtc_jp.handler.voice import SttAudio, SttAudioBuffer, TtsAudio

from fastrtc_jp.handler.vad import VadOptions, VadHandler
from fastrtc_jp.handler.stt_handler import SttHandler
from fastrtc_jp.handler.agent_task import AgentTask
from fastrtc_jp.handler.emit import EmitManager
from fastrtc_jp.handler.session import AgentMessage, AgentSession
from fastrtc_jp.text_to_speech.opt import SpkOptions
from fastrtc_jp.text_to_speech.tts_provider import TtsProvider

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

async def wait_for_item(queue: asyncio.Queue, timeout: float = 0.1) -> Any:
    """
    Wait for an item from an asyncio.Queue with a timeout.

    This function attempts to retrieve an item from the queue using asyncio.wait_for.
    If the timeout is reached, it returns None.

    This is useful to avoid blocking `emit` when the queue is empty.
    """

    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        return None

class HdrStat(Enum):
    NotStarted = "NotStarted"
    Init = "Init"
    Idle = "Idle"
    Wait = "Wait"
    Thinking = "Thinking"
    Talking = "Talking"
    Error = "Error"
    Shutdown = "Shutdown"
    Stopped = "Stopped"

Typ = TypeVar('Typ')

class EventValue(Generic[Typ]):
    def __init__(self, value: Typ):
        self.event:asyncio.Event = asyncio.Event()
        self.value:Typ = value
        self.event.set()  # 初期値を設定しておく

    def __repr__(self):
        return f"EventValue({self.value})"

    def __str__(self):
        return str(self.value)

    def set_value(self, value: Typ):
        """Set the value and trigger the event."""
        if self.value != value:
            self.value = value
            self.event.set()

    def set(self):
        self.event.set()

    def is_set(self) -> bool:
        return self.event.is_set()

    def clear(self):
        self.event.clear()

    def get_value(self) -> tuple[bool, Typ]:
        value = self.value
        if self.event.is_set():
            self.event.clear()
            return True, value
        return False, value


class AsyncVoiceStreamHandler(AsyncStreamHandler):
    logger = getLogger(f"{__name__}.{__qualname__}")
    def __init__(self,
        stt_hdr: SttHandler,
        agent_hdr: AgentHandler,
        *,
        tts_provider:Type[TtsProvider],
        vad_hdr:VadHandler|None=None,
        vad_options:VadOptions|None = None,
    ):
        """初期化"""
        super().__init__(
            expected_layout = 'mono',
            output_sample_rate = 24000,
            output_frame_size = None,
            input_sample_rate = 16000,
        )
        self.agent_id: str = "default"
        self.session_id: str = "default"
        self.user_id: str = "default"
        self._stat:HdrStat = HdrStat.Init
        self.stt_hdr: SttHandler = stt_hdr
        self.agent_hdr:AgentHandler = agent_hdr
        default_profile = next(iter(agent_hdr.get_profile_list()))
        self.agent_profile:EventValue[str] = EventValue(default_profile)
        if vad_hdr is None:
            self.vad_options:VadOptions = vad_options or VadOptions()
            self.vad_hdr = VadHandler(self.vad_options)
        else:
            self.vad_hdr = vad_hdr
            self.vad_options = vad_options or vad_hdr.vad_options
        self.emit_manager: EmitManager = EmitManager()


        self.stt_queue:asyncio.Queue[SttAudio] = asyncio.Queue()
        self.agent_queue:asyncio.Queue[AgentTask] = asyncio.Queue()
        self.tts_queue:asyncio.Queue[TtsAudio] = asyncio.Queue()

        self.session:AgentSession|None = None

        self._stt_service:STTService = STTService(stt_hdr.get_stt_model)

        self.tts_provider = tts_provider
        self._tts_service:TTSService = TTSService(tts_provider)

        self._task_list:list[asyncio.Task] = []

        #self.wakeup_words: list[str] = [w for w in (wakeup_words or []) if w]
        self.wakeup_time:float = time.time()
        self._last_emit_time:float = time.time()

        # status for ui
        self._stat_update_time:float = time.time()
        self._stat_emit_time:float = 0
        self._stat_dict:dict[str, str|dict|list|tuple] = {
            "stat": self._stat.value,
            "profile": self.agent_profile.value,
        }
        self._in_listen:bool = False
        self._stat_messages = []

    def get_stat(self) -> HdrStat:
        return self._stat

    def set_stat(self, stat:HdrStat) -> None:
        if self._stat != stat:
            self._stat = stat
            if self._in_listen:
                self._stat_dict["stat"] = "Listen"
            else:
                self._stat_dict["stat"] = self._stat.value
            self._stat_update_time = time.time()

    def is_running(self) -> bool:
        return self._stat in (HdrStat.Idle, HdrStat.Wait, HdrStat.Thinking, HdrStat.Talking)

    def set_lisetn(self, listen:bool):
        if self._in_listen != listen:
            self._in_listen = listen
            if listen:
                self._stat_dict["stat"] = "Listen"
            else:
                self._stat_dict["stat"] = self._stat.value
            self._stat_update_time = time.time()

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
                self.agent_hdr.copy(),
                vad_options = self.vad_options,
                tts_provider = self.tts_provider,
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
            await self.agent_hdr.start_up()
            await super().start_up()
            # 非同期タスクを開始
            self.set_stat(HdrStat.Init)
            self._task_list.append( asyncio.create_task(self._fn_task_stt()) )
            self._task_list.append( asyncio.create_task(self._fn_task_agent()) )
            self._task_list.append( asyncio.create_task(self._fn_task_tts()) )
            self._tts_service.start_up()
            self._stt_service.start_up()
            self._task_list.append( asyncio.create_task(self._fn_task_timer()) )
            self._task_list.append( asyncio.create_task(self._fn_task_args()) )
            await self.fetch_args()
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
            self.agent_hdr.shutdown()
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
                self.set_lisetn(True)
                self.emit_manager.set_pause(True)
            if stt_audio:
                self.stt_queue.put_nowait(stt_audio)

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
                if self._stat_emit_time < self._stat_update_time:
                    segment = AdditionalOutputs(self._stat_dict, self._stat_messages)
                    self._stat_emit_time = time.time()
                else:
                    await asyncio.sleep(0.1)
            elif isinstance(segment,tuple) and len(segment)>=2 and isinstance(segment[1],np.ndarray):
                self._keep_status()
            elif isinstance(segment, AdditionalOutputs):
                self.logger.debug(f"Emitting AdditionalOutputs: {segment}")
                self._stat_messages = segment.args[1]
                segment.args = (self._stat_dict, segment.args[1])
                self._stat_emit_time = time.time()
            return segment
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit) as ex:
            self.logger.debug(f"emit cancelled {ex}")
        except Exception as ex:
            self.set_stat(HdrStat.Error)
            traceback.print_exc()
            self.logger.exception(f"Error in emit: {ex}")

    def _keep_status(self):
        if self.get_stat()==HdrStat.Idle or self.get_stat()==HdrStat.Wait or self.get_stat()==HdrStat.Thinking or self.get_stat()==HdrStat.Talking:
            self._last_emit_time = time.time()

    async def set_profile(self, profile:str):
        self.agent_profile.set_value(profile)

    async def set_threshold(self, threshold:float, vad_model:str|None=None):
        if self.vad_hdr:
            await self.vad_hdr.set_threshold(threshold, vad_model)

    async def handle_args(self, args:tuple|list):
        print(f"[args] {args}")
        if len(args)>=1 and isinstance(args[0],str):
            await self.set_profile(args[0])
        if len(args)>=3:
            await self.set_threshold(args[1],args[2])
        elif len(args)>=2:
            await self.set_threshold(args[1])
    
    async def _fn_task_args(self):
        try:
            before = []
            while self.is_running():
                await self.fetch_args()
                after = self.latest_args[1:] if isinstance(self.latest_args, (list, tuple)) and len(self.latest_args) > 1 else []
                if before != after:
                    await self.handle_args(after)
                    before = after
                await asyncio.sleep(0.2)
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyboardInterrupt, SystemExit) as ex:
            self.logger.debug(f"args cancelled {ex}")
        except Exception as ex:
            self.set_stat(HdrStat.Error)
            traceback.print_exc()
            self.logger.exception(f"Error in args: {ex}")

    async def update_profile(self):
        if self.get_stat() == HdrStat.Idle or self.get_stat() == HdrStat.Wait:
            b, profile = self.agent_profile.get_value()
            if b:
                if self.session is not None:
                    await self.agent_hdr.end_session(self.session)
                    self.session = None
                self.logger.debug(f"update profile {profile}")
                self._stat_dict["profile"] = profile
                self._stat_update_time = time.time()
                await self.emit_manager.ads(AdditionalOutputs(self._stat_dict, []))


    async def new_session(self) ->AgentSession:
        if self.session is None:
            print(f"### new session")
            self.session = await self.agent_hdr.start_session(self.agent_id, self.session_id, self.agent_id, self.agent_profile.value)
        return self.session

    async def _fn_task_timer(self):
        try:
            while self.is_running():
                await asyncio.sleep(1.0)
                if self.get_stat()==HdrStat.Idle:
                    if self._in_listen and not self.vad_hdr.in_talking and self.stt_queue.qsize()==0:
                        bb = time.time() - self._last_emit_time
                        if  bb > 3.0:
                            self.set_lisetn(False)
                            print(f"<stt> timeout {bb} Idle")
                elif self.get_stat()==HdrStat.Thinking or self.get_stat()==HdrStat.Talking:
                    aa = time.time() - self._last_emit_time
                    if aa>self.vad_options.grace_period_duration:
                        print(f"<stt> timeout {aa} Talking")
                        self.set_stat(HdrStat.Wait)
                elif self.get_stat()==HdrStat.Wait:
                    aa = time.time() - self._last_emit_time
                    if aa>self.vad_options.listen_mode_duration:
                        print(f"<stt> timeout {aa} Idle")
                        self.set_stat(HdrStat.Idle)
                        if self.session is not None:
                            await self.agent_hdr.end_session(self.session)
                        self.session = None
                await self.update_profile()
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
                    # asyncio.create_task( self.request_args() )
                    # 非同期でttsを実行
                    stt_result: str|None = await self._stt_service.stt( (nx_stt_audio.rate, nx_stt_audio.audio) )
                    if stt_result:
                        self._keep_status()
                        nx_stt_audio.user_input = stt_result
                        if before_task is not None:
                            before_task.cancel()
                            if before_task.accepted<=0:
                                # 前回の入力がまったく処理されなかったら引き継ぐ
                                for s in before_task.stt:
                                    buffer_data.append(s)
                            before_task = None
                            if self.get_stat()==HdrStat.Thinking or self.get_stat()==HdrStat.Talking:
                                self.set_stat(HdrStat.Wait)
                        buffer_data.append(nx_stt_audio)

                        messages = await self.session.get_messages() if self.session else []
                        messages += buffer_data.to_messages()
                        await self.emit_manager.ads( AdditionalOutputs([],messages))
                        # listen mode switch
                        if self.get_stat()==HdrStat.Idle:
                            # if not self.wakeup_words or any(w in stt_result for w in self.wakeup_words):
                            if self.stt_hdr.is_wakeup([stt_result]):
                                self.set_stat(HdrStat.Wait)
                                self._keep_status()
                
                if self.get_stat()!=HdrStat.Idle and not self.vad_hdr.in_talking and self.stt_queue.qsize()==0:
                    self.emit_manager.set_pause(False)
                    if len(buffer_data)>0:
                        self.set_stat(HdrStat.Thinking)
                        before_task = AgentTask( self.agent_hdr, buffer_data.copy_to_list() )
                        buffer_data.reset()
                        # 処理したデータをq1に送る
                        print(f"<stt> put to agent_queue {before_task.stt[-1].user_input}")
                        self.agent_queue.put_nowait(before_task)
                        await asyncio.sleep(0.001)

                # タスク完了を通知
                if nx_stt_audio is not None:
                    self.stt_queue.task_done()

                if len(buffer_data)==0 and not self.vad_hdr.in_talking and self.stt_queue.qsize()==0:
                    self.set_lisetn(False)

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
                self._keep_status()
                no:int = 0
                session = await self.new_session()

                async for words in agent_task.execute(session):
                    tts_audio = TtsAudio(agent_task, no, words )
                    print(f"<agent> put to tts_queue {no} {tts_audio.ai_response}")
                    self.tts_queue.put_nowait(tts_audio)
                    if not agent_task.is_canceled() and self._stat==HdrStat.Thinking:
                        self.set_stat(HdrStat.Talking)
                    self._keep_status()
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
                    profile = self.agent_profile.value
                    opts:SpkOptions = self.agent_hdr.get_tts_options(profile)
                    print(f"<tts> start {profile} {opts.speaker_name} {tts_data.ai_response}")
                    result = await self._tts_service.tts(opts, tts_data.ai_response)
                    print(f" tts result {type(result)}")
                    tts_data.set_audio(result)
                    # 処理したデータをq1に送る
                    await self.emit_manager.put(tts_data)
                    self._keep_status()
                    await asyncio.sleep(0.05)
                # タスク完了を通知
                self.tts_queue.task_done()
            except (asyncio.CancelledError, asyncio.TimeoutError, EOFError, KeyboardInterrupt, SystemExit) as ex:
                self.logger.debug(f"task tts cancelled {ex}")
                break
            except Exception as e:
                self.logger.exception(f"Error in process_tts: {e}")
                continue