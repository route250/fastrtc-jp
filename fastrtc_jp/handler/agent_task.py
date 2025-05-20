import asyncio
from enum import Enum
import re
import traceback
from dataclasses import dataclass
from logging import getLogger
from typing import Any, AsyncIterator, AsyncIterable, Iterable, Protocol

from numpy.typing import NDArray

from fastrtc_jp.handler.agent_handler import AgentHandler
from fastrtc_jp.handler.session import AgentMessage, AgentSession

import typing
if typing.TYPE_CHECKING:
    from fastrtc_jp.handler.voice import SttAudio

def stream_split_ja(text:str, idx:int) -> tuple[str|None,str]:
    """文章の分割を判定する。idx==0の時は短めに判定する"""
    regex0=r'[!?、。？！\n]+\s*'
    regex1=r'[!?。？！\n]+\s*'
    regex = regex1 if idx<=0 else regex0
    x = re.search(regex, text)
    p = x.span()[1] if x else len(text)
    if p < len(text):
        return text[:p], text[p:]
    else:
        return None,text


@dataclass
class SeassionObj:
    agent_id:str
    user_id:str
    session_id:str
    obj:Any|None = None

class AgentStat(Enum):
    NOT_START = 0
    RUNNING = 1
    PLAYING = 2
    DONE = 3
    CANCEL = 4


class AgentTask:

    logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(self, session:AgentSession, driver:AgentHandler, stt:list['SttAudio']):
        self.session:AgentSession = session
        self.driver:AgentHandler = driver
        self.stt:list['SttAudio'] = stt
        self.stat:AgentStat = AgentStat.NOT_START
        # 生成した文字列セグメントの数
        self.seg_total:int = 0
        # 再生完了した数
        self.accepted:int = 0
        # 再生が完了したテキスト
        self.ai_response:str = ""
        # 完了フラグ
        self._cancel:bool = False
        self.done_play:asyncio.Event = asyncio.Event()
        self.done_running:asyncio.Event = asyncio.Event()

    def accept(self,ai_response:str):
        """再生が完了した文章をhistoryに追加する"""
        if self.is_canceled():
            return
        if self.accepted<=0:
            # 初回は、user_inputも追加する
            for s in self.stt:
                if s.user_input:
                    self.session.add_user(s.user_input)
        self.session.add_ai(ai_response)
        self.ai_response += ai_response
        self.accepted += 1
        if self.stat==AgentStat.PLAYING and self.seg_total<=self.accepted:
            print(f"[AGENT] accept {self.accepted}/{self.seg_total} done")
            self.done_play.set()
        else:
            print(f"[AGENT] accept {self.accepted}/{self.seg_total}")

    def cancel(self):
        if self.done_play.is_set():
            if self._cancel:
                print(f"[AGENT] ignore canceled")
            else:
                print(f"[AGENT] already done")
            return
        print(f"[AGENT] cancel")
        self._cancel = True
        self.done_play.set()

    def is_canceled(self) ->bool:
        return self.done_play.is_set()

    async def execute(self):
        """AGENTを実行する"""
        if self.is_canceled():
            print(f"[AGENT] cancelled")
            self.done_running.set()
            return
        try:
            self.stat = AgentStat.RUNNING
            ses:AgentSession = self.session
            await self.driver.before_run(ses)
            committed = False
            # llmへの入力を作成する
            user_input:str = "\n".join( [ a.user_input for a in self.stt if a.user_input] )
            prompt_str:str = self.session.make_input( user_input )
            print(f"[AGENT] hist")
            print(f"{prompt_str.replace("\n\n","\n")}")
            # llmを実行 レスポンスを非同期で取得する
            full_content:str=""
            buffer:str=""
            idx:int=0
            print(f"[AGENT] start")
            async for run_res in self.driver.run(ses, user_input): # type: ignore
                full_content += run_res
                if not self.is_canceled():
                    buffer+=run_res
                    segment,buffer = stream_split_ja(buffer,idx)
                    if segment:
                        self.seg_total+=1
                        yield segment
                        await asyncio.sleep(0.1)
                        idx+=1
            if not self.is_canceled() and buffer:
                self.seg_total+=1
                yield buffer
            self.stat = AgentStat.PLAYING
            if self.seg_total>self.accepted and not self.is_canceled():
                print(f"[AGENT] wait")
                await self.done_play.wait()
            else:
                self.done_play.set()
            if self.seg_total>0:
                if not self.is_canceled():
                    print(f"[AGENT] commit")
                    await self.driver.commit(ses,full_content, self.ai_response)
                    committed = True
                else:
                    print(f"[AGENT] cancel")
            else:
                print(f"[AGENT] no output")
        
        except Exception as ex:
            traceback.print_exc()

        finally:
            if not committed:
                # なかったことにする
                print(f"[AGENT] rollback")
                await self.driver.rollback(ses)
            print(f"[AGENT] done")
            self.stat = AgentStat.DONE
            self.done_running.set()
    
    def get_messages(self) -> list[dict]:
        return self.session.get_messages()
