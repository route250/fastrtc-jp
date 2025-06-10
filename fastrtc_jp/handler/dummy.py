
import asyncio
from functools import lru_cache
import sys,os,time,re
import random
from datetime import datetime
from dataclasses import dataclass
from logging import getLogger
from typing import AsyncGenerator, AsyncIterator, Protocol
from fastrtc_jp.handler.agent_handler import AgentHandler
from fastrtc_jp.handler.session import AgentSession, AgentSessionABC
from fastrtc_jp.text_to_speech.gtts import GTTSOptions
from fastrtc_jp.text_to_speech.opt import SpkOptions
from fastrtc_jp.utils.util import to_lang_code

async def dummy_response(user_input:str, language:str='ja') -> AsyncGenerator[str,None]:
    ai_response:str = make_dummy_response(user_input,language)
    for char in ai_response:
        await asyncio.sleep(0.1)
        yield char

def make_dummy_response(text: str, language:str='ja') ->str:
    now = datetime.now()
    lang:str = to_lang_code(language)
    if lang=='ja':
        date_str = now.strftime("%m月%d日")
        time_str = now.strftime("%p").replace("AM", "午前").replace("PM", "午後") + now.strftime("%I時%M分").lstrip("0")
        beginnings = [
            "ただいま動作テスト中です。",
            "ダミー応答を返します。",
            "テスト応答を生成しました。",
            "これはテストメッセージです。",
            "こんにちは、これは確認用の応答です。",
            "テスト起動中です。"
        ]

        bodies = [
            f"認識された内容は {text} です。現在の日時は {date_str} {time_str}。",
            f"{text} と認識されました。時刻は {time_str}。",
            f"認識内容 {text} ただいまの時刻は {time_str} です。",
            f"あなたの音声は {text} と認識しました。記録時刻は{date_str}。",
            f"認識結果 {text} が入力されました。現在時刻は{time_str}。",
            f"認識したテキストは {text} 日付は{date_str}、時刻は{time_str}です。"
        ]

        closings = [
            "おわり。",
            "テスト完了。",
            "以上です。",
            "確認終了。",
            "おしまい。",
            "テストおわり。"
        ]

        extras = [
            "本日は晴天なり。",
            "マイクのテスト中です。",
            "ただいま通信チェックを行っています。",
            "これは訓練ではありません。",
            "周囲の音にご注意ください。",
            "これはテスト放送です。",
            "声は届いていますか？",
            "入力は正常に処理されました。"
        ]
    else:
        date_str = now.strftime("%b %d")
        time_str = now.strftime("%p").replace("AM", "AM").replace("PM", "PM") + now.strftime(" %I:%M").lstrip("0")
        beginnings = [
            "Hi, I'm just checking things.",
            "Hello, this is a quick system test.",
            "Hey, let's see if everything works.",
            "Testing, testing.",
            "Just making sure I'm online.",
            "Let's do a quick check."
        ]

        bodies = [
            f"I heard you say {text}.",
            f"You said {text}.",
            f"Okay, I got {text}.",
            f"It sounds like you said {text}.",
            f"{text}, right?",
            f"That's what I heard: {text}."
        ]

        closings = [
            "That's all for now.",
            "Test finished.",
            "All set.",
            "Looks good.",
            "Everything's working.",
            "Done for now."
        ]

        extras = [
            "Hope you're having a good day.",
            "Just making sure the mic is working.",
            "Everything seems fine here.",
            "No worries, this is only a test.",
            "Let me know if you hear me clearly.",
            "This is just a quick check.",
            "Thanks for helping with the test.",
            "All systems are go."
        ]

    extra_phrase = random.choice(extras) if random.random() < 0.5 else ""

    return f"{random.choice(beginnings)} {random.choice(bodies)} {extra_phrase} {random.choice(closings)}"


class DummyAgentHandler(AgentHandler):

    logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(self,):
        pass

    #Override
    def copy(self) ->"AgentHandler":
        return self

    #Override
    def reset(self):
        pass

    #Override
    async def start_up(self):
        pass

    #Override
    def shutdown(self):
        pass

    #Override
    async def start_session(self, agent_id:str, user_id:str, session_id:str, profile) -> AgentSession:
        return AgentSessionABC(agent_id,user_id,session_id)

    #Override
    async def before_run(self, session:AgentSession) -> None:
        pass

    #Override
    async def run(self, session:AgentSession, user_input:str|None) -> AsyncGenerator[str,None]:
        if user_input:
            async for aa in dummy_response(user_input):
                yield aa

    #Override
    async def commit(self, session:AgentSession, output_text:str|None, replace_text:str|None ) -> None:
        pass

    #Override
    async def rollback(self, session:AgentSession) -> None:
        pass

    #Override
    async def end_session(self, session:AgentSession) -> None:
        pass

