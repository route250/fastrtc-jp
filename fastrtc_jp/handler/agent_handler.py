
from abc import ABC, abstractmethod
from functools import lru_cache
from logging import getLogger
from typing import AsyncGenerator
from fastrtc.text_to_speech.tts import TTSModel, TTSOptions
from fastrtc_jp.handler.session import AgentSession
from fastrtc_jp.text_to_speech.opt import SpkOptions

class AgentHandler(ABC):

    @abstractmethod
    def copy(self) ->"AgentHandler": ...

    @abstractmethod
    async def start_up(self): ...

    @abstractmethod
    def shutdown(self): ...

    @lru_cache(maxsize=1)
    def get_profile_list(self) -> dict[str,SpkOptions]:
        from fastrtc_jp.text_to_speech.voicevox import get_voicevox_options_list
        from fastrtc_jp.text_to_speech.style_bert_vits2 import get_sbv2_options_list
        from fastrtc_jp.text_to_speech.gtts import get_gtts_options_list
        m:dict[str,SpkOptions] = {}
        m.update(get_voicevox_options_list())
        m.update(get_sbv2_options_list())
        m.update(get_gtts_options_list())
        return m

    #Ovverride
    def get_tts_options(self, profile_name:str) -> SpkOptions:
        map = self.get_profile_list()
        opts:SpkOptions|None = map.get(profile_name)
        if opts is None:
            from fastrtc_jp.text_to_speech.gtts import GTTSOptions
            opts = GTTSOptions()
        return opts

    @abstractmethod
    async def start_session(self, agent_id:str, user_id:str, session_id:str, profile ) -> AgentSession: ...

    @abstractmethod
    async def before_run(self, session:AgentSession) -> None: ...

    @abstractmethod
    async def run(self, session:AgentSession, user_input:str|None) -> AsyncGenerator[str,None]: ...

    @abstractmethod
    async def commit(self, session:AgentSession, output_text:str|None, replace_text:str|None ) -> None: ...

    @abstractmethod
    async def rollback(self, session:AgentSession) -> None: ...

    @abstractmethod
    async def end_session(self, session:AgentSession) -> None: ...
