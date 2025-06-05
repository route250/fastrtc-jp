
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from fastrtc.text_to_speech.tts import TTSModel, TTSOptions
from fastrtc_jp.handler.session import AgentSession
from fastrtc_jp.text_to_speech.opt import SpkOptions

class AgentHandler(ABC):

    def copy(self) ->"AgentHandler":
        return self

    async def start_up(self):
        pass

    def shutdown(self):
        pass

    @abstractmethod
    def get_profile_list(self) -> dict[str,SpkOptions]: ...

    @abstractmethod
    def get_tts_options(self, profile:str) -> SpkOptions: ...

    @abstractmethod
    async def start_session(self, session:AgentSession, profile ) -> AgentSession: ...

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
