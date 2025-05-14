
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from fastrtc_jp.handler.session import AgentSession


class AgentDriver(ABC):

    def copy(self) ->"AgentDriver":
        return self

    def reset(self):
        pass

    async def start_up(self):
        pass

    def shutdown(self):
        pass

    @abstractmethod
    async def start_session(self, session:AgentSession) -> AgentSession: ...

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
