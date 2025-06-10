import sys,os,time,re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger

ROLE_USER='user'
ROLE_AI='assistant'

@dataclass
class AgentMessage:
    role:str
    content:str

    def to_dict(self):
        return {'role': self.role, 'content': self.content }

class AgentSession(ABC):

    def __init__(self, agent_id:str, user_id:str, session_id:str ):
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id

    @abstractmethod
    def add_user(self, content:str ): ...

    @abstractmethod
    def add_ai(self, content:str ): ...

    @abstractmethod
    def make_input(self, user_input:str) ->str: ...

    @abstractmethod
    async def get_messages(self) -> list[dict]: ...

    @abstractmethod
    async def commit(self, output_text:str|None, replace_text:str|None ) -> None: ...

    @abstractmethod
    async def rollback(self) -> None: ...
    
class AgentSessionABC(AgentSession):
    logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(self, agent_id:str, user_id:str, session_id:str ):
        super().__init__(agent_id, user_id, session_id)
        self.hist:list[AgentMessage] = []

    def add_user(self, content:str ):
        if len(self.hist)>0 and self.hist[-1].role==ROLE_USER:
            self.hist[-1].content+=content
        else:
            self.hist.append( AgentMessage(ROLE_USER,content))

    def add_ai(self, content:str ):
        if len(self.hist)>0 and self.hist[-1].role==ROLE_AI:
            self.hist[-1].content+=content
        else:
            self.hist.append( AgentMessage(ROLE_AI,content))

    def make_input(self, user_input:str) ->str:
        prompt_array:list[str] = []
        for x in self.hist:
            print(f"{x.role}: {x.content}")
            prompt_array.append( f"{x.role}: {x.content}" )
        prompt_array.append( f"{ROLE_USER}: {user_input}" )
        return "\n\n".join(prompt_array)

    async def get_messages(self) -> list[dict]:
        return [ m.to_dict() for m in self.hist ]

    async def commit(self, output_text:str|None, replace_text:str|None ) -> None:
        if len(self.hist)>0 and self.hist[-1].role==ROLE_AI and self.hist[-1].content==output_text:
            if replace_text:
                self.hist[-1].content=replace_text
            else:
                self.hist.pop(-1)

    async def rollback(self) -> None:
        if len(self.hist)>0 and self.hist[-1].role==ROLE_AI:
            self.hist.pop(-1)
        if len(self.hist)>0 and self.hist[-1].role==ROLE_USER:
            self.hist.pop(-1)
