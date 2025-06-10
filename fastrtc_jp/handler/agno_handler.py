from functools import lru_cache
import sys,os
sys.path.insert(0, '.')
from typing import Any, AsyncGenerator
from dotenv import load_dotenv
import asyncio

from fastrtc_jp.handler.agent_handler import AgentHandler
from fastrtc_jp.handler.session import AgentSession
from fastrtc_jp.text_to_speech.gtts import GTTSOptions
from fastrtc_jp.text_to_speech.opt import SpkOptions
from fastrtc_jp.text_to_speech.tts_provider import TtsProvider


from agno.agent import Agent, RunResponse
from agno.run.team import TeamRunResponse
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.agent import AgentRun
from agno.models.openai import OpenAIChat
from agno.models.message import Message
from agno.run.response import RunEvent
from agno.storage.sqlite import SqliteStorage
from agno.storage.session.agent import AgentSession as agno_AgentSession

from logging import getLogger
logger = getLogger(__name__)


def make_agent( *, db_file:str|None = None, agent_id:str|None=None, session_id:str|None = None) ->Agent:
    db_memory = None
    db_storage = None
    if db_file:
        db_memory = Memory(
            model=OpenAIChat(id="gpt-4.1-nano"),
            db=SqliteMemoryDb(table_name="user_memories", db_file=db_file),
        )
        db_storage=SqliteStorage(table_name="agent_sessions", db_file=db_file)

    agent = Agent(
        agent_id=agent_id,
        session_id=session_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        memory=db_memory,
        storage=db_storage,
        add_history_to_messages=True,
        num_history_runs=5,
        stream_intermediate_steps=True,
        telemetry=False,
    )
    return agent

def update_message( message:Message|None,a,b):
    if isinstance(message,Message):
        if message.content==a:
            print(f"    update_message {a} -> {b}")
            message.content=b

def update_message_list( messages: list[Message]|None, a,b ):
    if isinstance(messages,list):
        for m in messages:
            update_message(m,a,b)

def update_run_response(run_res:RunResponse|TeamRunResponse, a:str, b:str):
    if isinstance(run_res,RunResponse):
        if( run_res.content == a ):
            run_res.content = b
            update_message_list(run_res.messages,a,b)

def update_agent_runs(agent:Agent|None, a, b):
    if agent is None or agent.session_id is None:
        return
    if a is None or a==b:
        return
    if isinstance(agent.memory ,Memory):
        if isinstance(agent.memory.runs ,dict):
            runs = agent.memory.runs.get(agent.session_id)
            if isinstance(runs, list) and len(runs)>0:
                update_run_response(runs[-1],a,b)
    if agent.storage:
        ss = agent.storage.read(session_id=agent.session_id)
        if isinstance(ss, agno_AgentSession) and ss.memory:
            mm:list[dict[str,Any]] = ss.memory.get('runs',[{'messages':[]}])[-1]['messages']
            if len(mm)>0 and mm[-1]['role'] == 'assistant':
                print(f"    update_storage {a} -> {b}")
                mm[-1]['content'] = b
                agent.storage.upsert(ss)

def rollback_agent_last_run(agent:Agent|None):
    if agent is None or agent.session_id is None:
        return
    if isinstance(agent.memory ,Memory):
        if isinstance(agent.memory.runs ,dict):
            runs = agent.memory.runs.get(agent.session_id)
            if isinstance(runs, list) and len(runs)>0:
                x = runs.pop(-1)
                print(f"    rollback message ")
    if agent.storage:
        ss = agent.storage.read(session_id=agent.session_id)
        if isinstance(ss, agno_AgentSession) and ss.memory:
            runs = ss.memory.get('runs')
            if isinstance(runs,list) and len(runs)>0:
                x = runs.pop(-1)  # Remove the last run
                print(f"    rollback storage")
                agent.storage.upsert(ss)


class AgnoSession(AgentSession):
    def __init__(self, agent_id:str, user_id:str, session_id:str, db_file:str|None = None):
        super().__init__(agent_id=agent_id, user_id=user_id, session_id=session_id)
        self.db_file:str|None = db_file
        self.agent:Agent|None = None
        self.agno_session_id:str|None = None

    def get_agent(self) -> Agent:
        if self.agent is None:
            self.agent = make_agent(db_file=self.db_file,agent_id=self.agent_id,session_id=self.session_id)
        return self.agent

    def _read_session(self) -> agno_AgentSession|None:
        agent:Agent = self.get_agent()
        if agent.session_id and agent.storage:
            ss = agent.storage.read(session_id=agent.session_id)
            if isinstance(ss,agno_AgentSession):
                return ss
        return None

    async def _write_session(self, session:agno_AgentSession) -> None:
        agent:Agent = self.get_agent()
        if agent.session_id and agent.storage:
            agent.storage.upsert(session)
        else:
            logger.warning("Cannot write session, agent storage is not set up.")

    def _pull_id(self):
        if self.agent:
            if self.agent.agent_id and self.agent_id is None:
                self.agent_id = self.agent.agent_id
            if self.agent.session_id and self.agno_session_id is None:
                self.agno_session_id = self.agent.session_id

    def add_user(self, content:str ):
        pass

    def add_ai(self, content:str ):
        pass


    def make_input(self, user_input:str) ->str:
        # prompt_array:list[str] = []
        # for x in self.hist:
        #     print(f"{x.role}: {x.content}")
        #     prompt_array.append( f"{x.role}: {x.content}" )
        # prompt_array.append( f"{ROLE_USER}: {user_input}" )
        # return "\n\n".join(prompt_array)
        return user_input

    async def get_messages(self) -> list[dict]:
        ss:agno_AgentSession|None = self._read_session()
        if isinstance(ss, agno_AgentSession) and ss.memory:
            mm = ss.memory.get('runs',[{'messages':[]}])[-1]['messages']
            return [ {'role': m['role'], 'content': m['content']} for m in mm ]
        return []

    async def commit(self, output_text:str|None, replace_text:str|None ) -> None:
        update_agent_runs(self.agent, output_text, replace_text)

    async def rollback(self):
        rollback_agent_last_run(self.agent)

class AgnoHander(AgentHandler):

    def __init__(self, db_file:str|None = None):
        self.db_file:str|None = None
        if db_file:
            db_dir = os.path.dirname(db_file)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
                if os.path.isdir(db_dir):
                    logger.info(f"Using existing directory for database: {db_dir}")
                    self.db_file = db_file
        self.session_map:dict[str,AgentSession] = {}

    def copy(self) ->"AgnoHander":
        return AgnoHander(db_file=self.db_file)

    async def start_up(self):
        pass

    def shutdown(self):
        pass


    #Ovrride
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
            logger.warning(f"get_tts_options: No options found for {profile_name}, using default.")
            opts = GTTSOptions()
        return opts


    async def start_session(self, agent_id:str, user_id:str, session_id:str, profile:str|None ) -> AgentSession:
        return AgnoSession(agent_id, user_id, session_id, db_file=self.db_file)

    async def before_run(self, session:AgentSession) -> None:
        pass

    async def run(self, session:AgentSession, user_input:str|None) -> AsyncGenerator[str,None]:
        if not isinstance(session, AgnoSession):
            raise TypeError(f"Expected AgnoSession, got {type(session)}")
        agent:Agent = session.get_agent()
        print("-------------------")
        print(f"input:{user_input}")
        ai_response = ""
        res_itr = agent.run( user_input, stream=True)
        session._pull_id()  # Ensure agent_id and session_id are set
        for run_res in res_itr:
            if run_res.event==RunEvent.run_response:
                delta:str = str(run_res.content)
                ai_response += delta
                yield delta
        print(f"AI response: {ai_response}")

    async def commit(self, session:AgentSession, output_text:str|None, replace_text:str|None ) -> None:
        if session is not None and output_text is not None and output_text != replace_text:
            await session.commit(output_text, replace_text)

    async def rollback(self, session:AgentSession) -> None:
        await session.rollback()

    async def end_session(self, session:AgentSession) -> None:
        pass

async def test_main():
    import sys,os
    tmppath = "./tmp/agno/agno.db"
    os.makedirs( os.path.dirname(tmppath), exist_ok=True)
    if os.path.exists(tmppath):
        os.remove(tmppath)

    user_id="test_user"
    agent_id="test_agent"
    profile = "default"

    session_list = ["normal-session","replace_session","rollback_session"]
    input_list = ("こんにちは","今日は何日？","さっきなんて言ったの？")
    for s, test_id in enumerate(session_list):
        print(f"----\n Test ID: {test_id}\n ----")
        session_id = f"ses_{test_id}"
        hdr:AgnoHander = AgnoHander(db_file=tmppath)

        try:
            await hdr.start_up()
            ses = await hdr.start_session(agent_id, user_id, session_id, profile)
            try:
                for i,user_input in enumerate(input_list):
                    await hdr.before_run(ses)                
                    ai_response = ""
                    res_itr = hdr.run(ses, user_input)
                    async for run_res in res_itr:
                        ai_response += run_res
                    if i==1:
                        if s==1:
                            await hdr.commit(ses,ai_response,"あいうえお")
                        elif s==2:
                            await hdr.rollback(ses)
                    else:
                        await hdr.commit(ses, ai_response, ai_response)
            finally:
                await hdr.end_session(ses)
            

            print("================")
            ses2: AgnoSession = AgnoSession(ses.agent_id,ses.user_id,ses.session_id, db_file=tmppath)
            mesgs2 = await ses2.get_messages()
            print(f"Messages in session {ses2.session_id}:")
            for msg in mesgs2:
                print(f"{msg}")

        finally:
            hdr.shutdown()


if __name__ == "__main__":
    load_dotenv("config.env")
    asyncio.run(test_main())