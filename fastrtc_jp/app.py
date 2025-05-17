import sys,os
from pathlib import Path
import platform
from typing import Type, AsyncGenerator

sys.path.insert(0, '.venv/lib/python3.12/site-packages')
sys.path.insert(0, '.venv/lib/python3.11/site-packages')
sys.path.insert(0, '.')

from datetime import datetime
import numpy as np
from numpy.typing import NDArray

from logging import getLogger
from fastrtc.reply_on_pause import AlgoOptions 
from fastrtc.stream import Stream

import gradio as gr
from gradio.components import ChatMessage
from fastrtc import AdditionalOutputs, WebRTC

from fastrtc import get_silero_model, ReplyOnPause
from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.text_to_speech.tts import TTSModel, TTSOptions

from fastrtc_jp.handler.agent_driver import AgentSession, AgentDriver
from fastrtc_jp.speech_to_text.sr_google import GoogleSTT
from fastrtc_jp.text_to_speech.gtts import GTTSModel
from fastrtc_jp.handler.stream_handler import AsyncVoiceStreamHandler
from fastrtc_jp.utils.util import load_dotenv, setup_logger
from fastrtc_jp.handler.vad import VadOptions

from fastrtc_jp.handler.dummy import dummy_response

logger = getLogger(__name__)


vadmodel = get_silero_model()
def get_vad(frame:tuple[int, NDArray[np.float32]] ) -> float:
    value,_ = vadmodel.vad(frame,None)
    return value

def get_stt_model() -> STTModel:
    return GoogleSTT()

def get_tts_model() -> TTSModel:
    return GTTSModel()

class DummyDriver(AgentDriver):
    def __init__(self,):
        pass

    #Override
    def copy(self) ->"AgentDriver":
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
    async def start_session(self, session:AgentSession) -> AgentSession:
        return session

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


def test_speech_gr():
    loggerx = getLogger("handler.speech_handler")
    loggerx.setLevel("DEBUG")

    with gr.Blocks(fill_height=True,fill_width=True) as demo:
        gr.HTML(
        """
        <h1 style='text-align: center'>
        Talk to Sample (Powered by WebRTC ⚡️)
        </h1>
        """
        )

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                with gr.Row(scale=15):
                    dmydata = gr.JSON(label="debug")
                with gr.Row(scale=1):
                    dropdown = gr.Dropdown(
                        label="Options",
                        choices=["Option 1", "Option 2", "Option 3"],
                        value="Option 1"
                    )
                with gr.Row(scale=1):
                    audio = WebRTC(label="Stream",mode="send-receive", modality="audio" )
            with gr.Column(scale=3):
                chat_area = gr.Chatbot(label="chat", type="messages")

        vad_options = VadOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.5,
            speech_threshold=0.1,
        )

        audio.stream(
            AsyncVoiceStreamHandler(
                DummyDriver(),
                vad_fn=get_vad,
                get_stt_model_fn=get_stt_model,
                get_tts_model_fn=get_tts_model,
                vad_options=vad_options,
            ),
            inputs=[audio,dropdown],
            outputs=[audio],
            time_limit=None
        )
        def process_outputs(*args) ->tuple:
            try:
                # print(f"追加出力の型: {type(args)}, 値: {args}",flush=True)
                
                if len(args)>=2:
                    if isinstance(args[1],list) and len(args[1])>0 and isinstance(args[1][0],dict):
                        hist = []
                        for m in args[1]:
                            if isinstance(m,dict):
                                role = m.get('role')
                                content = m.get('content')
                                if role and content:
                                    hist.append( ChatMessage(role=role,content=content))
                        datestr = datetime.now().strftime("%Y-%m-%d")
                        timestr = datetime.now().strftime("%H:%M:%S")
                        return {'date':datestr,'time':timestr},hist
            except Exception as ex:
                print(f"ERROR:{ex}",flush=True)
            return {},[ChatMessage(role='user', content='empty1')]
            
        audio.on_additional_outputs(
            process_outputs,
            outputs=[dmydata,chat_area],
            queue=True, show_progress="hidden"
        )
    
        demo.launch()

if __name__ == "__main__":
    load_dotenv()
    # os.environ["AGNO_MONITOR"] = "false"
    # os.environ["AGNO_TELEMETRY"] = "false" # テレメトリを無効化
    setup_logger()
    test_speech_gr()
