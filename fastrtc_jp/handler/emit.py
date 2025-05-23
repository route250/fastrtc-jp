
import asyncio
import time
from fastrtc import AdditionalOutputs, wait_for_item
from fastrtc_jp.handler.voice import TtsAudio
from logging import getLogger

class EmitManager:

    logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(self):
        self._pause:bool = False
        self.emit_queue:asyncio.Queue[TtsAudio] = asyncio.Queue()
        self.ads_queue:asyncio.Queue[AdditionalOutputs] = asyncio.Queue()
        self.emit_data:TtsAudio|None = None
        self.emit_time:float = 0

    def set_pause(self, b:bool ):
        self._pause = b

    async def ads(self, data:AdditionalOutputs ):
        self.ads_queue.put_nowait(data)

    async def put(self, tts_data:TtsAudio ):
        print(f"put to emit_queue {tts_data.no} {tts_data.ai_response}")
        self.emit_queue.put_nowait(tts_data)

    async def _seek_next(self):
        if self.emit_data is not None and self.emit_data.is_done():
            print(f"emit done {self.emit_data.no} {self.emit_data.ai_response}")
            if self.emit_data.is_accepted():
                print(f"emit accepted {self.emit_data.no} {self.emit_data.ai_response}")
                await self.ads_queue.put( AdditionalOutputs([], self.emit_data.get_messages()))
            self.emit_data = None
        while self.emit_data is None or self.emit_data.is_canceled():
            if self.emit_data is not None:
                print(f"emit cancel {self.emit_data.no} {self.emit_data.ai_response}")
                await self.ads_queue.put( AdditionalOutputs([], self.emit_data.get_messages()))
            self.emit_data = await wait_for_item(self.emit_queue,0.01)
            if self.emit_data is None:
                break

    async def get_emit_segment(self):
        ads:AdditionalOutputs|None = await wait_for_item(self.ads_queue,0.01)
        if ads is not None:
            print("[EMIT] additional outputs")
            return ads

        await self._seek_next()

        if self.emit_data is None:
            return None
        else:
            if self.emit_data.is_accepted():
                print(f"emit accepted {self.emit_data.no} {self.emit_data.ai_response}")
                await self.ads_queue.put( AdditionalOutputs([], self.emit_data.get_messages()))

        if self._pause:
            return None

        now:float = time.time()
        if now < self.emit_time-0.1:
            return None
        if self.emit_time<now:
            self.emit_time = now

        if self.emit_data.pos==0:
            print(f"emit start {self.emit_data.no} {self.emit_data.ai_response}")

        emit_interval = 0.6
        emit_segment = self.emit_data.get_emit_data(emit_interval)
        if emit_segment is None:
            print(f"emit break???? {self.emit_data.no} {self.emit_data.ai_response}")
            return None
 
        emit_sec = emit_segment[1].shape[1]/emit_segment[0]
        self.emit_time += emit_sec
        # print(f"exit {al}")
        return emit_segment
