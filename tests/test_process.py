import sys,os
sys.path.insert(0,'.')
sys.path.insert(0,'fastrtc_jp')
import unittest
import numpy as np
from numpy.typing import NDArray
import asyncio
from fastrtc.text_to_speech.tts import TTSModel, TTSOptions
from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc_jp.handler.service import STTService, TTSService


def encode_text( text:str ) ->NDArray[np.float32]:
    text_bytes = text.encode("utf-8")
    text_len = len(text_bytes)
    # 4バイトで長さを格納し、その後にバイナリデータを格納
    header_np = np.array([text_len], dtype=np.int32).view(np.uint8)
    text_np = np.frombuffer(text_bytes, dtype=np.uint8)
    # 先頭に長さとバイナリを詰める
    result = np.concatenate([header_np, text_np])
    return result.astype(np.float32).reshape(1, -1)

def decode_text( audio:NDArray[np.float32] ) ->str:
    audio_uint8 = audio.astype(np.uint8).reshape(-1)
    text_len = np.frombuffer(audio_uint8[:4], dtype=np.int32)[0]
    text_bytes = audio_uint8[4:4+text_len].tobytes()
    return text_bytes.decode("utf-8")

class DummySTTModel(STTModel):

    def stt(self, frame:tuple[int,NDArray]):
        sr, audio = frame
        return decode_text(audio)

def get_dummy_stt_model():
    return DummySTTModel()

class DummyTTSModel(TTSModel):
    def tts(self, text:str, options):
        audio = encode_text(text)
        return (24000,audio)

    async def stream_tts(self, text, options):
        # Dummy async generator
        yield self.tts(text,options)

    def stream_tts_sync(self, text, options):
        # Dummy generator
        yield self.tts(text,options)

def get_dummy_tts_model():
    return DummyTTSModel()

class TestProcessServices(unittest.IsolatedAsyncioTestCase):
    async def test_stt_service(self):
        for num in range(10):
            expected:str = f"test{num:08}"
            audio = encode_text(expected)
            dbg = decode_text(audio)
            print(dbg)
            frame=(16000,audio)
            service = STTService(get_dummy_stt_model)
            result = await service.stt(frame)
            self.assertEqual(result, expected)
        service.shutdown()

    async def test_tts_service(self):
        text = "test"
        options = {}
        expected_sr = 24000
        expected_audio = encode_text(text)
        service = TTSService(get_dummy_tts_model)
        sample_rate, audio = await service.tts(text, options)
        self.assertEqual(sample_rate, expected_sr)
        self.assertIsInstance(audio, np.ndarray)
        np.testing.assert_array_equal(audio, expected_audio)
        service.shutdown()

if __name__ == "__main__":
        sys.path.insert(0,'.')
        sys.path.insert(0,'fastrtc_jp')
        expected:str = "test0123"
        audio = encode_text(expected)
        dbg = decode_text(audio)
        print(dbg)
