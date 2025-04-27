import sys,os,asyncio
#sys.path.insert(0,'./')
from dotenv import load_dotenv
from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel
from utils import play_audio

def test():
    if os.path.exists('config.env'):
        load_dotenv('config.env')
    elif os.path.exists('.env'):
        load_dotenv('.env')
    hostlist:str|None = os.getenv('VOICEVOX_HOSTLIST')
    tts = VoicevoxTTSModel(hostlist)
    text = "こんにちは、今日も良い天気ですね。"

    print("test stream_tts")
    async def test_async():
        async for sample_rate,audio in tts.stream_tts(text):
            print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
            play_audio(audio,sample_rate)
    asyncio.run(test_async())

    print("test stream_tts_sync")
    for sample_rate,audio in tts.stream_tts_sync(text):
        print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
        play_audio(audio,sample_rate)

    print("test tts")
    sample_rate,audio = tts.tts(text)
    print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
    play_audio(audio,sample_rate)

    return audio

if __name__ == "__main__":
    test()