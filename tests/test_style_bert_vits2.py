import sys,os,asyncio
from pathlib import Path
sys.path.insert(0,'./')
from dotenv import load_dotenv
from fastrtc_jp.text_to_speech.style_bert_vits2 import StyleBertVits2,StyleBertVits2Options
from utils import play_audio

def test_buildin_model(model:str|None=None,speaker_id:int|None=None):
    tts = StyleBertVits2()
    text = f"こんにちは、{model}です。今日も良い天気ですね。"
    options:StyleBertVits2Options|None = StyleBertVits2Options(model=model,speaker_id=speaker_id) if model else None
    print(f"test model:{model}")
    for sample_rate,audio in tts.stream_tts_sync(text,options=options):
        print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
        play_audio(audio,sample_rate)

def test_local_model():
    # https://github.com/litagin02/Style-Bert-VITS2/blob/master/library.ipynb
    # BERTモデルをロード（ローカルに手動でダウンロードする必要はありません）
    # model_assetsディレクトリにダウンロードされます
    tmpdir=os.path.join('tmp','model_assets')
    os.makedirs(tmpdir, exist_ok=True) 
    assets_root = Path(tmpdir)

    model_file = "jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors"
    config_file = "jvnv-M1-jp/config.json"
    style_file = "jvnv-M1-jp/style_vectors.npy"

    options:StyleBertVits2Options = StyleBertVits2Options( device="cpu",
        model_path= assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
    )
    tts = StyleBertVits2()
    text = "こんにちは、今日も良い天気ですね。"
    for sample_rate,audio in tts.stream_tts_sync(text,options=options):
        print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
        play_audio(audio,sample_rate)

def test():

    tts = StyleBertVits2()
    option=None
    text = "こんにちは、今日も良い天気ですね。"

    print("test stream_tts")
    async def test_async():
        async for sample_rate,audio in tts.stream_tts(text,option):
            print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
            play_audio(audio,sample_rate)
    asyncio.run(test_async())

    print("test stream_tts_sync")
    for sample_rate,audio in tts.stream_tts_sync(text,option):
        print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
        play_audio(audio,sample_rate)

    print("test tts")
    sample_rate,audio = tts.tts(text,option)
    print(f"  audio: sr:{sample_rate} {audio.shape} {audio.dtype}")
    play_audio(audio,sample_rate)

    return audio

if __name__ == "__main__":
    test_buildin_model('rinne')
    test_buildin_model('girl')
    test_buildin_model('girl',1)
    test_buildin_model('girl',2)
    test_buildin_model('girl',3)
    test_buildin_model('girl',4)
    test_buildin_model('AbeShinzo')
    test_buildin_model('sakura-miko')
    #download_jvnv_F1_jp('tsukuyomi-chan')
    test_buildin_model('jvnv-F1-jp')
    test_buildin_model('jvnv-F2-jp')
    test_buildin_model('jvnv-M1-jp')
    test_buildin_model('jvnv-M2-jp')
    test_buildin_model()
    test()