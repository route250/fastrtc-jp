import sys,os
sys.path.insert(0,'./')
import re
import json
import logging
import time
from typing import BinaryIO
from io import BytesIO
import numpy as np
import requests
from requests.exceptions import HTTPError
import wave
from fastrtc_jp.speech_to_text.sr_google import GoogleSTT
from fastrtc_jp.speech_to_text.mlx_whisper import MlxWhisper

logger = logging.getLogger(__name__)

class GitHubRepo:
    def __init__(self, cache_dir, owner,repo,branch,*, retry:int=2):
        self._cache_dir = cache_dir
        self._owner = owner
        self._repo = repo
        self._branch = branch
        self._retry:int = retry

    def _do_get(self,api_url) ->requests.Response:
        for r in range(self._retry+1):
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                return response
            except HTTPError as ex:
                if r<self._retry and 'rate limit exceeded' in str(ex):
                    print(f" wait {api_url}")
                    time.sleep(30)
                    continue
                raise ex
        raise HTTPError("")

    def list_files(self, path:str) -> list[str]:
        cache_path = os.path.join( self._cache_dir, 'files', path )
        if not os.path.exists(cache_path):
            api_url = f"https://api.github.com/repos/{self._owner}/{self._repo}/contents/{path}"
            if self._branch and self._branch!='main' and self._branch!='master':
                api_url = f'{api_url}?ref={self._branch}'
            response = self._do_get(api_url)
            items = response.json()
            # ディレクトリ内のファイル名一覧を取得
            file_list = [item['name'] for item in items if item['type'] == 'file']
            os.makedirs(os.path.dirname(cache_path),exist_ok=True)
            with open(cache_path,'w', encoding='utf-8') as f:
                json.dump(file_list,f,ensure_ascii=False, indent=2 )
            return file_list
        else:
            with open(cache_path, 'r', encoding='utf-8') as f:
                file_list = json.load(f)
            return file_list

    def download_file(self, path:str, out_stream:BinaryIO):
        cache_path = os.path.join( self._cache_dir, 'contents', path )
        if not os.path.exists(cache_path):
            api_url = f"https://api.github.com/repos/{self._owner}/{self._repo}/contents/{path}"
            if self._branch and self._branch!='main' and self._branch!='master':
                api_url = f'{api_url}?ref={self._branch}'
            response = self._do_get(api_url)
            data = response.json()
            download_url = data.get("download_url")
            if not download_url:
                raise Exception("download_url not found")
            file_response = self._do_get(download_url)
            os.makedirs(os.path.dirname(cache_path),exist_ok=True)
            with open(cache_path,'wb') as f:
                f.write(file_response.content)
        with open(cache_path,'rb') as f:
            out_stream.write(f.read())

    def read_text_file(self,path:str) ->str:
        buf = BytesIO()
        self.download_file(path, buf)
        return buf.getvalue().decode('utf-8')

def load_wave(path:str) -> tuple[int,np.ndarray]:
    with open(path, 'rb') as fp:
        return load_waveb(fp)

def load_waveb( fp:BinaryIO ) -> tuple[int,np.ndarray]:
    fp.seek(0)
    with wave.open(fp,'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio = wf.readframes(n_frames)
        dtype = np.int16 if wf.getsampwidth() == 2 else np.uint8
        data = np.frombuffer(audio, dtype=dtype)
        data = data.reshape(-1, wf.getnchannels()).T  # shape: (numch, length)
    return sample_rate, data.astype(np.int16)

def load_call_info(repo:GitHubRepo, file:str) -> dict|None:
    source = repo.read_text_file(file)
    text = source

    text = re.sub(r'^.*export const callNameInfos.*}\s*=\s*{', '{', text, flags=re.DOTALL)
    text = re.sub(r'}[\s;]*$', '}', text, flags=re.DOTALL)
    # キーを修正
    text = re.sub(r'(?<=\{|,)\s*([^\s:"]+)\s*:', r'"\1":', text)
    # 不要なカンマを削除
    text = re.sub(r'\s*,\s*}','}',text)
    text = re.sub(r'\s*,\s*]',']',text)
    try:
        data = json.loads(text)
        logger.info(f"loaded {file}\n{json.dumps(data,ensure_ascii=False,indent=2)}")
        return data
    except:
        logger.error(f"can not load {file}\n{source}\n----\n{text}")


def load_char_info(repo:GitHubRepo, file:str) -> dict|None:
    source = repo.read_text_file(file)
    text = source
    # const key = "藍田ノエル" satisfies CharacterKey;からkeyの文字列を抽出
    match = re.search(r'const\s+key\s*=\s*"([^"]+)"\s*satisfies\s+CharacterKey;', text)
    if not match:
        return None
    key = match.group(1) if match else ""
    # 'export default {'より前を'{'に置換
    text = '{' + text.split('export default {', 1)[1]
    # '} satisfies CharacterInfo'より後ろを}に置換
    text = text.split('} satisfies CharacterInfo', 1)[0] + '}'
    # key, を削除
    text = re.sub(r'\s*key\s*,\s*', f"key: \"{key}\",", text)
    # getCharacterAssetsを削除
    text = re.sub(r'\s*\.*getCharacterAssets\([^)]*\)\s*,\s*','',text)
    # キーを修正
    text = re.sub(r'(?<=\{|,)\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', text)
    # 不要なカンマを削除
    text = re.sub(r'\s*,\s*}','}',text)
    text = re.sub(r'\s*,\s*]',']',text)
    # <br/>を削除
    text = re.sub(r'<br\s*/>',' ',text)

    try:
        data = json.loads(text)
        logger.info(f"loaded {file}\n{json.dumps(data,ensure_ascii=False,indent=2)}")
        return data
    except Exception as ex:
        logger.error(f"can not load {file} {ex}\n{source}\n----\n{text}")

def main():

    output_dir = 'tmp/voicevox_data'
    cache_dir = os.path.join(output_dir,'cache')
    os.makedirs(cache_dir, exist_ok=True)
    json_file=os.path.join(output_dir,'voicevox_charinfo.json')
    log_file=os.path.join(output_dir,'voicevox_charinfo.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    repo:GitHubRepo = GitHubRepo( cache_dir, 'VOICEVOX','voicevox_blog','master')
    source_target='src/constants/characterInfos'
    source_callinfo=os.path.join(source_target,'callNameInfo.ts')
    audio_targets=('src/assets/talk-audios','src/assets/dormitory-audios')

    print(f"loading {source_callinfo}")
    call_info = load_call_info( repo, source_callinfo )
    if call_info is None:
        logger.info(f"can not loaded {source_callinfo}")
        return
    logger.info(f"loaded {source_callinfo}\n{json.dumps(call_info,ensure_ascii=False,indent=2)}")

    char_files = repo.list_files(source_target)
    char_info_list=[]
    for char_file in char_files:
        if char_file == source_callinfo or not char_file.endswith('.ts'):
            continue
        print(f"loading {char_file}")
        char_info = load_char_info( repo, f"{source_target}/{char_file}")
        if char_info is None:
            print(f"SKIP: {char_file}")
            logger.warning(f"skip {char_file}\n{json.dumps(char_info,ensure_ascii=False,indent=2)}")
            continue
        logger.info(f"loaded {char_file}\n{json.dumps(char_info,ensure_ascii=False,indent=2)}")
        name = char_info.get('key')
        if not name:
            print(f"ERRIR;not found name in {char_file}")
            logger.error(f"not found name in {char_file}")
            continue
        aaa = call_info.get(name)
        if not aaa:
            print(f"ERRIR;not found {name} in {source_target}")
            logger.error(f"not found {name} in {source_target}")
            continue
        me = aaa.get('me')
        if me:
            char_info['me'] = me
            del aaa['me']
        you = aaa.get('you')
        if you:
            char_info['you'] = you
            del aaa['you']
        char_info['callNameInfo'] = aaa
        char_info_list.append(char_info)

    print(f"save to {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(char_info_list, f, ensure_ascii=False, indent=2)

    stt = MlxWhisper(
        path_or_hf_repo='mlx-community/whisper-large-v3-turbo',
        language='ja',
    )
    # gstt = GoogleSTT()
    for audio_path in audio_targets:
        dir = os.path.join(output_dir,os.path.basename(audio_path))
        os.makedirs(dir,exist_ok=True)
        for wave_path in repo.list_files(audio_path):
            if not wave_path.endswith('.wav'):
                continue
            save_text = os.path.join(dir,wave_path.replace('.wav','.txt'))
            if 'talk-audios' in audio_path and 'normal' not in wave_path:
                if os.path.exists(save_text):
                    print(f"remove {save_text}")
                    os.remove(save_text)
                continue
            callinfo = None
            for info in char_info_list:
                if wave_path.startswith(info['id']):
                    print(f"found {wave_path} in {info['id']}")
                    callinfo = info
                    break
            if callinfo is None:
                print(f"not found {wave_path} in {audio_path}")
                logger.error(f"not found {wave_path} in {audio_path}")
                continue
            text = ""
            if os.path.exists(save_text):
                with open(save_text,'r') as f:
                    text = f.read()
            if not text:
                prompt = None
                for info in char_info_list:
                    if wave_path.startswith(info['id']):
                        print(f"found {wave_path} in {info['id']}")
                        sample_set = set()
                        sample_set.add(info['key'])
                        sample_set.add(info['name'])
                        for x in info['me']:
                            sample_set.add(x)
                        for x in info['you']:
                            sample_set.add(x)
                        prompt = " ".join(sample_set)
                        print(f"prompt {prompt}")
                        break
                stt.set_initial_prompt(prompt)

                buf = BytesIO()
                repo.download_file( f"{audio_path}/{wave_path}", buf )
                print(f"load {wave_path}")
                audio_data = load_waveb(buf)
                print(f"speech-to-text {wave_path}")
                text = stt.stt( audio_data )
                print(f"[WHISPER] {text}")
                if len(text)>0:
                    print(f"speech-to-text {text}")
                    with open(save_text,'w') as f:
                        f.write(text)
                else:
                    print(f"speech-to-text failed {wave_path}")
                    logger.error(f"speech-to-text failed {wave_path}")
                    if os.path.exists(save_text):
                        print(f"remove {save_text}")
                        os.remove(save_text)
            if text:
                samples = callinfo.get('samples',[])
                if text not in samples:
                    samples.append(text)
                    callinfo['samples'] = samples
    print(f"save to {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(char_info_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
