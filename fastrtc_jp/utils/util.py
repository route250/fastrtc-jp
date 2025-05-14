import sys,os
import httpx
from dotenv import load_dotenv as _load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

_EMPTY_DATA = np.zeros((1,), dtype=np.float32)

def to_lang_code(language:str|None=None) ->str:
    if not language or len(language)<2:
        return "ja"
    else:
        return language[:2].lower()

async def get_availavle_url( hostlist:str, timeout: float = 0.5 ) -> str | None:
    """
    音声合成サーバーのURLを取得する
    :param hostlist: ホスト名のリスト
    :return: 音声合成サーバーのURL
    """
    url_list:list[str] = list(set(hostlist.split(',')))
    if len(url_list)>0:
        async with httpx.AsyncClient(timeout=httpx.Timeout( timeout, connect=timeout, read=timeout)) as client:
            for url in url_list:
                    try:
                        response = await client.get(url)
                        if response.status_code == 200 or response.status_code == 404:
                            return url
                    except (httpx.ConnectError, httpx.TimeoutException):
                        continue
    return None


def load_dotenv():
    os.environ["AGNO_MONITOR"] = "false"
    os.environ["AGNO_TELEMETRY"] = "false"
    ENVFILE = os.path.expanduser("~/.api_key.txt")
    if os.path.exists(ENVFILE):
        _load_dotenv(ENVFILE)
    else:
        print(f"ENVFILEYが見つかりません")
    if os.path.exists('config.env'):
        _load_dotenv('config.env')
    elif os.path.exists('.env'):
        _load_dotenv('.env')
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEYが設定されていません")
        sys.exit(1)


def setup_logger(log_dir:str = './tmp/logs/'):
    # Set up logging to output to ./tmp/logs/
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'application.log')
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])

    # Set less important loggers to ERROR level
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    logging.getLogger('aioice').setLevel(logging.ERROR)
    logging.getLogger('aiortc').setLevel(logging.ERROR)
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('openai').setLevel(logging.ERROR)
    logging.getLogger('fastrtc').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('gtts').setLevel(logging.ERROR)