import numpy as np
import librosa

def resample_audio( orig_sr: int, audio: np.ndarray, target_sr: int) -> np.ndarray:
    """
    音声データをリサンプリングする関数

    Args:
        orig_sr (int): 元のサンプリングレート
        audio (np.ndarray): 入力音声データ (np.float32)
        target_sr (int): 目標のサンプリングレート

    Returns:
        np.ndarray: リサンプリングされた音声データ (np.float32)
    """
    if orig_sr == target_sr:
        return audio
    
    resampled = librosa.resample(
        y=audio,
        orig_sr=orig_sr,
        target_sr=target_sr
    )
    return resampled.astype(np.float32)