import numpy as np
from numpy.typing import NDArray
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

def time_stretch1(audio: NDArray[np.float32], sample_rate:int, rate: float) -> NDArray[np.float32]:
    """音声の速度を調整する"""
    if 0.91<rate<1.09:
        return audio
    return librosa.effects.time_stretch(audio, rate=rate, n_fft=256, hop_length=32)

def time_stretch2(audio: NDArray[np.float32], sample_rate:int, rate: float) -> NDArray[np.float32]:
    x = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-8)
    b = sample_rate // rate
    return librosa.resample(x, orig_sr=sample_rate, target_sr=b)
