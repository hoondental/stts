import math
import numpy as np
from scipy import signal
from scipy.signal import get_window
import librosa.util as librosa_util
import librosa

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from shutil import copyfile



def stft(y, n_fft=1024, win_length=None, hop_length=None): 
    win_length = n_fft if win_length is None else win_length
    hop_length = int(win_length / 4) if hop_length is None else hop_length
    S = librosa.core.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(S)
    angle = np.angle(S)
    return mag, angle

def istft(mag, angle, win_length=None, hop_length=None): 
    n_fft = 2 * (S.shape[0] - 1)
    win_length = n_fft if win_length is None else win_length
    hop_length = int(win_length / 4) if hop_length is None else hop_length
    S = mag * np.exp(1j * angle)
    return librosa.core.istft(S, hop_length=hop_length, win_length=win_length)

def spectrogram(y, n_fft=1024, mel_basis=None, win_length=None, hop_length=None, preemph=None, decibel=True, normalize=True):
    if preemph is not None:
        y = _preemphasis(y, preemph)
    mag, angle = stft(y, n_fft, win_length, hop_length)
    mel = _linear_to_mel(mag, mel_basis) if mel_basis is not None else None
    if decibel:
        mag = _amp_to_db(mag)
        mel = _amp_to_db(mel) if mel_basis is not None else None
    if normalize:
        mag = _normalize(mag)
        mel = _normalize(mel) if mel_basis is not None else None
    return mag, mel, angle

def griffin_lim(S, n_fft, win_length, hop_length, angle=None, max_iters=30):
    angle = 2 * np.pi * (np.random.rand(*S.shape) - 0.5) if angle is None else angle
    phase = np.exp(1j * angle)
    cS = np.abs(S).astype(np.complex)
    signal = istft(cS * phase, win_length, hop_length)
    for i in range(max_iters):
        X = stft(signal, n_fft, win_length, hop_length)
        angle = np.angle(X)
        phase = np.exp(1j * angle)
        signal = istft(cS * phase, win_length, hop_length)
    return signal


def inv_spectrogram_griffin_lim(spectrogram, n_fft, win_length, hop_length, power=1.0, preemph=None, angle=None, max_iters=30):
    A = _db_to_amp(_denormalize(spectrogram))
    B = A ** power
    wav =  griffin_lim(B, n_fft, win_length=win_length, hop_length=hop_length, angle=angle, max_iters=max_iters)
    if preemph is not None:
        wav = _inv_preemphasis(wav, preemph)
    return wav


def find_endpoint(wav, sample_rate, threshold_db = -40, min_silence_sec=0.8):
    abs_wav = np.abs(wav)
    window_length = int(sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(0, len(wav) - window_length + 1, hop_length):
        if np.max(abs_wav[x:x+window_length]) < threshold:
            return x
    return len(wav)

def load_wav(path, sample_rate=None, trim_db=None, mono=True):
    wav, sr = librosa.core.load(path, sr=sample_rate, mono=mono)
    trimmed, index = librosa.effects.trim(wav, top_db=trim_db) if trim_db else (wav, None)
    return trimmed, sr

def save_wav(wav, path, sample_rate=16000, max_amp=1.0, max_normalize=False):
    max_wav = max(0.01, np.max(np.abs(wav)))
    if max_normalize:
        wav /= max_wav
        max_wav = 1.0
    if max_wav > max_amp:
        wav *= (max_amp / max_wav)
    librosa.output.write_wav(path, wav.astype(np.float32), sample_rate)

def window_sumsquare(window, n_frames, hot_length=200, win_length=None, n_fft=800, dtype=np.float32, norm=None):
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalze(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

def read_pcm(path_pcm):
    with open(path_pcm, 'rb') as f:
        pcm = f.read()
    data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return data
    
def pcm2wav(path_pcm, path_wav, sample_rate=16000):
    with open(path_pcm, 'rb') as f:
        pcm = f.read()
    data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    librosa.output.write_wav(path_wav, data, sample_rate)


def _mel_basis(sample_rate, n_fft, n_mels):
    return librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels)

def _linear_to_mel(spectrogram, mel_basis):
    return np.dot(mel_basis, spectrogram)

def _mel_to_linear(mel, inv_mel_basis=None, mel_basis=None):
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(mel_basis)
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel))

def _amp_to_db(x, min_amp=1e-10):
    return 20 * np.log10(np.maximum(min_amp, x)) # min dB is -200dB

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S, min_db=-80, max_db=20):
    return np.clip((S - min_db) / (max_db - min_db), 1e-10, 1)

def _denormalize(S, min_db=-80, max_db=20):
    return S * (max_db - min_db) + min_db 

def _preemphasis(x, preemph=0.97):
    return signal.lfilter([1, -preemph], [1], x)

def _inv_preemphasis(x, preemph=0.97):
    return signal.lfilter([1], [1, -preemph], x)

def _dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, min=clip_val) * C)
#    return torch.log(torch.clamp(x, min=clip_val) * C)

def _dynamic_range_decompression(x, C=1):
    return np.exp(x) / C
#    return torch.exp(x) / C

def _stft_params(sample_rate, frame_shift_ms, frame_length_ms, num_freq):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length


