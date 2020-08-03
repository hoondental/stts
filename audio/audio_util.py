import math
import numpy as np
from scipy import signal
from scipy.signal import get_window
import librosa.util as librosa_util
import librosa
from . import audio


from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from shutil import copyfile

def wav_to_spec_save(wav_path, sample_rate=None, trim_db=None, mono=True, n_fft=1024, win_length=None, hop_length=None, 
                     mel_basis=None, spec_path=None, angle_path=None, mel_path=None, decibel=True, normalize=True):
    if wav_path.endswith('pcm'):
        wav = audio.read_pcm(wav_path)
    else:
        wav, sr = audio.load_wav(wav_path, sample_rate=sample_rate, trim_db=trim_db, mono=mono)
    if mono:
        mag, mel, angle = audio.spectrogram(wav, n_fft, mel_basis, win_length, hop_length, decibel, normalize)
    else:
        mag0, mel0, angle0 = audio.spectrogram(wav[0], n_fft, mel_basis, win_length, hop_length, decibel, normalize)
        mag1, mel1, angle1 = audio.spectrogram(wav[1], n_fft, mel_basis, win_length, hop_length, decibel, normalize)
        mag = np.stack([mag0, mag1], axis=0)
        mel = np.stack([mel0, mel1], axis=0)
        angle = np.stack([angle0, angle1], axis=0)
    if spec_path is not None:
        np.save(spec_path, mag)
    if mel_path is not None:
        np.save(mel_path, mel)
    if angle_path is not None:
        np.save(angle_path, angle)
    return mag.shape[-1]
       
    
def wav_to_spec_save_many(wav_paths, sample_rate=16000, trim_db=None, mono=True, n_fft=1024, win_length=None, hop_length=None, 
                     n_mels=80, spec_paths=None, angle_paths=None, mel_paths=None, decibel=True, normalize=True, num_workers=12):
    mel_basis = audio._mel_basis(sample_rate, n_fft, n_mels) 
    num_files = len(wav_paths)
    if spec_paths is not None:
        assert len(spec_paths) == num_files
    if mel_paths is not None:
        assert len(mel_paths) == num_files
    if angle_paths is not None:
        assert len(angle_paths) == num_files
     
    executor = ProcessPoolExecutor(max_workers=num_workers)
    jobs = []
    for i in range(num_files):
        wav_path = wav_paths[i]
        spec_path = None if spec_paths is None else spec_paths[i]
        mel_path = None if mel_paths is None else mel_paths[i]
        angle_path = None if angle_paths is None else angle_paths[i]
        _partial = partial(wav_to_spec_save, wav_path, sample_rate, trim_db, mono, n_fft, win_length, hop_length, mel_basis, 
                           spec_path, angle_path, mel_path, decibel, normalize)
        job = executor.submit(_partial)
        jobs.append(job)
    n_frames = [job.result() for job in tqdm(jobs)]
    return n_frames


    
