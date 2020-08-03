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
        wav = read_pcm(wav_path)
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

def wav_to_spec_save_batch(wav_paths, sample_rate=None, trim_db=None, mono=True, n_fft=1024, win_length=None, hop_length=None, 
                     mel_basis=None, spec_paths=None, angle_paths=None, mel_paths=None, decibel=True, normalize=True):
    n_frames = []
    for i, path in enumerate(wav_paths):
        spec_path = spec_paths[i] if spec_paths is not None else None
        mel_path = mel_paths[i] if mel_paths is not None else None
        angle_path = angle_paths[i] if angle_paths is not None else None
        n_frames.append(wav_to_spec_save(path, sample_rate, trim_db, mono, n_fft, win_length, hop_length, mel_basis, spec_path, angle_path, 
                                         mel_path, decibel, normalize))
    return n_frames
       
    
def wav_to_spec_save_many(wav_paths, sample_rate=16000, trim_db=None, mono=True, n_fft=1024, win_length=None, hop_length=None, 
                     n_mels=80, spec_paths=None, angle_paths=None, mel_paths=None, decibel=True, normalize=True, 
                          batch_size=512, num_workers=12):
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
    _wav_paths = []
    _spec_paths = [] if spec_paths is not None else None
    _mel_paths = [] if mel_paths is not None else None
    _angle_paths = [] if angle_paths is not None else None
    _count = 0
    for i in range(num_files):
        _wav_paths.append(wav_paths[i])
        if spec_paths is not None:
            _spec_paths.append(spec_paths[i])
        if mel_paths is not None:
            _mel_paths.append(mel_paths[i])
        if angle_paths is not None:
            _angle_paths.append(angle_paths[i])
        _count += 1
        if _count == batch_size or i == num_files - 1:
            _partial = partial(wav_to_spec_save_batch, _wav_paths, sample_rate, trim_db, mono, n_fft, win_length, hop_length, mel_basis, 
                           _spec_paths, angle_paths, mel_paths, decibel, normalize)
            job = executor.submit(_partial)
            jobs.append(job)
            _wav_paths = []
            _spec_paths = [] if spec_paths is not None else None
            _mel_paths = [] if mel_paths is not None else None
            _angle_paths = [] if angle_paths is not None else None
            _count = 0
    results = [job.result() for job in tqdm(jobs)]
    n_frames = []
    for result in results:
        n_frames = n_frames + result    
    return n_frames

    
