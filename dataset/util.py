import os
import sys
#sys.path.append(os.path.dirname(__file__))
#sys.path.append('..')
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import math
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
import librosa
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from shutil import copyfile
import argparse
import re
import pickle
from scipy.io import wavfile
import scipy.io

from torch.utils.data.sampler import Sampler

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

    

        
    
# meta in memory use absolute path, metafile use relative path w.r.t the base dir of the metafile

def read_meta(meta_path, spec_mel=False):  
    dir_base = os.path.dirname(meta_path)
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    meta = []
    if spec_mel:
        for line in lines:
            path, specpath, melpath, nframe, script = line.strip().split('|')
            _nframe = int(nframe)
            _fname = path.split('/')[-1]
            _pidx = _fname.rfind('.')
            if _pidx > 0:
                _fname = _fname[:_pidx]
            _path = os.path.join(dir_base, path)  
            _specpath = os.path.join(dir_base, specpath)
            _melpath = os.path.join(dir_base, melpath)
            meta.append([_fname, _path, _specpath, _melpath, _nframe, script])
    else:
        for line in lines:
            path, script = line.strip().split('|')
            _fname = path.split('/')[-1]
            _pidx = _fname.rfind('.')
            if _pidx > 0:
                _fname = _fname[:_pidx]
            _path = os.path.join(dir_base, path)        
            meta.append([_fname, _path, script])
    return meta

def save_meta(meta, meta_path, spec_mel=False):
    dir_base = os.path.dirname(meta_path)
    with open(meta_path, 'w') as f:
        lines = ''
        if spec_mel:
            for _meta in meta:
                fname, path, specpath, melpath, nframe, script = _meta
                _path = os.path.relpath(path, dir_base)
                _specpath = os.path.relpath(specpath, dir_base)
                _melpath = os.path.relpath(melpath, dir_base)
                _nframe = str(nframe)
                lines += '|'.join([_path, _specpath, _melpath, _nframe, script]) + '\n'
        else:
            for _meta in meta:
                fname, path, script = _meta
                _path = os.path.relpath(path, dir_base)
                lines += '|'.join([_path, script]) + '\n'
        f.write(lines[:-1])
        
        
def seperate_train_val(meta_path, val_ratio=0.2, shuffle=False, train_metafile='train_meta.txt', val_metafile='val_meta.txt'):
    dir_base = os.path.dirname(meta_path)
    meta = read_meta(meta_path)
    idx = [i for i in range(len(meta))]
    train = []
    val = []
    if shuffle:
        idx = np.random.permutation(idx)
    n_val = int(len(meta) * val_ratio)
    n_train = len(meta) - n_val
    val_idx = np.random.choice(idx, size=n_val, replace=False)
    for i in idx:
        if i in val_idx:
            val.append(meta[i])
        else:
            train.append(meta[i])
    save_meta(train, os.path.join(dir_base, train_metafile))
    save_meta(val, os.path.join(dir_base, val_metafile))
    
   
                  

def preprocess(meta_path, dir_data, out_subdir='train', sample_rate=22050, n_fft=1024, win_length=None, 
               hop_length=None, n_mels=80, mono=True, trim_db=None, decibel=True, normalize=True):
    meta = read_meta(meta_path)
    out_dir = os.path.join(dir_data, out_subdir)
    process_wavfiles(meta, out_dir, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, decibel, normalize)
                     
    
    
def process_wavfiles(meta, out_dir, sample_rate=22050, trim_db=None, mono=True, n_fft=1024, win_length=None, hop_length=None, 
                     n_mels=80, decibel=True, normalize=True):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length / 4)
    _spec_dir = 'spec'
    _mel_dir = 'mel'
    spec_dir = os.path.join(out_dir, _spec_dir)
    mel_dir = os.path.join(out_dir, _mel_dir)
    meta_path = os.path.join(out_dir, 'meta.txt')
   
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(spec_dir):
        os.mkdir(spec_dir)
    if not os.path.exists(mel_dir):
        os.mkdir(mel_dir)
    wav_paths = []
    spec_paths = []
    mel_paths = []
    _meta = []
    for fname, path, script in meta:
        wav_paths.append(path)
        spec_file = fname + '.spec.npy'
        mel_file = fname + '.mel.npy'
        spec_path = os.path.join(spec_dir, spec_file)
        mel_path = os.path.join(mel_dir, mel_file)
        spec_paths.append(spec_path)
        mel_paths.append(mel_path)
        _meta.append([fname, path, spec_path, mel_path, None, script])
                     
                     
    n_frames = audio_util.wav_to_spec_save_many(wav_paths, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, 
                                           spec_paths, None, mel_paths, decibel, normalize)
    
    for i, _m in enumerate(_meta):
        _m[4] = n_frames[i]
                  
    save_meta(_meta, meta_path, spec_mel=True)
            
    with open(os.path.join(out_dir, 'settings.txt'), 'w') as f:
        f.write('sample_rate:' + str(sample_rate) + '\n')
        f.write('n_fft:' + str(n_fft) + '\n')
        f.write('win_length:' + str(win_length) + '\n')
        f.write('hop_length:' + str(hop_length) + '\n')
        f.write('n_mels:' + str(n_mels) + '\n')
        f.write('trim_db:' + str(trim_db) + '\n')
        f.write('mono:' + str(mono) + '\n')
        f.write('decibel:' + str(decibel) + '\n')
        f.write('normalize:' + str(normalize) + '\n')
        
        
class LengthSampler(Sampler):
    def __init__(self, lengths, batch_size, noise=10.0, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.noise = noise
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.lengths)
    
    def __iter__(self):
        _lengths = torch.tensor(self.lengths, dtype=torch.float32)
        _lengths = _lengths + 2 * self.noise * torch.rand_like(_lengths)
        _sorted, _idx = torch.sort(_lengths)
        
        if self.shuffle:
            _len = len(_lengths)
            _num_full_batches = _len // self.batch_size
            _full_batch_size = _num_full_batches * self.batch_size
            _full_batch_idx = _idx[:_full_batch_size]
            _remnant_idx = _idx[_full_batch_size:]
            _rand_batch_idx = torch.randperm(_num_full_batches)
            _rand_full_idx = _full_batch_idx.reshape(_num_full_batches, self.batch_size)[_rand_batch_idx].reshape(-1) 
            _rand_idx = torch.cat([_rand_full_idx, _remnant_idx], dim=0)
            return iter(list(_rand_idx.numpy()))
        else:
            return iter(list(_idx.numpy()))