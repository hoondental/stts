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

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random


from ..text import text_kr as tkr, text_en as ten
from .util import read_meta, save_meta, LengthSampler


 

class VoiceDataset(Dataset):
    def __init__(self, meta_path, symbols, use_spec=True, use_mel=True, stride=1, add_sos=False, add_eos=False, tensor_type='torch'):
        self.use_spec = use_spec
        self.use_mel = use_mel
        self.stride = stride
        
        self.symbols = symbols
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.pad = symbols[0]
        self.sos = symbols[1]
        self.eos = symbols[2]
        self._symbol2idx = {s:i for i, s in enumerate(symbols)}
                
        self.tensor_type = tensor_type
        self.meta_dir = os.path.dirname(meta_path)
        self.meta_path = meta_path
        self.meta = read_meta(meta_path, spec_mel=True)        
        
        self._fname = []
        self._script = []
        self._nscript = []
        self._stext = []
        self._itext = []
        self._n_frame = []
        self._wav_path = []
        self._spec_path = []
        self._mel_path = []        
        for fname, wav_path, spec_path, mel_path, n_frame, script in self.meta:
            self._fname.append(fname)
            self._script.append(script)
            self._nscript.append(None)
            self._stext.append(None)
            self._itext.append(None)
            self._wav_path.append(wav_path)
            self._spec_path.append(spec_path)
            self._mel_path.append(mel_path)
            self._n_frame.append(int(n_frame))
            
        self.process_scripts()
            
    
    def process_scripts(self, num_workers=12, batch_size=512):
        executor = ProcessPoolExecutor(max_workers=num_workers)
        jobs = []
        _len = len(self.meta)
        _count = 0
        i_scripts = []
        for i in range(_len):
            i_scripts.append((i, self._script[i]))
            _count += 1
            if _count == batch_size or i == _len - 1:
                _partial = partial(self._process_script_batch, i_scripts)
                job = executor.submit(_partial)
                jobs.append(job)
                _count = 0
                i_scripts = []

        results = [job.result() for job in tqdm(jobs)]  
        corrupted = []
        for result in results:
            for i, nscript, stext, itext in result:
                if itext is not None:
                    self._nscript[i] = nscript
                    self._stext[i] = stext
                    self._itext[i] = itext
                else:
                    corrupted.append(i)
                    
        for j in range(len(corrupted)):
            i = len(corrupted) - 1 - j
            self._fname.pop(i)
            self._script.pop(i)
            self._nscript.pop(i)
            self._stext.pop(i)
            self._itext.pop(i)
            self._wav_path.pop(i)
            self._spec_path.pop(i)
            self._mel_path.pop(i)
            self._n_frame.pop(i)
    
            
            
    def _process_script(self, i, script):
        nscript = list(script)
        stext = list(nscript)
        if self.add_sos:
            stext = [self.sos] + stext
        if self.add_eos:
            stext = stext + [self.eos]
        itext = [self._symbol2idx[s] for s in stext]
        return (i, nscript, stext, itext)
    
    def _process_script_batch(self, i_scripts):
        results = []
        for i, script in i_scripts:
            results.append(self._process_script(i, script))
        return results
   
            
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        _spec_path = self._spec_path[idx]
        _mel_path = self._mel_path[idx]
        _text = self._itext[idx]

        _text = np.array(_text, dtype=np.int64)
        _n_frame = 0
        sample = {'idx':idx, 'text':_text, 'n_text':len(_text)}
        if self.use_spec:
            _spec = np.load(_spec_path)[...,::self.stride]
            _n_frame = _spec.shape[-1]
            sample['spec'] = _spec
        if self.use_mel:
            _mel = np.load(_mel_path)[...,::self.stride]        
            _n_frame = _mel.shape[-1]
            sample['mel'] = _mel
        sample['n_frame'] = _n_frame
        return sample
    
            
    def collate(self, samples):
        text_lengths = []
        n_frames = []
        idxes = []
        texts = []
        specs = []
        mels = []
        for i, s in enumerate(samples):
            text_lengths.append(s['n_text'])
            n_frames.append(s['n_frame'])
            idxes.append(s['idx'])
        max_text_len = max(text_lengths)
        max_n_frame = max(n_frames)
        
        for i, s in enumerate(samples):
            texts.append(np.pad(s['text'], (0, max_text_len - text_lengths[i]), constant_values=0))
            if self.use_spec:
                specs.append(np.pad(s['spec'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))
            if self.use_mel:
                mels.append(np.pad(s['mel'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))  
                    
        if self.tensor_type == 'torch':
            tensor = torch.tensor
            int32 = torch.int32
            int64 = torch.int64
            float32 = torch.float32
        elif self.tensor_type == 'numpy':
            tensor = np.array
            int32 = np.int32
            int64 = np.int64
            float32 = np.float32
        else:
            raise Exception('only torch or numpy is supported')
            
        batch = {'idx':tensor(idxes, dtype=int64), 
                 'text':tensor(texts, dtype=int64), 
                 'n_text':tensor(text_lengths, dtype=int32),
                 'n_frame':tensor(n_frames, dtype=int32)}
        if self.use_spec:
            batch['spec'] = tensor(specs, dtype=float32)
        if self.use_mel:
            batch['mel'] = tensor(mels, dtype=float32)
        return batch
    
    def get_length_sampler(self, batch_size, noise=10.0, shuffle=True):
        sampler = LengthSampler(self._n_frame, batch_size, noise, shuffle)
        return sampler
        
        



class KrDataset(VoiceDataset):
    def __init__(self, meta_path, symbols=tkr._ksymbols, use_spec=True, use_mel=False, stride=1, add_sos=False, add_eos=False, tensor_type='torch'):
        super().__init__(meta_path, symbols, use_spec, use_mel, stride, add_sos, add_eos, tensor_type)

            
    def _process_script(self, i, script):
#        nscript = tkr.text_normalize(script)
#        stext = tkr.text2symbol(nscript, self.add_sos, self.add_eos)
#        itext = tkr.symbol2idx(stext)
        try:
            nscript = tkr.text_normalize(script)
            stext = tkr.text2symbol(nscript, self.add_sos, self.add_eos)
            itext = tkr.symbol2idx(stext)
        except Exception as ex:
            print(ex)
            print(i)
            print(script)
            print(nscript)
            print(stext)
            return (i, None, None, None)
        else:
            return (i, nscript, stext, itext)
    
    
class EnDataset(VoiceDataset):
    def __init__(self, meta_path, symbols=ten._chars, use_spec=True, use_mel=False, stride=1, add_sos=False, add_eos=False, tensor_type='torch'):
        super().__init__(meta_path, symbols, use_spec, use_mel, stride, add_sos, add_eos, tensor_type)

            
    def _process_script(self, i, script):
        nscript = ten.text_normalize(script)
        stext = list(nscript)
        if self.add_sos:
            stext = [self.sos] + stext
        if self.add_eos:
            stext = stext + [self.eos]
        itext = ten.char2idx(stext)
        return (i, nscript, stext, itext)
    
class EnPhDataset(VoiceDataset):
    def __init__(self, meta_path, symbols=ten._phones, use_phone2=False, 
                 use_spec=True, use_mel=False, stride=1, add_sos=False, add_eos=False, tensor_type='torch'):
        super().__init__(meta_path, symbols, use_spec, use_mel, stride, add_sos, add_eos, tensor_type)
        self.use_phone2 = use_phone2

            
    def _process_script(self, i, script):
        nscript = ten.text_normalize(script)
        stext = list(ten.text2phone(nscript))
        if self.add_sos:
            stext = [self.sos] + stext
        if self.add_eos:
            stext = stext + [self.eos]
            
        if self.use_phone2:
            itext = ten.phone2idx2(stext)
        else:
            itext = ten.phone2idx(stext)
        return (i, nscript, stext, itext)






    
    
