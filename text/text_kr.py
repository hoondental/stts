from __future__ import division
import time
import math
import os, copy
import re
import unicodedata
import numpy as np
import librosa
from scipy import signal
from .g2p_kr import G2p
from .g2p_kr.utils import compose
g2p = G2p()


_chars = [c for c in 'abcdefghijklmnopqrstuvwxyz']

_bonset = [b'\xe1\x84\x80', b'\xe1\x84\x81', b'\xe1\x84\x82', b'\xe1\x84\x83', b'\xe1\x84\x84', b'\xe1\x84\x85', b'\xe1\x84\x86', b'\xe1\x84\x87',
           b'\xe1\x84\x88', b'\xe1\x84\x89', b'\xe1\x84\x8a', b'\xe1\x84\x8b', b'\xe1\x84\x8c', b'\xe1\x84\x8d', b'\xe1\x84\x8e', b'\xe1\x84\x8f',
           b'\xe1\x84\x90', b'\xe1\x84\x91', b'\xe1\x84\x92']
_bnucleus = [b'\xe1\x85\xa1', b'\xe1\x85\xa2', b'\xe1\x85\xa3', b'\xe1\x85\xa4', b'\xe1\x85\xa5', b'\xe1\x85\xa6', b'\xe1\x85\xa7',
             b'\xe1\x85\xa8', b'\xe1\x85\xa9', b'\xe1\x85\xaa', b'\xe1\x85\xab', b'\xe1\x85\xac', b'\xe1\x85\xad', b'\xe1\x85\xae',
             b'\xe1\x85\xaf', b'\xe1\x85\xb0', b'\xe1\x85\xb1', b'\xe1\x85\xb2', b'\xe1\x85\xb3', b'\xe1\x85\xb4', b'\xe1\x85\xb5']
_bcoda = [b'\xe1\x86\xa8', b'\xe1\x86\xa9', b'\xe1\x86\xaa', b'\xe1\x86\xab', b'\xe1\x86\xac', b'\xe1\x86\xad', b'\xe1\x86\xae',
       b'\xe1\x86\xaf', b'\xe1\x86\xb0', b'\xe1\x86\xb1', b'\xe1\x86\xb2', b'\xe1\x86\xb3', b'\xe1\x86\xb4', b'\xe1\x86\xb5', b'\xe1\x86\xb6',
       b'\xe1\x86\xb7', b'\xe1\x86\xb8', b'\xe1\x86\xb9', b'\xe1\x86\xba', b'\xe1\x86\xbb', b'\xe1\x86\xbc', b'\xe1\x86\xbd', b'\xe1\x86\xbe',
       b'\xe1\x86\xbf', b'\xe1\x87\x80', b'\xe1\x87\x81', b'\xe1\x87\x82']

_onset = [c.decode() for c in _bonset]
_nucleus = [c.decode() for c in _bnucleus]
_coda = [c.decode() for c in _bcoda]
_bcodaph = [b'\xe1\x86\xa8', b'\xe1\x86\xab', b'\xe1\x86\xae', b'\xe1\x86\xaf', b'\xe1\x86\xb7', b'\xe1\x86\xb8', b'\xe1\x86\xbc']
_codaph = [c.decode() for c in _bcodaph]

_punctuations = [c for c in "',.?!"] # only 5 punctuations are allowed, If some of these are not used, better to fill with somethin else to keep same the indices of the chars.
_spaces = [' ']
_specials = ['<pad>', '<sos>', '<eos>'] # Pad, SOS, EOS, 0, 1, 2
_eksymbols = _specials + _punctuations + _spaces + _chars + _onset + _nucleus + _coda
_ksymbols_gr = _specials + _punctuations + _spaces + _onset + _nucleus + _coda
_ksymbols = _specials + _punctuations + _spaces + _onset + _nucleus + _codaph

_ksymbol2idx = {char:idx for idx, char in enumerate(_ksymbols)}
_idx2ksymbol = [char for char in _ksymbols]
_eksymbol2idx = {char:idx for idx, char in enumerate(_eksymbols)}
_idx2eksymbol = [char for char in _eksymbols]





def text2symbol(text, add_sos=False, add_eos=False, pad=0):
    text = list(g2p(text, to_syl=False))
    if add_sos:
        text = [_specials[1]] + text
    if add_eos:
        text = text + [_specials[2]] 
    if pad > 0:
        text = text + [_specials[0]] * pad
    return text

def symbol2idx(symbols):
    return [_ksymbol2idx[s] for s in symbols]

def text2idx(text, add_sos=False, add_eos=False, pad=0):
    symbols = text2symbol(text, add_sos, add_eos, pad)
    return symbol2idx(symbols)

def idx2symbol(idx):
    return [_idx2ksymbol[i] for i in idx]

def symbol2text(symbols):
    pad, sos, eos = _specials
    try:
        while(True):
            symbols.remove(pad)
    except Exception:
        pass
    try:
        while(True):
            symbols.remove(sos)
    except Exception:
        pass
    try:
        while(True):
            symbols.remove(eos)
    except Exception:
        pass
    return compose(''.join(symbols))

def idx2text(idx):
    symbols = idx2symbol(idx)
    return symbol2text(symbols)
              



    

#cmu_dict phones


_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]



def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text




def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = expand_abbreviations(text)
    text = re.sub("[\"\-()[\]“”~></]", " ", text)
    text = re.sub("[;:]", ".", text)
    text = re.sub("[’]", "'", text)
    text = re.sub("[.]+", ".", text)
    text = re.sub("[']+", "'", text)
    text = re.sub("[ ]+", " ", text)
    text = text.strip()
    return text

