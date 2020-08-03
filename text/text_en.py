from __future__ import division
import time
import math
import os, copy
import re
import unicodedata
import numpy as np
import librosa
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from num2words import num2words
from g2p_en import G2p
_g2p = G2p()

# In g2p_en by Kyubyong's github, See the source codes. The set of phonemes is different from this. Especially, ["<pad>", "<unk>", "<s>", "</s>"] instead of [P, S, E] 
# Should be resolved later


_letters = [c for c in 'abcdefghijklmnopqrstuvwxyz']

_punctuations = [c for c in "',.?!"] 
_spaces = [' ']
_specials = ['<pad>', '<sos>', '<eos>'] # Pad, SOS, EOS, 0, 1, 2
_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

_chars = _specials + _punctuations + _spaces + _letters
_char2idx = {c:i for i, c in enumerate(_chars)}
_idx2char = [c for c in _chars]

_vphones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
_stress = ['0', '1', '2']
_vphones0 = [v + _stress[0] for v in _vphones]
_vphones1 = [v + _stress[1] for v in _vphones]
_vphones2 = [v + _stress[2] for v in _vphones]
_sphones = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
_phones = _specials + _punctuations + _spaces + _sphones + _vphones0 + _vphones1 + _vphones2
_phone2idx = {p:i for i, p in enumerate(_phones)}
_idx2phone = [p for p in _phones]

_vstart = len(_specials) + len(_punctuations) + len(_spaces) + len(_sphones)
_len_vphones = len(_vphones)
              

def char2idx(text):
    return [_char2idx[c] for c in text]

def idx2char(idx, to_string=False):
    chars = [_idx2char[i] for i in idx]
    if to_string:
        pad, sos, eos = _specials
        try:
            while(True):
                chars.remove(pad)
        except Exception:
            pass
        try:
            while(True):
                chars.remove(sos)
        except Exception:
            pass
        try:
            while(True):
                chars.remove(eos)
        except Exception:
            pass
        chars = "".join(chars)
    return chars

def phone2idx(phones):
    return [_phone2idx[phone] for phone in phones]

def idx2phone(idx):
    return [_idx2phone[i] for i in idx]

def _idx2idx2(idx):
    i0 = min(_vstart, idx) + max(0, idx - _vstart) % _len_vphones
    i1 = max(-_len_vphones, idx - _vstart) // _len_vphones + 1
    return (i0, i1)

def _idx22idx(idx):
    return idx[0] + max(0, idx[1] - 1) * _len_vphones


def phone2idx2(phones):
    return [_idx2idx2(_phone2idx[phone]) for phone in phones]

def idx22phone(idx):
    return [_idx2phone[_idx22idx(i)] for i in idx]


def text2phone(text):
    return _g2p(text)


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

def text_num2words(text):
    def tonumber(s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                raise ValueError("error in detecting numebrs")

    def is_number(s):
        if not re.search('\d', s):
            return False
        if s[0] >= 'a' and s[0] <= 'z' or s[0] >= 'A' and s[0] <= 'Z':
            return False
        if s[-1] >= 'a' and s[-1] <= 'z' or s[-1] >= 'A' and s[-1] <= 'Z':
            return False

        for i in range(1, len(s) - 1):
            c = s[i]
            if not (c >= '0' and c <= '9' or c == '.'):
                return False
        return True

    def strip_number(s):
        if not is_number(s):
            if re.search('\d', s):
                return ''.join([' ' + num2words(int(c)) + ' ' if c >= '0' and c <= '9' else c for c in s])
            else:
                return s
        i = 0
        if s[i] == '.':
            s = '0' + s
        while s[i] < '0' or s[i] > '9':
            i += 1
        j = len(s) - 1
        while s[j] < '0' or s[j] > '9':
            j -= 1
        start = s[:i]
        end = '' if j == len(s) - 1 else s[j + 1:]
        word = tonumber(s[i: j+1])
        return start + ' ' + num2words(word).replace(',', ' ') + ' ' + end

    text = " ".join([strip_number(s) for s in text.split()])
    return text


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = text_num2words(text)
    text = expand_abbreviations(text)
    text = re.sub("[\"\-()[\]“”><~]", " ", text)
    text = re.sub("[;:]", ".", text)
    text = re.sub("[’]", "'", text)
    text = re.sub("[^{}]".format(_char_vocab), " ", text)
    text = re.sub("[.]+", ".", text)
    text = re.sub("[']+", "'", text)
    text = re.sub("[ ]+", " ", text)
    text = text.strip()
    if text[-1] >= 'a' and text[-1] <= 'z':
        text += '.'
    return text

