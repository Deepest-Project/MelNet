# coding: utf-8
import re
import string
import numpy as np

from text import cleaners
from text.symbols import symbols, en_symbols, PAD, EOS
from text.korean import jamo_to_korean



# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}   # 80개
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
isEn=False


# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

puncuation_table = str.maketrans({key: None for key in string.punctuation})

def convert_to_en_symbols():
    '''Converts built-in korean symbols to english, to be used for english training
    
'''
    global _symbol_to_id, _id_to_symbol, isEn
    if not isEn:
        print(" [!] Converting to english mode")
    _symbol_to_id = {s: i for i, s in enumerate(en_symbols)}
    _id_to_symbol = {i: s for i, s in enumerate(en_symbols)}
    isEn=True

def remove_puncuations(text):
    return text.translate(puncuation_table)

def text_to_sequence(text, cleaner_names=['korean_cleaners'], as_token=False):    
    return _text_to_sequence(text, cleaner_names, as_token)

def _text_to_sequence(text, cleaner_names, as_token):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

        The text can optionally have ARPAbet sequences enclosed in curly braces embedded
        in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

        Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through

        Returns:
            List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id[EOS])  # [14, 29, 45, 2, 27, 62, 20, 21, 4, 39, 45, 1]

    if as_token:
        return sequence_to_text(sequence, combine_jamo=True)
    else:
        return np.array(sequence, dtype=np.int32)


def sequence_to_text(sequence, skip_eos_and_pad=False, combine_jamo=False):
    '''Converts a sequence of IDs back to a string'''
        
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]

            if not skip_eos_and_pad or s not in [EOS, PAD]:
                result += s

    result = result.replace('}{', ' ')

    if combine_jamo:
        return jamo_to_korean(result)
    else:
        return result



def _clean_text(text, cleaner_names):
    
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text) # '존경하는' --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']

    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'
