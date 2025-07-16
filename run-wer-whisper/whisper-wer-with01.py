import pandas as pd
import numpy as np

from whisper_normalizer.basic import BasicTextNormalizer

import jiwer
import glob
import os
import re

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


hypotheses = []
references = []

# glob to get file list-- directory name, glob will give list of all files
hyps = glob.glob('/Users/emily/Desktop/whisper-output-turbo-word/*.txt')
refs = glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt')

for hyp, ref in zip(hyps, refs):
    with open(hyp) as hypfile, open(ref) as reffile: #loops through each file #hypfile = file handle
        hyp_read = hypfile.read() #get file contents w/ .read() method
        ref_read = reffile.read()
        hypotheses.append(hyp_read) #append() can add whole list as element to list
        references.append(ref_read)


#load lists into dictionary/dataframe
whisper_wer_data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

#add file names/labels
file_name = [os.path.basename(x) for x in glob.glob('/Users/emily/Desktop/whisper-output-turbo-word/*.txt')]
whisper_wer_data["file name"] = file_name

unnormalized_wer = jiwer.wer(list(whisper_wer_data["reference"]), list(whisper_wer_data["hypothesis"]))
print(f"Raw Data WER: {unnormalized_wer * 100:.2f}%") #51.08%

#clean data
normalizer = BasicTextNormalizer() # still need to remove /n in hypotheses, removed apostrophes
whisper_wer_data["hypothesis_clean"] = [normalizer(text) for text in whisper_wer_data["hypothesis"]]
whisper_wer_data["reference_clean"] = [normalizer(text) for text in whisper_wer_data["reference"]]

#lower
whisper_wer_data['hypothesis_clean'] = [np.char.lower(x) for x in whisper_wer_data['hypothesis_clean']]
whisper_wer_data['reference_clean'] = [np.char.lower(x) for x in whisper_wer_data['reference_clean']]

whisper_wer_data = whisper_wer_data.astype("string") #converts df to str type
cleaned_wer = jiwer.wer(list(whisper_wer_data['reference_clean']), list(whisper_wer_data['hypothesis_clean'])) # # typeerror: got list
print(f"Normalized & Lowered Data WER: {cleaned_wer * 100:.2f}%") #30.98%
# print(whisper_wer_data)

# word tokenize #splits l'avez correctly
whisper_wer_data['hyp_word_tokens'] = [word_tokenize(x, language='french') for x in whisper_wer_data['hypothesis_clean']]
whisper_wer_data['ref_word_tokens'] = [word_tokenize(x, language='french') for x in whisper_wer_data['reference_clean']]

whisper_wer_data[['hyp_word_tokens', 'ref_word_tokens']] = whisper_wer_data[['hyp_word_tokens', 'ref_word_tokens']].astype("string")
token_wer = jiwer.wer(list(whisper_wer_data['ref_word_tokens']), list(whisper_wer_data['hyp_word_tokens'])) # # typeerror: got list
print(f"Tokenized Data WER: {token_wer * 100:.2f}%") #31.05%

