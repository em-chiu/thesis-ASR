import pandas as pd
import numpy as np

from whisper_normalizer.basic import BasicTextNormalizer

import jiwer
import glob
import os

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


hypotheses = []
references = []

# glob to get file list-- directory name, glob will give list of all files
hyps = glob.glob('/Users/emily/Desktop/google-stt-txt/*.txt')
refs = glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt')

for hyp, ref in zip(hyps, refs):
    with open(hyp) as hypfile, open(ref) as reffile: #loops through each file #hypfile = file handle
        hyp_read = hypfile.read() #get file contents w/ .read() method
        ref_read = reffile.read()
        hypotheses.append(hyp_read) #append() can add whole list as element to list
        references.append(ref_read)

#load lists into dictionary/dataframe
google_wer_data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

#add file names/labels
file_name = [os.path.basename(x) for x in glob.glob('/Users/emily/Desktop/whisper-output-turbo-word/*.txt')]
google_wer_data["file name"] = file_name

unnormalized_wer = jiwer.wer(list(google_wer_data["reference"]), list(google_wer_data["hypothesis"]))
print(f"Raw Data WER: {unnormalized_wer * 100:.2f}%") #51.67%

#clean data
normalizer = BasicTextNormalizer() # still need to remove /n in hypotheses, removed apostrophes
google_wer_data["hypothesis_clean"] = [normalizer(text) for text in google_wer_data["hypothesis"]]
google_wer_data["reference_clean"] = [normalizer(text) for text in google_wer_data["reference"]]

#lower
google_wer_data['hypothesis_clean'] = [np.char.lower(x) for x in google_wer_data['hypothesis_clean']]
google_wer_data['reference_clean'] = [np.char.lower(x) for x in google_wer_data['reference_clean']]

google_wer_data = google_wer_data.astype("string") #converts df to str type
cleaned_wer = jiwer.wer(list(google_wer_data['reference_clean']), list(google_wer_data['hypothesis_clean'])) # # typeerror: got list
print(f"Normalized & Lowered Data WER: {cleaned_wer * 100:.2f}%") #37.43%

# word tokenize 
google_wer_data['hyp_word_tokens'] = [word_tokenize(x, language='french') for x in google_wer_data['hypothesis_clean']]
google_wer_data['ref_word_tokens'] = [word_tokenize(x, language='french') for x in google_wer_data['reference_clean']]

google_wer_data[['hyp_word_tokens', 'ref_word_tokens']] = google_wer_data[['hyp_word_tokens', 'ref_word_tokens']].astype("string")
token_wer = jiwer.wer(list(google_wer_data['ref_word_tokens']), list(google_wer_data['hyp_word_tokens'])) # # typeerror: got list
print(f"Tokenized Data WER: {token_wer * 100:.2f}%") #37.50%

one_wer = jiwer.wer(list(google_wer_data.loc[3, ['ref_word_tokens']]), list(google_wer_data.loc[3, ['hyp_word_tokens']])) #bac08
print(f"one file WER: {one_wer * 100:.2f}%") #41.82%
print(google_wer_data["file name"])
