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
hyps = glob.glob('/Users/emily/Desktop/google-stt-txt/*.txt')
refs = glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt')

for hyp, ref in zip(hyps, refs):
    with open(hyp) as hypfile, open(ref) as reffile: #loops through each file #hypfile = file handle
        hyp_read = hypfile.read() #get file contents w/ .read() method
        ref_read = reffile.read()
        hypotheses.append(hyp_read) #append() can add whole list as element to list
        references.append(ref_read)

# # datafile.read.split (to get file contents, split by white space)
# #append into list after splitting (OR split each file contents when i'm ready)

#load lists into dictionary/dataframe
google_wer_data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

#add file names/labels
file_name = [os.path.basename(x) for x in glob.glob('/Users/emily/Desktop/whisper-output-turbo-word/*.txt')]
google_wer_data["file name"] = file_name

unnormalized_wer = jiwer.wer(list(google_wer_data["reference"]), list(google_wer_data["hypothesis"]))
print(f"raw data WER: {unnormalized_wer * 100:.2f} %") # 51.17 %

#clean data
normalizer = BasicTextNormalizer() # still need to remove /n in hypotheses, removed apostrophes
google_wer_data["hypothesis_clean"] = [normalizer(text) for text in google_wer_data["hypothesis"]]
google_wer_data["reference_clean"] = [normalizer(text) for text in google_wer_data["reference"]]

#lower
google_wer_data['hypothesis_clean'] = [np.char.lower(x) for x in google_wer_data['hypothesis_clean']]
google_wer_data['reference_clean'] = [np.char.lower(x) for x in google_wer_data['reference_clean']]

google_wer_data = google_wer_data.astype("string") #converts df to str type
cleaned_wer = jiwer.wer(list(google_wer_data['reference_clean']), list(google_wer_data['hypothesis_clean'])) # # typeerror: got list
print(f"Cleaned data WER: {cleaned_wer * 100:.2f} %") #31.04 %
# print(google_wer_data)

# word tokenize #splits l'avez correctly
google_wer_data['hyp_word_tokens'] = [word_tokenize(x, language='french') for x in google_wer_data['hypothesis_clean']]
google_wer_data['ref_word_tokens'] = [word_tokenize(x, language='french') for x in google_wer_data['reference_clean']]

google_wer_data[['hyp_word_tokens', 'ref_word_tokens']] = google_wer_data[['hyp_word_tokens', 'ref_word_tokens']].astype("string")
token_wer = jiwer.wer(list(google_wer_data['ref_word_tokens']), list(google_wer_data['hyp_word_tokens'])) # # typeerror: got list
print(f"Tokenized data WER: {token_wer * 100:.2f} %") #31.11 %
# print(google_wer_data['hyp_word_tokens'])

one_wer = jiwer.wer(list(google_wer_data.loc[3, ['ref_word_tokens']]), list(google_wer_data.loc[3, ['hyp_word_tokens']])) #bac08
print(f"one file WER: {one_wer * 100:.2f} %") #36.79 %
print(google_wer_data["file name"])

# print(whisper_wer_data.dtypes)
# split/list of words, string = single word

# whisper_wer_data['hyp_split'] =  [re.sub("[^\w]", " ", str(x).split()) for x in whisper_wer_data['hypothesis_clean']]
# whisper_wer_data['ref_split'] =  [re.sub("[^\w]", " ", str(x).split()) for x in whisper_wer_data['reference_clean']]
#TypeError: expected string or bytes-like object, got 'list'


# split_wer = jiwer.wer(list(whisper_wer_data['ref_split']), list(whisper_wer_data['hyp_split']))
# print(f"split data WER: {split_wer * 100:.2f} %") #


# jiwer.SubstituteWords({" ": " ", " ": " "}) #dictionary: Mapping[str, str] 
# for b. Expanding written abbreviations, c. Spelling all compound words with space

#jiwer.RemoveSpecificWords(["uh", "oh"]) #words_to_remove: List[str])
# remove filled pauses e.g. uh/um

