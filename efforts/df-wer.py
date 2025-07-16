import pandas as pd
import numpy as np

from whisper_normalizer.basic import BasicTextNormalizer

import jiwer
import glob
import os

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

from text_to_num import alpha2digit

hypotheses = []
references = []

# glob to get file list-- directory name, glob will give list of all files
hyps = glob.glob('/Users/emily/Desktop/whisper-output-turbo-word/*.txt')
refs = glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt')
# print(refs)
# print(hyps)

for hyp, ref in zip(hyps, refs):
    with open(hyp) as hypfile, open(ref) as reffile: #loops through each file #hypfile = file handle
        hyp_read = hypfile.read() #get file contents w/ .read() method
        ref_read = reffile.read()
        hypotheses.append(hyp_read) #append() can add whole list as element to list
        references.append(ref_read)
        # print(references)
        # print(len(references)) #list, 35?

# # datafile.read.split (to get file contents, split by white space)
# #append into list after splitting (OR split each file contents when i'm ready)

# # https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=EBGSITeBYPTT
#load lists into dictionary/dataframe
whisper_wer_data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

#add file names/labels
file_name = [os.path.basename(x) for x in glob.glob('/Users/emily/Desktop/whisper-output-turbo-word/*.txt')]
whisper_wer_data["file name"] = file_name

normalizer = BasicTextNormalizer() # still need to remove /n in hypotheses, removed apostrophes
whisper_wer_data["hypothesis_clean"] = [normalizer(text) for text in whisper_wer_data["hypothesis"]]
whisper_wer_data["reference_clean"] = [normalizer(text) for text in whisper_wer_data["reference"]]
# print(whisper_wer_data) 

#lower
whisper_wer_data['hypothesis_low'] = [np.char.lower(x) for x in whisper_wer_data['hypothesis_clean']]
whisper_wer_data['reference_low'] = [np.char.lower(x) for x in whisper_wer_data['reference_clean']]
# print (whisper_wer_data['hypothesis_clean'])
# print(whisper_wer_data.dtypes) ##data type of each column, all object #NEED StringDtype??
# Columns with mixed types are stored with the object dtype 

string_data = whisper_wer_data.astype("string") #creates str type of df
# print(string_data.dtypes.value_counts()) #string[python]    5


# word tokenize
string_data['hyp_word_tokens'] = [word_tokenize(x, language='french') for x in string_data['hypothesis_low']]
string_data['ref_word_tokens'] = [word_tokenize(x, language='french') for x in string_data['reference_low']]
# = string_data['hypothesis_low'].apply(nltk.word_tokenize(language='french')) #asked for text in required position even when following proper syntax when using .apply()
# print (string_data['ref_word_tokens']) 
# TypeError: cannot use a string pattern on a bytes-like object #bytes and not str when trying to tokenize?


string_data[['hyp_word_tokens', 'ref_word_tokens']] = string_data[['hyp_word_tokens', 'ref_word_tokens']].astype("string")
# print(string_data.dtypes)
token_wer = jiwer.wer(list(string_data['ref_word_tokens']), list(string_data['hyp_word_tokens'])) # # typeerror: got list
print(f"Tokenized data WER: {token_wer * 100:.2f} %")

# jiwer.SubstituteWords({" ": " ", " ": " "}) #dictionary: Mapping[str, str] 
# for b. Expanding written abbreviations, c. Spelling all compound words with space
# Converting numerals to words
# whisper_wer_data['hypothesis_clean'] = [alpha2digit(x,"fr") for x in whisper_wer_data['hypothesis_clean']]
#TypeError: cannot use a string pattern on a bytes-like object
# print(whisper_wer_data['hypothesis_clean'])


#jiwer.RemoveSpecificWords(["uh", "oh"]) #words_to_remove: List[str])
# remove filled pauses e.g. uh/um

#to do: WER needs list of strings, string = single word
# b) wordList = re.sub("[^\w]", " ",  insert_string_variable).split()
# https://stackoverflow.com/questions/6181763/converting-a-string-to-a-list-of-words
# c) toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
# https://stackoverflow.com/questions/42428390/nltk-french-tokenizer-in-python-not-working



transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "), #to create list of strings
    jiwer.RemovePunctuation() #but need apostrophes
]) 

# wer = jiwer.wer(list(whisper_wer_data["reference_clean"]), list(whisper_wer_data["hypothesis_clean"]),
#     truth_transform=transformation, 
#     hypothesis_transform=transformation
# )




# print(whisper_wer_data.dtypes.value_counts())
# #object 5




# print(string_data)
# wer = jiwer.wer(list(string_data["reference_clean"]), list(string_data["hypothesis_clean"]))
# print(f"WER: {wer * 100:.2f} %")
# # WER: 31.04 %

# unnormalized_wer = jiwer.wer(list(string_data["reference"]), list(string_data["hypothesis"]))
# print(f"WER: {unnormalized_wer * 100:.2f} %")
# #WER: 51.17 %
# print(string_data.keys()) # reference, hypothesis