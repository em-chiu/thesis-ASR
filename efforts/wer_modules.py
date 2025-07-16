import pandas as pd
import numpy as np

import jiwer
import glob
import os

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
print(f"raw data WER: {unnormalized_wer * 100:.2f} %")
# #WER: 51.17 %


string_wer_data = whisper_wer_data.astype("string") #creates copy of df in str type

# transformation = jiwer.Compose([
#     jiwer.ToLowerCase(),
#     jiwer.RemoveWhiteSpace(replace_by_space=True),
#     jiwer.RemoveMultipleSpaces(),
#     jiwer.RemovePunctuation(), #but need apostrophes
#     jiwer.ReduceToListOfListOfWords(word_delimiter=" ") #to create list of strings
# ]) 

# string_wer_data["transform_ref"] = transformation(string_wer_data["reference"])
# print(string_wer_data["transform_ref"])


# transform_wer = jiwer.wer(
#     (string_wer_data["reference"]), 
#     (string_wer_data["hypothesis"]),
#     truth_transform=transformation, 
#     hypothesis_transform=transformation
# )
# print(string_wer_data["hypothesis"])
# print(f"transformed WER: {transform_wer * 100:.2f} %")

# transformedwer = jiwer.wer(list(truth_transform), list(hypothesis_transform))
# print(f"transformed_WER: {transformedwer * 100:.2f} %")

# # print(string_wer_data.dtypes)


# wer_standardize = tr.Compose(
#     [
#         tr.ToLowerCase(),
#         tr.ExpandCommonEnglishContractions(),
#         tr.RemoveKaldiNonWords(),
#         tr.RemoveWhiteSpace(replace_by_space=True),
#         tr.RemoveMultipleSpaces(),
#         tr.Strip(),
#         tr.ReduceToListOfListOfWords(),
#     ]
# )

# transform_wer = jiwer.wer(
#     (string_wer_data["reference"]), 
#     (string_wer_data["hypothesis"]),
#     truth_transform=jiwer.wer_standardize, 
#     hypothesis_transform=jiwer.wer_standardize
# )

string_wer_data["standardized_ref"] = jiwer.wer_standardize((string_wer_data["reference"]))
string_wer_data["standardized_hyp"] = jiwer.wer_standardize((string_wer_data["hypothesis"]))
# print(string_wer_data["reference"])

assert isinstance(string_wer_data["reference"], list)
assert all(isinstance(e, str) for e in string_wer_data["reference"])
# assert https://www.reddit.com/r/learnpython/comments/1bjh9bn/what_does_the_assert_keyword_do_in_python/
#if condition returns True, then nothing happens

# stand_wer = jiwer.wer(list(string_wer_data["standardized_ref"]), list(string_wer_data["standardized_hyp"]))
# print(f"transform_wer: {transform_wer * 100:.2f} %")


# jiwer.SubstituteWords({" ": " ", " ": " "}) #dictionary: Mapping[str, str] 
# for b. Expanding written abbreviations, c. Spelling all compound words with space


#jiwer.RemoveSpecificWords(["uh", "oh"]) #words_to_remove: List[str])
# remove filled pauses e.g. uh/um




