import pandas as pd
import numpy as np

# import jiwer as tr
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

string_wer_data = whisper_wer_data.astype("string") #creates copy of df in str type

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

# string_wer_data["standardized_ref"] = tr.wer_standardize((string_wer_data["reference"]))
# string_wer_data["standardized_hyp"] = tr.wer_standardize((string_wer_data["hypothesis"]))

# assert isinstance(string_wer_data["reference"], list)
# assert all(isinstance(e, str) for e in string_wer_data["reference"])

# for entry in string_wer_data["standardized_ref"]:
#     with open(entry) as entryline:
#         entry_read = entryline.read()
#         wer_standardize(entry_read)



print(jiwer.ReduceToListOfListOfWords()(string_wer_data["reference"]))
