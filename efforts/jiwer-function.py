import jiwer
from jiwer import wer

from whisper_normalizer.basic import BasicTextNormalizer

import glob
import os.path

#    function for inputting file name to be evaluated

def calculate_wer():
    # reference_base_path = '/Users/emily/Desktop/cd-reference-txt/*.txt'
    # reference_file_name = &quot;file_name.txt&quot;
    # #join path 
    # ref_file_path = os.path.join(reference_base_path, reference_file_name)

    # hypothesis_base_path = '/Users/emily/Desktop/whisper-output-base/'
    # hypothesis_file_name = "file.txt" #&quot;file_name.txt&quot; placeholder?
    # #join path 
    # hyp_file_path = os.path.join(hypothesis_base_path, hypothesis_file_name)

    hypotheses = []
    references = []
    hyps = glob.glob('/Users/emily/Desktop/whisper-output-base/*.txt')
    refs = glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt')
    for hyp, ref in zip(hyps, refs):
        with open(hyp) as hypfile, open(ref) as reffile: #loops through each file #hypfile = file handle
            hyp_read = hypfile.read() #get file contents w/ .read() method
            ref_read = reffile.read()
            # print(hyp_read)
            hypotheses.append(hyp_read) #append() can add whole list as element to list
            references.append(ref_read)

        # with open(reference_file_name, 'w') as reference_file, open(hypothesis_file_name, 'w') as hypothesis_file:
        #     reference_file.write(normalizer = BasicTextNormalizer())
        #     hypothesis_file.write(normalizer = BasicTextNormalizer())

        wer = jiwer.wer(references, hypotheses)
        return (f"WER: {wer * 100:.2f}%")
    
if __name__ == "__main__": # allows function to run when running file in terminal
    calculate_wer()
    # https://www.reddit.com/r/learnpython/comments/1bjh9bn/comment/kvrp7j4/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
    #assert statement as unit tests