import glob
import json
import re

hyps = glob.glob('/Users/emily/Desktop/google-stt/*.json')
ref = glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt')



transcripts = []

#convert json to txt
# Load JSON data (from a file or a string)
for hyp in hyps:
    for hy in hyp: # want to iterate through each file?
        with open(hyp, encoding="utf-8") as f:
            json_data = json.load(f)
            print(json_data['results'][0])

# https://stackoverflow.com/questions/64733686/how-to-convert-one-key-value-of-json-files-to-txt-files


# extract json results, iterate through
# # https://stackoverflow.com/questions/74777241/google-cloud-speech-to-text-api-convert-json-format-in-plain-text-w-o-variable
#             myrows = json_data['results']
#             for row in myrows: # iterating through index, 0 1 2 etc
#                 trans = row['alternatives'][0]['transcript']
#                 transcripts.append(trans)

#             # Convert to a string representation
#             output_data = json.dumps(transcripts, ensure_ascii=False, indent=4) 
#             #ensure_ascii=False outputs non-ASCII characters as-is #indent for pretty formatting
#             # break # stops the loop, looks at just one instead of all files


#             # Write to a text file
#             # with open('data.txt', 'w') as f:
#             #     f.write(txt_data)

# print(type(output_data)) #str
# print(type(transcripts)) #list
# print(len(transcripts)) #229





