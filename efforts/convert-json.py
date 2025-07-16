import glob
import json
import re

hyps = glob.glob('/Users/emily/Desktop/google-stt/*.json')
ref = glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt')
# print(hyp)


transcripts = []

#convert json to txt
# Load JSON data (from a file or a string)
for hyp in hyps:
    with open(hyp, encoding="utf-8") as f:
        json_data = json.load(f)
        # print(json_data['results'])


# https://stackoverflow.com/questions/64733686/how-to-convert-one-key-value-of-json-files-to-txt-files

        # https://pynative.com/python-parse-multiple-json-objects-from-file/
# jsonlist = []
    #     for jsonObj in f:
    #         transcriptDict = json.loads(jsonObj)
    #         jsonlist.append(transcriptDict)
    # print("Printing each JSON Decoded Object")
    # for transcript in jsonlist:
    #     print(transcript['results']) #not pulling from all relevant objects


        # tran = json_data['results'][0]['alternatives'][0]['transcript']
        # print(tran) #only pulls from first transcript sub-key, most accurate effort thus far


# extract json results, iterate through
# https://stackoverflow.com/questions/74777241/google-cloud-speech-to-text-api-convert-json-format-in-plain-text-w-o-variable
        myrows = json_data['results']
        for row in myrows: # iterating through index, 0 1 2 etc
            trans = row['alternatives'][0]['transcript']
            transcripts.append(trans)

        # Convert to a string representation
        output_data = json.dumps(transcripts, ensure_ascii=False, indent=4) 
        #ensure_ascii=False outputs non-ASCII characters as-is #indent for pretty formatting
        # break # stops the loop, looks at just one instead of all files


        # Write to a text file
        with open('all-data.txt', 'w') as t:
            t.write(output_data)

print(type(output_data)) #str
print(type(transcripts)) #list
print(len(transcripts)) #229




# https://www.geeksforgeeks.org/extract-multiple-json-objects-from-one-file-using-python/
    #     file_cont = f.read()
    # pattern = r'(?:transcript)'
    # # find all JSON Objects from a file by passing re pattern
    # json_objs = re.findall(pattern, file_cont)
    # # parse each JSON object
    # for obj_string in json_objs:
    #     obj = json.loads(obj_string)
    #     print(obj)
    #json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)




    # for object in json_data['results']:
    #     transcript = object['alternatives']
    #     relevant_dict = filter(lambda x: x['transcript'], object['alternatives'])[0]
    #     get = relevant_dict['transcript'][0]
    #     print(get)
    # table.append([name, get])



