import jiwer
from jiwer import wer

import whisper
# print(whisper.__file__) #check whisper install
# from whisper.tokenizer import get_tokenizer


# files = ['alc02', 'alc03', 'alc04', 'alc05a', 'alc05b', 'alc05c', 'alc06', 'alc07', 'bac01', 'bac02', 'bac03', 'bac04', 'bac05', 'bac06', 'bac07', 'bac08', 'bac09', 'bac10', 'bac11', 'bac12', 'bac13', 'bac14', 'bac15', 'bac16', 'drc01', 'drc02a', 'drc02b', 'drc03', 'drc05', 'drc06', 'mpc01a', 'mpc01b', 'mpc02', 'mpc03', 'mpc04']

#testing simple WER
with open('/Users/emily/Desktop/whisper-output-turbo-word/alc02.txt', 'r') as alc02:
    hypothesis = alc02.read()
reference = "Oui, j'ai appris Petit Poucet avec ma maman, et Petit Poucet était un enfant d'une grande famille, mais il était le dernier enfant de la famille. Et tout naturellement il y avait plusieurs frères et soeurs avant lui. Et le papa et la maman pouvaient pas gâter tous les enfants. Il n-avait toujours un autre plus petit qu'avait besoin de leurs soins. Alors Petit Poucet était le dernier. Ils avaient plus de temps et, euh, pour faire pour lui et ils l'ont gâté beaucoup plus que les autres. Et les autres frères et soeurs étaient jaloux que Petit Poucet avait plus d'intention du papa et de la maman, mais Petit Poucet était un enfant beaucoup smart et il observait tout et il pouvait se défendre avec les plus vieux. Fait, un jour et il s'est aperçu que le... les frères et soeurs étaient pas autour. Il a commencé chercher et il les a trouvé au magasin, et il a approché tout doucement et il les a attendu après parler. Ils ont dit, ' On va prendre les chevaux. On va les atteler dessus le wagon. On va aller dans le bois, et on va s'amuser. On va jouer dans le bois et on va laisser Petit Poucet dans le bois, on va le perdre. ' Alors, il était beaucoup smart, comme j'ai dit déjà. Il s'est préparé. Il a ramassé des, des pierres et il a mis dans ses poches jusqu'à qu'il a pu les serrer dans une place convenable pour lui les prendre sans les frères et les soeurs s'aperçoient de trop. Le jour ils ont attelé le wagon et ils ont parti avec deux chevaux attelés dessus le wagon. Ils ont parti dans le bois. Quand ils ont cru qu'ils estiont assez loin, ils ont arrêté puis ils ont commencé jouer cache-et-faite. Et tout naturel lui, il s'est caché un peu loin. Ils ont dit, ' Mais, c'est là le moment pour on part. ' Ils ont parti vite et lui, il a resté dans le bois. Mais tout le long du chemin, tous les temps en temps il lâchait une petite pierre, une petite pierre pour marquer son chemin. Il savait que ils vouliont le perdre. Alors, après eux ils ont parti, il a commencé marcher trouver les pierres où il avait marqué le chemin, et la nuit l'a pris et il était toujours dans le bois et il s'a aperçu il y avait une vieille maison. Il dit, ' J'ai peur de coucher dans le bois moi tout seul. Je vas aller à la maison mander pour rester. ' Quand il a arrivé, il a frappé, une vieille femme qu'a sorti. Il a dit il était perdu et il aurait aimé coucher. Elle lui dit, ' Non, non cher! Tu peux pas coucher ici. ' Elle dit, ' C'est la maison du vieux diable, ' et elle dit, ' Le vieux diable va te manger, ' elle dit ' quand il va revenir. ' Mais il dit, ' Je vas prendre une chance, ' il dit. Il dit, ' J'ai peur dans le bois, ' et il dit ' j'aimerais mieux d'être dans la maison, ' il dit ' prendre une chance. ' Alors la femme l'a accepté. Elle a préparé le souper. Elle a donné à souper à ses petits diables et à lui. Là, elle dit, ' Il faut vous-autres se couche, ' elle dit, ' avant le vieux diable arrive. ' Ça fait, les enfants étaient tous couchés. Mais la maman a pris des petits bonnets, elle a mis à tous ses enfants. Mais lui, il a observé ça. Lui, il en avait pas un. Alors la nuit s'est fait. Il commençait à faire noir. Le vieux diable a arrivé. Il respirait fort. ' Ouh, ' il dit, ' je sens la viande fraîche. ' ' Oh, ' elle dit, ' tu t'imagines de ça, ' elle dit, ' il y a pas de viande fraîche. ' ' Ouh, ' il dit, ' je sens la viande fraîche. ' Elle dit, ' T'as faim, ' elle dit, ' Viens souper, ' elle dit. Puis elle dit, ' on va se coucher de bonne heure. Je suis lasse et j'ai travaillé beaucoup aujourd'hui. ' Alors elle lui a donné son souper. Après souper, ' Oh, ' il dit, ' je sens la viande fraîche toujours. ' ' Oh, ' elle dit, ' tu t'imagines de ça. ' Puis ils se sont couchés, mais pendant que tout ça était après aller, Petit Poucet a pris le bonnet de un des enfants du vieux diable qu'était à peu près sa grosseur à lui et il l'a mis sur sa tête à lui. Alors, après le vieux diable était couché. ' Ouh, ' il dit, ' je sens la viande fraîche toujours. ' Il s'est levé, il a été au lit, mais il faisait noir. Il a passé sa main, touché les petits bonnets. Quand il a arrivé à la tête qu'avait pas un petit bonnet, il a pris l'enfant et il a été dehors et il l'a tué. Mais Petit Poucet avait trop peur, lui, pour dormir. Fait, quand il a vu que le vieux diable avait pris un, lui, il a parti, il a échappé. Mais avant partir il avait vu où le vieux diable avait mis ses bottes et son or, alors il a volé l'or et les bottes à le vieux diable, et il a mis les bottes à le vieux diable il aurait pu marcher beaucoup plus vite avec les bottes. Là quand le jour s'est fait, il a cherché pour trouver son chemin parce qui était marqué avec les pierres. Il l'a trouvé. Alors, il a parti à suivre les pierres, il s'a rentourné chez lui. Quand il a arrivé, ils étaient tout surpris de le voir arriver, mais contents de le recevoir il était riche. Il avait un gros sac d'or et les belles bottes du diable. Fait, tout était magnifique, ils estiont contents de le revoir."


whisper_tokenizer = get_tokenizer(multilingual=False, language = "french")
# whisper_tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, language = "french")
# print(whisper_tokenizer.encode(reference)) 
#into integers
print(whisper_tokenizer.decode(whisper_tokenizer.encode(reference)))
# print(whisper_tokenizer.encode(hypothesis))
print(whisper_tokenizer.decode(whisper_tokenizer.encode(hypothesis)))
#https://github.com/OpenT2S/LlamaVoice/blob/08462bc08b6f281af3d51ed11d91611c1c8821f1/llamavoice/tokenizer/tokenizer.py#L277



# # create template that has placeholder
# # for file in files:
# #     with open(file + '.txt') as datafile:
# #         hypotheses.extend(datafile) # extend() adds each element from the iterable to the list
# #         references.extend(datafile)
# # read files then extend into empty lists (extend is flat list)


# for hyps in glob.glob('/Users/emily/Desktop/whisper-output-turbo-word/*.txt'):
#     # with open(hyps) as hypfile, open(refs) as reffile: #loops through each file #hypfile = file handle
#     with open(hyps) as hypfile: #loops through each file #hypfile = file handle
#         hyp_read = hypfile.read() 
#         hypotheses.append(hyp_read)
# print("hypotheses list:", hypotheses) # check list

# for refs in glob.glob('/Users/emily/Desktop/cd-reference-txt/*.txt'): # # wildcard pattern
#     with open(refs) as reffile: #loops through each file #hypfile = file handle
#         ref_read = reffile.read() 
#         references.append(ref_read)
# print("references list:", references) # check list


#text to digit # converted "dix-neuf cent dix-huit" into "10 910 8" # not helpful when speaking isn't standard, don't use
# string_data['hypothesis_low'] = [alpha2digit(x, "fr") for x in string_data['hypothesis_low']]
# string_data['reference_low'] = [alpha2digit(x, "fr") for x in string_data['reference_low']]
# print(string_data['reference_low'], string_data['file name'])
# text2dig_wer = jiwer.wer(list(string_data["hypothesis_low"]), list(string_data["reference_low"]))
# print(f"WER: {text2dig_wer * 100:.2f} %") #WER: 32.05 % WER worsened after converting to digits