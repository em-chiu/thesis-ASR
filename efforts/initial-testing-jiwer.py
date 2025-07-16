import jiwer
from jiwer import wer

import whisper
# print(whisper.__file__) #check whisper install
from whisper_normalizer.basic import BasicTextNormalizer
from whisper.tokenizer import get_tokenizer
# from whisper.tokenizer import split_to_word_tokens
from whisper.tokenizer import Tokenizer


import os.path

#testing simple WER
with open('/Users/emily/Desktop/whisper-output-turbo-word/alc02.txt', 'r') as alc02:
    hypothesis = alc02.read()
reference = "Oui, j'ai appris Petit Poucet avec ma maman, et Petit Poucet était un enfant d'une grande famille, mais il était le dernier enfant de la famille. Et tout naturellement il y avait plusieurs frères et soeurs avant lui. Et le papa et la maman pouvaient pas gâter tous les enfants. Il n-avait toujours un autre plus petit qu'avait besoin de leurs soins. Alors Petit Poucet était le dernier. Ils avaient plus de temps et, euh, pour faire pour lui et ils l'ont gâté beaucoup plus que les autres. Et les autres frères et soeurs étaient jaloux que Petit Poucet avait plus d'intention du papa et de la maman, mais Petit Poucet était un enfant beaucoup smart et il observait tout et il pouvait se défendre avec les plus vieux. Fait, un jour et il s'est aperçu que le... les frères et soeurs étaient pas autour. Il a commencé chercher et il les a trouvé au magasin, et il a approché tout doucement et il les a attendu après parler. Ils ont dit, ' On va prendre les chevaux. On va les atteler dessus le wagon. On va aller dans le bois, et on va s'amuser. On va jouer dans le bois et on va laisser Petit Poucet dans le bois, on va le perdre. ' Alors, il était beaucoup smart, comme j'ai dit déjà. Il s'est préparé. Il a ramassé des, des pierres et il a mis dans ses poches jusqu'à qu'il a pu les serrer dans une place convenable pour lui les prendre sans les frères et les soeurs s'aperçoient de trop. Le jour ils ont attelé le wagon et ils ont parti avec deux chevaux attelés dessus le wagon. Ils ont parti dans le bois. Quand ils ont cru qu'ils estiont assez loin, ils ont arrêté puis ils ont commencé jouer cache-et-faite. Et tout naturel lui, il s'est caché un peu loin. Ils ont dit, ' Mais, c'est là le moment pour on part. ' Ils ont parti vite et lui, il a resté dans le bois. Mais tout le long du chemin, tous les temps en temps il lâchait une petite pierre, une petite pierre pour marquer son chemin. Il savait que ils vouliont le perdre. Alors, après eux ils ont parti, il a commencé marcher trouver les pierres où il avait marqué le chemin, et la nuit l'a pris et il était toujours dans le bois et il s'a aperçu il y avait une vieille maison. Il dit, ' J'ai peur de coucher dans le bois moi tout seul. Je vas aller à la maison mander pour rester. ' Quand il a arrivé, il a frappé, une vieille femme qu'a sorti. Il a dit il était perdu et il aurait aimé coucher. Elle lui dit, ' Non, non cher! Tu peux pas coucher ici. ' Elle dit, ' C'est la maison du vieux diable, ' et elle dit, ' Le vieux diable va te manger, ' elle dit ' quand il va revenir. ' Mais il dit, ' Je vas prendre une chance, ' il dit. Il dit, ' J'ai peur dans le bois, ' et il dit ' j'aimerais mieux d'être dans la maison, ' il dit ' prendre une chance. ' Alors la femme l'a accepté. Elle a préparé le souper. Elle a donné à souper à ses petits diables et à lui. Là, elle dit, ' Il faut vous-autres se couche, ' elle dit, ' avant le vieux diable arrive. ' Ça fait, les enfants étaient tous couchés. Mais la maman a pris des petits bonnets, elle a mis à tous ses enfants. Mais lui, il a observé ça. Lui, il en avait pas un. Alors la nuit s'est fait. Il commençait à faire noir. Le vieux diable a arrivé. Il respirait fort. ' Ouh, ' il dit, ' je sens la viande fraîche. ' ' Oh, ' elle dit, ' tu t'imagines de ça, ' elle dit, ' il y a pas de viande fraîche. ' ' Ouh, ' il dit, ' je sens la viande fraîche. ' Elle dit, ' T'as faim, ' elle dit, ' Viens souper, ' elle dit. Puis elle dit, ' on va se coucher de bonne heure. Je suis lasse et j'ai travaillé beaucoup aujourd'hui. ' Alors elle lui a donné son souper. Après souper, ' Oh, ' il dit, ' je sens la viande fraîche toujours. ' ' Oh, ' elle dit, ' tu t'imagines de ça. ' Puis ils se sont couchés, mais pendant que tout ça était après aller, Petit Poucet a pris le bonnet de un des enfants du vieux diable qu'était à peu près sa grosseur à lui et il l'a mis sur sa tête à lui. Alors, après le vieux diable était couché. ' Ouh, ' il dit, ' je sens la viande fraîche toujours. ' Il s'est levé, il a été au lit, mais il faisait noir. Il a passé sa main, touché les petits bonnets. Quand il a arrivé à la tête qu'avait pas un petit bonnet, il a pris l'enfant et il a été dehors et il l'a tué. Mais Petit Poucet avait trop peur, lui, pour dormir. Fait, quand il a vu que le vieux diable avait pris un, lui, il a parti, il a échappé. Mais avant partir il avait vu où le vieux diable avait mis ses bottes et son or, alors il a volé l'or et les bottes à le vieux diable, et il a mis les bottes à le vieux diable il aurait pu marcher beaucoup plus vite avec les bottes. Là quand le jour s'est fait, il a cherché pour trouver son chemin parce qui était marqué avec les pierres. Il l'a trouvé. Alors, il a parti à suivre les pierres, il s'a rentourné chez lui. Quand il a arrivé, ils étaient tout surpris de le voir arriver, mais contents de le recevoir il était riche. Il avait un gros sac d'or et les belles bottes du diable. Fait, tout était magnifique, ils estiont contents de le revoir."

# normalizer = BasicTextNormalizer()
# #removes apostrophes, WER: 19.62%
# # norm_ref = normalizer(reference)
# # norm_hyp = normalizer(hypothesis)



# def split_to_word_tokens(self):
#       return self.split_tokens_on_spaces()
# norm_ref = Tokenizer.split_to_word_tokens(reference)
# norm_hyp = Tokenizer.split_to_word_tokens(hypothesis)
# # assert whisper_tokenizer.decode(reference) == norm_ref
# # assert whisper_tokenizer.decode(hypothesis) == norm_hyp

# wer = jiwer.wer(norm_ref, norm_hyp)
# print(f"WER: {wer * 100:.2f}%") 


# a) pandas read in tsv file for whisper output?
# b1) open all files and 
# b2) append to empty hyp/ref lists
# b3) load lists into dataframe
# b4) clean/normalize dataframe columns
#to do:normalize text
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
]) 
## OR normalizer + additional jiwer.transformations?
# normalizer = BasicTextNormalizer()


# transformation did improve WER (31.18) than w/o but is it doing all it needs to do?
# wer_result = jiwer.wer(
#     reference, 
#     hypothesis, 
#     reference_transform=transformation, 
#     hypothesis_transform=transformation
# )



# https://colab.research.google.com/gist/kurianbenoy/7d27d9ec193a4a97ec7821235bddc506/hello-world_whisper_normalizer.ipynb#scrollTo=7lmDPPGVqVW2
# normalizer sample

# https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=dl-KBDflMhrg
# whisper sample

# if __name__ == "__main__":
#     reference = "hello world"
#     hypothesis = "hello duck"
#     error = wer(reference, hypothesis)
#     print(error)

# wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

# wer_contiguous = tr.Compose(
#     [
#         tr.RemoveMultipleSpaces(),
#         tr.Strip(),
#         tr.ReduceToSingleSentence(),
#         tr.ReduceToListOfListOfWords(),
#     ]
# )

# output = jiwer.wer_contiguous(reference, hypothesis)
# wer = output.wer

# print(file_name)



