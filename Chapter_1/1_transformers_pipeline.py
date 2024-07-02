#%%
from transformers import pipeline
import pandas as pd 
import torch

# Set the device to GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#Original text from the book
text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

text_interstellar = '''
Sometimes I just need to see the start. Or end. Or a trailer.
Or the music and theme from Hans Zimmer. Or the whole movie.
Just to feel that thing, I only get from this movie. 
That the earth, space and time are something special, mystical.
I never forget the first time I saw this movie, in an IMAX theatre in 2014. 
I was struck by it. Totally got me. And it stil does, 7 years later. This is the best movie 
ever made for me. Because of the feeling it gives me, no other movie can. So hard to get all 
of this emotion in only one movie. Brilliant.
'''
# The base models are trained on Englisht text, there is not a check to
# verify the language and load a multilingual model. 
text_interstellar_cat = '''
Que ens arribin des de Hollywood cintes tan espectaculars, originals 
i volgudament èpiques com INTERSTELLAR és sens dubte un miracle del que ens hem d’alegrar 
en els temps que corren. Actualment, es poden comptar amb la mà els directors que com Christopher Nolan, 
aposten per històries pròpies, grans formats cinematogràfics i un respecte reverencial a la 
misteriosa atracció cap al cinema que tots sentim. Trobar directors que ho facin dins del sistema 
de Hollywood amb èxit i grans pressupostos és més difícil si cap. Per això Nolan és un dels 
reis del moment; un director únic, privilegiat i com tot autor, admirat i criticat a parts iguals.
'''

#%%
### Text Classification ###
classifier = pipeline('text-classification', device=device)
outputs = classifier(text)
pd.DataFrame(outputs)

#%%
### Named Entity Recognition ###
ner_tagger = pipeline('ner', aggregation_strategy='simple', device=device)
outputs = ner_tagger(text)
pd.DataFrame(outputs)

# %%
### Question answreing ###
reader = pipeline('question-answering', device=device)
question = 'What did Bumblebee order from Amazon?'
question = 'What does the customer want?'
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
# %%
### Summarization ###
summarizer = pipeline('summarization', device=device)
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
# %%
### Translation ###
translator = pipeline('translation_end_to_end', 
                      model = 'Helsinki-NLP/opus-mt-en-de',
                      device=device)
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])
# %%
###Text Generation ###
generator = pipeline('text-generation', device=device)
respone = 'Dear Bumblebee, we are sorry for the inconvenience. We will proceed to '
prompt = text + "\n\nCustomer service response:\n" + respone
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
# %%
