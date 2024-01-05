import random
import json
from model import NeuralNet
import torch
from utils import bagOfWords, tokenize
import numpy as np
import time
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file=open('data.json','r')
data=file.read()
data=json.loads(data)
modelFile= 'data.pt'
'''
"input_size": len(xtrain[0]),
"hidden_size": HIDDEN_SIZE,
"output_size": len(ytrain),
"vocab": vocab,
"tags": tags
'''

savedModel=torch.load(modelFile)
INPUT_SIZE=savedModel['input_size']
HIDDEN_SIZE=savedModel['hidden_size']
OUTPUT_SIZE=savedModel['output_size']
vocab=savedModel['vocab']
model_state=savedModel['model_state']
tags=savedModel['tags']

model= NeuralNet(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE)
model.load_state_dict(model_state)
model.eval()

bot_name='rhish san'
print("Hi, I'm rhish san and I'm here to address any queries you might have regarding our store. Let's Chat! (type q to exit)")
while(True):
    sentence = input("You: ")
    if sentence == "q":
        break
    sentence = tokenize(sentence)
    X = np.array(bagOfWords(sentence, vocab),dtype=np.float32)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    model=model.to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in data['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                if(tag=='goodbye'):
                    time.sleep(1)
                    exit()
    else:
        print(f"{bot_name}: I do not understand...")
