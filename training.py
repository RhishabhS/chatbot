from utils import tokenize, stem, bagOfWords
import numpy as np
import json
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet
# training hyperparameters
BATCH_SIZE=8
HIDDEN_SIZE=8
LEARNING_RATE=0.01
NUM_EPOCHS=500
logs=open('logs.txt','w')
print('For detailed training logs, refer to logs.txt...')
file=open('data.json','r')
data=file.read()
data=json.loads(data)
vocab_list=[]
vocab={}
training=[]
tag_dict={}
tag_idx=0
stop_words=['.','!',',']
for intent in data['intents']:
    tag=intent['tag']
    patterns=intent['patterns']
    responses=intent['responses']
    if tag not in tag_dict.keys():
        tag_dict[tag]=tag_idx
        tag_idx+=1

    for pattern in patterns:
        token=tokenize(pattern)
        vocab_list.extend([stem(word) for word in token if word not in stop_words])
        training.append((pattern,tag))

print("Data on which model is trained ",file=logs)
print(training,file=logs)
vocab_list=sorted(list(set(vocab_list)))
vocab={word:idx for idx,word in enumerate(vocab_list)}
#create xtrain and ytrain
xtrain=[]
ytrain=[]

print('\nGenerating training data....\n',file=logs)
for x,y in training:
    print('Pattern:',x,file=logs)
    print('tag',y,file=logs)
    x=bagOfWords(tokenize(x),vocab)
    print('Input Vector:',x,file=logs)
    print('tag class:',tag_dict[y],file=logs)
    xtrain.append(x)
    ytrain.append(tag_dict[y])

xtrain=np.array(xtrain,dtype=np.float32)
ytrain=np.array(ytrain,dtype=np.float32)
print("Training Data succesfully generated\n",file=logs)
class ChatDataset(Dataset):
    def __init__(self):
        self.nsamples=len(xtrain)
        self.xdata=xtrain
        self.ydata=ytrain
    
    def __getitem__(self,index):
        return self.xdata[index],self.ydata[index]
    def __len__(self):
        return self.nsamples
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=True)
model=NeuralNet(input_size=len(xtrain[0]),hidden_size=HIDDEN_SIZE,num_classes=len(ytrain)).to(device)

loss=nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
print('Model Training begins....',file=logs)
for epoch in range(NUM_EPOCHS):
    for words,labels in train_loader:
        words=words.to(device)
        labels = labels.type(torch.long)
        labels=labels.to(device)
        outputs=model(words)
        loss_value=loss(outputs,labels)
        optimiser.zero_grad()
        loss_value.backward()
        optimiser.step()
    if((epoch+1)%100==0):
        print(f'epoch: {epoch+1}/{NUM_EPOCHS}, loss: {loss_value.item ():.4f}')
        print(f'epoch: {epoch+1}/{NUM_EPOCHS}, loss: {loss_value.item ():.4f}',file=logs)
print(f'Final: loss={loss_value.item():.4f}')
print(f'Final: loss={loss_value.item():.4f}',file=logs)

data = {
"model_state": model.state_dict(),
"input_size": len(xtrain[0]),
"hidden_size": HIDDEN_SIZE,
"output_size": len(ytrain),
"vocab": vocab,
"tags": list(tag_dict.keys())
}

FILE = "data.pt"
print("data saved: ",data,file=logs)
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
print(f'training complete. file saved to {FILE}',file=logs)

