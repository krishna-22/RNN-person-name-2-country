# importing necessary libraries

from io import open
import os, string, random, time, math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np,torch,torch.nn as nn
from IPython.display import clear_output

#data processing

languages = []
data = []
x = []
y = []
with open('lan.txt', 'r') as f: 
    for line in f:
        line = line.split(',')
        name = line[0].strip()
        lang = line[1].strip()
        if not lang in languages:
            languages.append(lang)
        x.append(name)
        y.append(lang)
        data.append((name, lang))
n_languages = len(languages)
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#rnn module

class RNN_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_net, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_, hidden, batch_size,verbose=False):
        out, hidden = self.rnn_cell(input_, hidden)
        output = self.h2o(hidden.view(-1,self.hidden_size))
        output = self.softmax(output)
        if verbose:
          print(hidden.shape,input_.data.shape)
          print(out.data.shape)
          print(output.shape)
        return output, hidden
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
        
 def batched_name_rep(names, max_word_size):
    rep = torch.zeros(max_word_size, len(names), n_letters)
    for name_index, name in enumerate(names):
        for letter_index, letter in enumerate(name):
            pos = all_letters.find(letter)
            rep[letter_index][name_index][pos] = 1
    return rep
def batched_lang_rep(langs):
    rep = torch.zeros([len(langs)], dtype=torch.long)
    for index, lang in enumerate(langs):
        rep[index] = languages.index(lang)
    return rep
    
def batched_dataloader(npoints,x,y, verbose=False, device = 'cpu'):
    names = []
    langs = []
    X_lengths = []
    
    for i in range(npoints):
        index_ = np.random.randint(len(x))
        name, lang = x[index_], y[index_]
        X_lengths.append(len(name))
        names.append(name)
        langs.append(lang)
    max_length = max(X_lengths)
    
    names_rep = batched_name_rep(names, max_length)
    langs_rep = batched_lang_rep(langs)
    
    #padded_names_rep = torch.nn.utils.rnn.pack_padded_sequence(names_rep, X_lengths, enforce_sorted = False)
    
    print(names)
    #print(padded_names_rep.data.shape,langs_rep.shape)
    return names_rep, langs_rep
    
net=RNN_net(n_letters,128,n_languages)
def infer(net,n_points,x,y,v):
  inputs_,labels=batched_dataloader(n_points,x,y)
  batch_size=n_points
  output,hidden=net(inputs_,net.init_hidden(batch_size),batch_size,v)
infer(net,2,x,y,True)

#training the rnn

def train(net,epochs,x,y,batch_size):
  start=0
  end=batch_size
  size_=len(x)
  steps=int(size_/batch_size)
  loss_={}
  index=1
  criterion = nn.NLLLoss()
  opt =torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
  for i in range(epochs*steps):
    inputs,labels=x[start:end],y[start:end]
    lengths=[]
    opt.zero_grad()
    for j in inputs:
      lengths.append(len(j))
    inputs,labels=batched_name_rep(inputs,max(lengths)),batched_lang_rep(labels)
    output,hidden=net(inputs,net.init_hidden(batch_size),batch_size)
    loss = criterion(output,labels)
    
    loss.backward()
    opt.step()
    if(end==size_):
      start=0
      end=batch_size
    else:
      start=end
      end=start+batch_size
    if int(i%steps)==0:
      loss_[index]=loss.data
      index=index+1
      print("--")
  plt.figure()
  plt.plot(loss_.keys(),loss_.values())
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.show()
train(net,10,x,y,50)
