from torch import nn
import torch
from torch.autograd import Variable

    
class UARNet(nn.Module):
  def __init__(self, n_chan, l_batch, n_hidden = 10, n_rnn_layers = 1, n_fc_layers = 1, type_rnn = 'rnn',non_linearity = 1):
    super(UARNet, self).__init__()
    self.n_hidden = n_hidden
    self.n_rnn_layers = n_rnn_layers
    self.n_fc = n_fc_layers
    self.type_rnn = type_rnn
    self.non_linearity = non_linearity
    if type_rnn == 'rnn':
        self.rnn = nn.RNN(n_chan, self.n_hidden, self.n_rnn_layers, batch_first=True)
    elif type_rnn == 'lstm':
        self.rnn = nn.LSTM(n_chan, self.n_hidden, self.n_rnn_layers, batch_first=True)
    elif type_rnn == 'gru':
        self.rnn = nn.GRU(n_chan, self.n_hidden, self.n_rnn_layers, batch_first=True)
    else:
        raise AssertionError('only rnn, gru and lstm are supported')
        
    self.fc = []
    for i in range(0,n_fc_layers-1):
        self.fc.append(nn.Linear(self.n_hidden,self.n_hidden))
    self.fc.append(nn.Linear(self.n_hidden,1))
    
  def forward(self,input):
    output, h = self.rnn(input)
    for linear_layer in self.fc:
        output = linear_layer(output)
    output = torch.sigmoid(output**self.non_linearity)

    return output

class MiniOnsetRNN(nn.Module):
  def __init__(self, n_chan,l_batch):
    super(MiniOnsetRNN, self).__init__()
    self.n_hidden= 10 #15
    self.n_layers= 1 #1
    self.rnn = nn.RNN(n_chan, self.n_hidden, self.n_layers, batch_first=True)
    self.fc = nn.Linear(self.n_hidden,1)
  def forward(self,input):
    x, h = self.rnn(input)
    output = torch.sigmoid(self.fc(x))**4

    return output