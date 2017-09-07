import torch
import torch.nn as nn
from torch.autograd import Variable
class GRU(nn.Module)
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #First hidden layer is convolutional layer for low-level feature extraction
        #Kernel size is 5 so it makes receptive field of 5 elementwise multiplication with kernel weights
        self.conv = nn.Conv1d(input_size, hidden_size, 5)

        #Defining two-layer GRU
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, dropout=0.01)

        #Defining output layer
        self.out = nn.Linear(hidden_size, output_size)
        
#http://www.pythonexample.com/code/pytorch%20conv1d/
    def forward(self, inputs, hidden):
        batch_size = inputs.size(1)
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        #Run through conv layer
        c = self.conv(inputs)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        c = c.transpose(1, 2).transpose(0, 1)

        output, hidden = self.gru(c, hidden)
        conv_seq_len = output.size(0)
        output = output.view(conv_seq_len * batch_size, self.hidden_size) # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = F.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.hidden_size))
