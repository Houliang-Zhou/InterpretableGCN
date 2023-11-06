import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
# from torch.autograd import Variable
# from torch_geometric.utils import to_dense_batch
# from pytorch_util import weights_init, gnn_spmm
# from torch.nn.parameter import Parameter, UninitializedParameter
# from torch.nn import init
import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.nn import knn_graph
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
# from torchdiffeq import odeint_adjoint as odeint
import utils



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, encoder_output_size, hidden_size, output_size, Pred_LENGTH=140, device=None):
        super(DecoderRNN, self).__init__()
        self.Pred_LENGTH=Pred_LENGTH
        self.device=device
        self.embedding = nn.Linear(encoder_output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, encoder_output_size)
        self.regress_out = nn.Linear(encoder_output_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        # decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)

        decoder_input = encoder_outputs[:, -1:, :]
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.Pred_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output.detach().clone()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.cat((encoder_outputs, decoder_outputs), dim=1)
        decoder_outputs = self.regress_out(decoder_outputs)

        # decoder_outputs = self.regress_out(encoder_outputs)

        return decoder_outputs # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class RNNAutoRegressive(nn.Module):
    def __init__(self, encoder_input_size, encoder_hidden_size, decoder_hidden_size, decoder_output_size=1, encoder_dropout_p=0.1, Pred_LENGTH=40, device=None):
        super(RNNAutoRegressive, self).__init__()
        self.encoder = EncoderRNN(encoder_input_size, encoder_hidden_size, encoder_dropout_p)
        self.decoder = DecoderRNN(encoder_hidden_size, decoder_hidden_size, decoder_output_size, Pred_LENGTH=Pred_LENGTH, device=device)

    def forward(self, input_tensor):
        B, N, T = input_tensor.size()
        input_tensor = input_tensor.reshape(B * N, -1)
        input_tensor = input_tensor.unsqueeze(-1)
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs = self.decoder(encoder_outputs, encoder_hidden)
        decoder_outputs = decoder_outputs.reshape(B, N, -1)
        return decoder_outputs

if __name__ == '__main__':
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    input_tensor = torch.randn(2,100,100).float()
    model = RNNAutoRegressive(encoder_input_size = 1, encoder_hidden_size=5, decoder_hidden_size=5)
    decoder_outputs = model(input_tensor)
