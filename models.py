import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder (LSTM)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pre_trained_embeddings,
                 n_layers=1, bidirectional=False, dropout=0.1):
        super(EncoderRNN, self).__init__()

        # Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        # self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding.weight = torch.nn.Parameter(pre_trained_embeddings)
        #self.embedding.weight.requires_grad=False

        self.lstm = nn.LSTM(hidden_size, hidden_size, self.n_layers, bidirectional=bidirectional)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # get word embeddings [timesteps x batch_size x hidden_size]
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)  # pack the sequence
        output, hidden = self.lstm(packed)

        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)  # unpack (back to padded)
        return output, hidden

    def init_hidden(self):
        # directions = 2 if self.bidirectional else 1
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


# In[19]:


# Attention class, different options
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size, max_len, device=device)  # B x 1 x S

        # Calculate energies for each encoder output
        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1
        # Shape == [1 x Batch_Size x N(1D weights)]
        return F.softmax(attn_energies, -1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = energy.transpose(0, 1)
            energy = self.other.mm(energy)
            return energy


# In[20]:


# Decoder


# In[21]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, pre_trained_embeddings, embedding_size=200,
                 n_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Parameters
        # 2 * encoder_hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        # Layers
        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.embedding.weight = torch.nn.Parameter(pre_trained_embeddings)
        #self.embedding.weight.requires_grad=False

        self.dropout = nn.Dropout(self.dropout)
        self.attn_model = attn_model
        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, self.n_layers)
        self.out = nn.Linear(self.hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Run this one-step at a time
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N
        embedded = self.dropout(embedded)

        # Combine embedded input word and last context, run through RNN
        rnn_output, hidden = self.lstm(embedded, last_hidden)
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N
        # print("Context size: ", context.size())
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        # Loung eq.5
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Loung eq.6
        o = self.out(concat_output)
        output = F.log_softmax(o, 1)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights