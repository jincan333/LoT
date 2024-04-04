import torch.nn as nn


class RNNModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # Token2Embeddings
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout) #(seq_len, batch_size, emb_size)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # input size(bptt, bsz)
        emb = self.drop(self.encoder(input))
        # emb size(bptt, bsz, embsize)
        # hid size(layers, bsz, nhid)
        output, hidden = self.rnn(emb, hidden)
        # output size(bptt, bsz, nhid)
        output = self.drop(output)
        # decoder: nhid -> ntoken
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, bsz):
        # LSTM h and c
        weight = next(self.parameters()).data
        return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)