# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))

for i in inputs:
    print ('i',i.shape)
    print ('i.view',i.view(1,1,-1).shape)
    out, hidden = lstm(i.view(1, 1, -1), hidden)

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print("inputs",inputs.shape)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)

print(out)
print(hidden)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

print(training_data)

word_to_ix = {}

for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, batch_size, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        print('embedding dim',embedding_dim)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(2, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, 1)#tagset_size)
        self.sig = nn.Sigmoid()

    def forward(self, sentence):
        #sentence size 4
        #embeds size 4x6
        #embeds view 4x1x6
        #input needs to be 1000x1x2
        #embeds = self.word_embeddings(sentence)
        #print(torch.max(embeds))
        #embeds 
        ##print('embeds',embeds.shape)
        ##print('embedsview',embeds.view(len(sentence),1,-1).shape)
        #assert sentence.shape == [1000,1,2]
        print(sentence.shape)
        lstm_out, hidden = self.lstm(sentence)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #tag_scores = F.log_softmax(tag_space, dim=1)
        sig_out = self.sig(tag_space)
        
        #reshape
        sig_out = sig_out.view(batch_size,-1)
        sig_out = sig_out[:,-1] #getlast batch label

        return sig_out, hidden

batch_size = 1
model = LSTMTagger(1, EMBEDDING_DIM, HIDDEN_DIM)
#loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range (100):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix)
        #targets = prepare_sequence(tags, tag_to_ix)
        #print('targets',targets.shape)
        # Step 3. Run our forward pass.
        #tag_scores = model(sentence_in)
        input = None
        targets = None
        
        ind = torch.FloatTensor([[x for x in range(1000)],[x for x in range(1000)]])
        ind = torch.reshape(ind,(1000,1,2))
        if i%2 == 0:
            #input = torch.sin((1000,1,2))
            input = torch.sin(ind)
            targets = torch.ones((1))
        else:
            input = torch.zeros((1000,1,2))
            targets = torch.zeros((1))

        scores, hidden = model(input)
        print('scores shape', scores.shape)
        print('target shape', targets.shape)
        #targets = torch.ones(scores.shape)
        
        print("scores", scores)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        #loss = loss_function(scores, targets)
        loss = criterion(scores,targets.float())
        loss.backward()
        optimizer.step()
#https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948
# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

