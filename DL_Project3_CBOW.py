#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import DATASETS
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
from tqdm import tqdm
import pickle
import random
import numpy as np
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import gensim.downloader
from torch import FloatTensor as FT

# Get the interactive Tools for Matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')


# In[2]:


# DEVICE = "mps" if torch.backends.mps.is_available() else  "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else  "cpu"

# The batch size in Adam or SGD
BATCH_SIZE = 512

# Number of epochs
NUM_EPOCHS = 10

# Predict from 2 words the inner word for CBOW
# I.e. I'll have a window like ["a", "b", "c"] of continuous text (each is a word)
# We'll predict each of wc = ["a", "c"] from "b" = wc for Skip-Gram
# For CBOW, we'll use ["a", "c"] to predict "b" = wo
WINDOW = 1

# Negative samples.
K = 4


# In[3]:


# data in Google Drive
from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('du -h text8')

f = open('/content/drive/MyDrive/sh/text8-1', 'r')
text = f.read()
# One big string of size 100M
print(len(text))


# In[4]:


punc = '!"#$%&()*+,-./:;<=>?@[\\]^_\'{|}~\t\n'

# Can do regular expressions here too
for c in punc:
    if c in text:
        text.replace(c, ' ')


# In[5]:


TOKENIZER = get_tokenizer("basic_english")


# In[6]:


words = TOKENIZER(text)
f = Counter(words)


# In[7]:


len(words)


# In[8]:


# a very crude filter on the text which removes all very popular words
text = [word for word in words if f[word] > 5]


# In[9]:


text[0:5]


# In[10]:


VOCAB = build_vocab_from_iterator([text])


# In[11]:


# word -> int hash map
stoi = VOCAB.get_stoi()
# int -> word hash map
itos = VOCAB.get_itos()


# In[12]:


stoi['as']


# In[13]:


# Total number of words
len(stoi)


# In[14]:


f = Counter(text)
# This is the probability that we pick a word in the corpus
z = {word: f[word] / len(text) for word in f}


# In[15]:


threshold = 1e-5
# Probability that word is kept while subsampling
# This is explained here and sightly differet from the paper: http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
p_keep = {word: (np.sqrt(z[word] / 0.001) + 1)*(0.0001 / z[word]) for word in f}


# In[16]:


# This is in the integer space
train_dataset = [word for word in text if random.random() < p_keep[word]]

# Rebuild the vocabulary
VOCAB = build_vocab_from_iterator([train_dataset])


# In[17]:


len(train_dataset)


# In[18]:


# word -> int mapping
stoi = VOCAB.get_stoi()
# int -> word mapping
itos = VOCAB.get_itos()


# In[19]:


# The vocabulary size after we do all the filters
len(VOCAB)


# In[20]:


# The probability we draw something for negative sampling
f = Counter(train_dataset)
p = torch.zeros(len(VOCAB))

# Downsample frequent words and upsample less frequent
s = sum([np.power(freq, 0.75) for word, freq in f.items()])

for word in f:
    p[stoi[word]] = np.power(f[word], 0.75) / s


# In[21]:


# Map everything to integers
train_dataset = [stoi[word] for word in text]


# In[22]:


# This just gets the (wc, wo) pairs that are positive - they are seen together!
def get_tokenized_dataset(dataset, verbose=False):
    x_list = []
    for i, token in enumerate(dataset):
        m = 1

        # Get the left and right tokens
        start = max(0,i-m)
        left_tokens = dataset[start:i]

        end = min(len(dataset)-1,i+m)
        right_tokens = dataset[i+1:end+1]

        # Check these are the same length, and if so use them to add a row of data. This should be a list like
        # [a, c, b] where b is the center word
        if len(left_tokens) == len(right_tokens):
            w_context = left_tokens + right_tokens

            wc = token

            x_list.extend(
                [[w_context[0],w_context[1], wc]]
            )

    return x_list


# In[23]:


train_x_list = get_tokenized_dataset(train_dataset, verbose=False)


# In[24]:


pickle.dump(train_x_list, open('train_x_list.pkl', 'wb'))


# In[25]:


train_x_list = pickle.load(open('train_x_list.pkl', 'rb'))


# In[26]:


# These are (wc, wo) pairs. All are y = +1 by design
train_x_list[:10]


# In[27]:


# The number of things of BATCH_SIZE = 512
assert(len(train_x_list) // BATCH_SIZE == 32579)


# ### Set up the dataloader.

# In[28]:


train_dl = DataLoader(
    TensorDataset(
        torch.tensor(train_x_list).to(DEVICE),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# In[29]:


for xb in train_dl:
    assert(xb[0].shape == (BATCH_SIZE, 3))
    break


# ### Words we'll use to asses the quality of the model ...

# In[30]:


valid_ids = torch.tensor([
    stoi['money'],
    stoi['lion'],
    stoi['africa'],
    stoi['musician'],
    stoi['dance'],
])


# ### Get the model.

# In[31]:


class CBOWNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOWNegativeSampling, self).__init__()
        self.A = nn.Embedding(vocab_size, embed_dim) # Context vectors - center word
        self.B = nn.Embedding(vocab_size, embed_dim) # Output vectors - words around the center word
        self.init_weights()

    def init_weights(self):
        # Is this the best way? Not sure
        initrange = 0.5
        self.A.weight.data.uniform_(-initrange, initrange)
        self.B.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # N is the batch size
        # x is (N, 3)

        # Context words are 2m things, m = 1 so w_context is (N, 2) while wc is N

        if x.shape[1]==(3):
            w_context, wc = x[:,:2], x[:,2]
        else:
            print(x)
            print(x.shape)
            # w_context, wc = x

        w_context, wc = x[:,:2], x[:,2]

        # Each of these is (N, 2, D) since each context has 2 word
        # We want this to be (N, D) and this is what we get

        # (N, 2, D)
        a = self.A(w_context)

        # (N, D)
        a_avg = a.mean(dim=1)

        # Each of these is (N, D) since each target has 1 word
        b = self.B(wc)

        # The product between each context and target vector. Look at the Skip-Gram code.
        # The logits is now (N, 1) since we sum across the final dimension.
        logits = (a_avg*b).sum(axis = -1)

        return logits


# In[33]:


@torch.no_grad()
def validate_embeddings(
    model,
    valid_ids,
    itos
):
    """ Validation logic """

    # We will use context embeddings to get the most similar words
    # Other strategies include: using target embeddings, mean embeddings after avaraging context/target
    embedding_weights = model.A.weight

    normalized_embeddings = embedding_weights.cpu() / np.sqrt(
        np.sum(embedding_weights.cpu().numpy()**2, axis=1, keepdims=True)
    )

    # Get the embeddings corresponding to valid_term_ids
    valid_embeddings = normalized_embeddings[valid_ids, :]

    # Compute the similarity between valid_term_ids (S) and all the embeddings (V)
    # We do S x d (d x V) => S x D and sort by negative similarity
    top_k = 10 # Top k items will be displayed
    similarity = np.dot(valid_embeddings.cpu().numpy(), normalized_embeddings.cpu().numpy().T)

    # Invert similarity matrix to negative
    # Ignore the first one because that would be the same word as the probe word
    similarity_top_k = np.argsort(-similarity, axis=1)[:, 1: top_k+1]

    # Print the output.
    for i, word_id in enumerate(valid_ids):
        # j >= 1 here since we don't want to include the word itself.
        similar_word_str = ', '.join([itos[j] for j in similarity_top_k[i, :] if j >= 1])
        print(f"{itos[word_id]}: {similar_word_str}")

    print('\n')


# ### Set up the model

# In[34]:


LR = 10.0
NUM_EPOCHS = 10
EMBED_DIM = 300


# In[35]:


model = CBOWNegativeSampling(len(VOCAB), EMBED_DIM).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# The learning rate is lowered every epoch by 1/10.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)


# In[36]:


model


# In[37]:


validate_embeddings(model, valid_ids, itos)


# ### Train the model

# In[40]:


ratios = []

def train(dataloader, model, optimizer, epoch):
    model.train()
    total_acc, total_count, total_loss, total_batches = 0, 0, 0.0, 0.0
    log_interval = 500

    for idx, x_batch in tqdm(enumerate(dataloader)):

        x_batch = x_batch[0]

        # if x_batch.shape[1] != 3: ### check
        #   print(x_batch.shape, x_batch) ### check

        batch_size = x_batch.shape[0]

        # Zero the gradient so they don't accumulate
        optimizer.zero_grad()

        logits = model(x_batch)

        # Get the positive samples loss. Notice we use weights here
        positive_loss = torch.nn.BCEWithLogitsLoss()(input=logits, target=torch.ones(batch_size).to(DEVICE).float())

        # For each batch, get some negative samples
        # We need a total of len(y_batch) * K samples across a batch
        # We then reshape this batch
        # These are effectively the output words
        negative_samples = torch.multinomial(p, batch_size * K, replacement=True).to(DEVICE)

        # Context words are 2m things, m = 1 so w_context is (N, 2) while wc is (N, )
        w_context, wc = x_batch[:, :2], x_batch[:2]

        """
        if w_context looks like below (batch_size = 3)
        [
        (a, b),
        (c, d),
        (e, f)
        ] and K = 2 we'd like to get:

        [
        (a, b),
        (a, b),
        (c, d),
        (c, d),
        (e, f),
        (e, f)
        ]

        This will be batch_size * K rows.
        """

        # This should be (N * K, 2)
        w_context = torch.concat([
            w.repeat(K, 1) for w in torch.tensor(w_context).split(1)
        ])

        # Add a last dimension and set wc to the negative samples
        wc = negative_samples.unsqueeze(1)

        # Get the negative samples. This should be (N * K, 3)
        # Concatenate the w_context and wc along the column. Make sure everything is on CUDA / MPS or CPU
        x_batch_negative = torch.cat([w_context,wc],dim = 1).to(DEVICE)

        """
        Note the way we formulated the targets: they are all 0 since these are negative samples.
        We do the BCEWithLogitsLoss by hand basically here.
        Notice we sum across the negative samples, per positive word.

        This is literally the equation in the lecture notes.
        """

        # (N, K, D) -> (N, D) -> (N)
        # Look at the Skip-Gram notebook
        negative_loss = model(x_batch_negative).neg().sigmoid().log().reshape(
            batch_size, K).sum(1).mean().neg().to(DEVICE)

        loss = (positive_loss + negative_loss).mean()

        # Get the gradients via back propagation
        loss.backward()

        # Clip the gradients? Generally a good idea
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # Used for optimization. This should be roughly 0.001, on average
        with torch.no_grad():
            r = [
                (LR * p.grad.std() / p.data.std()).log10().item() for _, p in model.named_parameters()
            ]
            ratios.append(r)

        # Do an optimization step. Update the parameters A and B
        optimizer.step()

        # Get the new loss.
        total_loss += 1.0 * loss.item()

        # Update the batch count
        total_batches += 1

        if idx % log_interval == 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| loss {:8.3f} ".format(
                    epoch,
                    idx,
                    len(dataloader),
                    total_loss / total_batches
                )
            )
            validate_embeddings(model, valid_ids, itos)
            total_loss, total_batches = 0.0, 0.0


# ### Some results from the run look like below:

# In[41]:


for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()

    train(train_dl, model, optimizer, epoch)
    # We have a learning rate scheduler here
    # Basically, given the state of the optimizer, this lowers the learning rate in a smart way
    scheduler.step()


# In[109]:


encoderA = model.A.weight
encoderB = model.B.weight

### Matrix A times the one-hot encodings of w just mean selecting the wth column of A.t()
validation_vectors_A = encoderA[valid_ids].t()
validation_vectors_B = encoderB[valid_ids].t()
print('using matrix A, the associated vectors for the validation words are:')
print(validation_vectors_A)
print('using matrix B, the associated vectors for the validation words are:')
print(validation_vectors_B)
print(validation_vectors.shape)


# In[113]:


logit_scores = torch.mm(encoderA,validation_vectors_A)

probabilities = torch.softmax(logit_scores, dim=1)

most_likely_one_hot_indices = torch.argmax(probabilities, dim=0)

print('generate the most likely one-hot encodings from the vectors: ')
print(most_likely_one_hot_indices)

print('the most predicted words from these one-hot encodings should match the validation words')
predicted_words = [itos[index.item()] for index in most_likely_one_hot_indices]
print(predicted_words)


# In[118]:


file_path_A = 'encoderA.pt'
torch.save(encoderA, file_path_A)
loaded_tensor_A = torch.load(file_path_A)
loaded_tensor_A

file_path_B = 'encoderB.pt'
torch.save(encoderB, file_path_B)
loaded_tensor_B = torch.load(file_path_B)
loaded_tensor_B


# In[122]:


loaded_tensor_A = torch.load('/content/drive/MyDrive/sh/encoderA.pt')
loaded_tensor_B = torch.load('/content/drive/MyDrive/sh/encoderB.pt')
loaded_tensor_B


# In[123]:


encoderB

