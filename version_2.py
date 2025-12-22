import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
n_embed = 32
vocab_size = 65
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# initial testing before any training 

# print("".join(chars))
# print(vocab_size)


#mapping characters: the reason we do it is because neural networks only know integers so to represent a character in integer
# we have to encode each cahracter and gvie it a value and then be able to decode it as well we do this as below

char_vals = {ch:i for i,ch in enumerate(chars)}
vals_char = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [char_vals[c] for c in s]
decode = lambda l: "".join([vals_char[i] for i in l])

name = "what is your name shawty"
enc_name = encode(name)
# print(enc_name)
# print(decode(enc_name))

# Now we are going to change this data to PyTorch tensors because that is what our model would want us to feed it


data = torch.tensor(encode(text), dtype= torch.long)

# print(data.shape)
# print(data[:100])


# Here we are going to split the data into training data and testing data

n = int(0.8*len(data))

training_data = data[:n]
testing_data = data[n:]

# print(testing_data)

# we are not going to feed the model all the training data at once because that would be computationally expensive and
# I am running a free version of Colab so we would split the data into chunks and block size and feed it to the model

block_size = 8

# print(training_data[:block_size+1])

x = training_data[:block_size]
y = training_data[1:block_size+1]


for i in range(block_size):
    context = x[:i+1]
    target = y[i]

    # I personally don't understand this part quite so I am trying to print it here and visualize it
    # print(f"when input is {context} the target: {target}")


# Now we are going to use batches

torch.manual_seed(1337)

batch_size = 4
block_size = 8 # Already intialized

def get_batch(split):
    data = training_data if split == "train" else testing_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

xb, yb = get_batch("train")

for b in range(batch_size):
    for t in range(block_size):
      context = xb[b,:t+1]
      target = yb[b,t]
    #   print(f"when input is {context.tolist()} the target: {target}")


# creating a self attention one head

class Head(nn.Module):
   
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
      
      B, T, C = x.shape
      k = self.key(x)   # (B, T, head_size)
      q = self.query(x) # (B, T, head_size)

      # compute attention scores
      wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=-1) # (B, T, T)

      # now performing weighted summation
      v = self.value(x)
      out = wei @ v # (B, T, head_size)
      return out


# we will start with creating the most basic form of neural network which is a bigram model
# A simple explanation of the bigram model: only used for words names and next character prediction in a way that
# doesn't need a lot of context. So it ignores the context and history and only uses the current word for prediction
# How it does that? Using markove assumption so P(A|B) = Count(A, B)/Count(A)

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        #instantiating the linear model head
        self.sa_head = Head(n_embed)
        self.lnmdl_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape
        
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x= token_emb + pos_embd
        x= self.sa_head(x)
        logits = self.lnmdl_head(x) # (B, T, vocab_size)
        


        # loss function

        if targets is None: # Corrected: 'target' changed to 'targets'
          loss = None
        else:
          B, T, C = logits.shape
          logits = logits.view(B*T, C)
          targets = targets.view(B*T)
          loss = F.cross_entropy(logits, targets)

        return logits, loss


    # Here we will write a generate function that will generate tokens from the Bigram Model
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            #controlling so we don't get an overflow of context
            idx_cond = idx[:, -block_size:]

            # Here we get the predictions
            logits, loss = self.forward(idx_cond)

            # Here we tell the model to only focus on the last time stamp
            logits = logits[:, -1, :]

            # Here we use softmax for probability
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sample index to the running sequenct
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



m = BigramLanguageModel()
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))


# Now we are going to train the Bigram Model as the text it outputed is really bad
# for this we will be using PyTorch to optimize the weights and biases of the model

optimizer = torch.optim.AdamW(m.parameters(), learning_rate)


batch_size = 32

for steps in range(20000):
  xb, yb = get_batch("train")

  logits, loss = m(xb, yb)

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()



print(loss.item())


idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))

