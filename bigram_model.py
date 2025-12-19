import torch
import torch.nn as nn
from torch.nn import functional as F


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
    print(f"when input is {context} the target: {target}")


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
      print(f"when input is {context.tolist()} the target: {target}")


# we will start with creating the most basic form of neural network which is a bigram model
# A simple explanation of the bigram model: only used for words names and next character prediction in a way that
# doesn't need a lot of context. So it ignores the context and history and only uses the current word for prediction
# How it does that? Using markove assumption so P(A|B) = Count(A, B)/Count(A)

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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

            # Here we get the predictions
            logits, loss = self.forward(idx)

            # Here we tell the model to only focus on the last time stamp
            logits = logits[:, -1, :]

            # Here we use softmax for probability
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sample index to the running sequenct
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))


# Now we are going to train the Bigram Model as the text it outputed is really bad
# for this we will be using PyTorch to optimize the weights and biases of the model

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


batch_size = 32

for steps in range(20000):
  xb, yb = get_batch("train")

  logits, loss = m(xb, yb)

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  optimizer.step()


print(loss.item())


idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))