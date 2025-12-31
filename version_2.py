"""
CookGPT - A GPT-style Transformer for Cooking Recipe Generation
================================================================

This is a character-level language model based on the GPT architecture, trained on cooking recipes.
Based on Andrej Karpathy's nanoGPT course, enhanced with full transformer components.

Key Transformer Components Implemented:
1. Multi-Head Self-Attention - Parallel attention mechanisms for richer pattern recognition
2. Feed-Forward Networks - Non-linear transformations after attention
3. Layer Normalization - Stabilizes training of deep networks
4. Residual Connections - Enables gradient flow in deep networks
5. Dropout - Regularization to prevent overfitting

Author: Based on Andrej Karpathy's nanoGPT, modified for cooking recipes
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# These control the size and behavior of our model. Tuned for Mac M2 Air.

batch_size = 16          # How many independent sequences to process in parallel
                          # Smaller than typical (32-64) to fit in M2 memory

block_size = 128          # Maximum context length for predictions
                          # Longer context helps the model understand recipe structure

max_iters = 5000         # Total training iterations (steps)
                          # Balanced between training time and model quality

eval_interval = 500       # How often to evaluate loss on train/val sets
                          # Helps monitor training progress without slowing it down

learning_rate = 3e-4      # Learning rate for optimizer
                          # 3e-4 is a good default for transformers (from GPT-3 paper)

eval_iters = 200          # Number of batches to average for loss estimation
                          # More batches = more accurate estimate, but slower

n_embed = 256             # Embedding dimension (model width)
                          # Each token is represented as a 256-dimensional vector
                          # Larger = more expressiveness, but more compute/memory

n_head = 4                # Number of attention heads in multi-head attention
                          # Multiple heads let the model attend to different aspects
                          # of the input simultaneously (e.g., ingredients vs. steps)

n_layer = 4               # Number of transformer blocks (model depth)
                          # Deeper models can learn more complex patterns
                          # But too deep can be hard to train and slow

dropout = 0.2             # Dropout probability for regularization
                          # During training, randomly zero out 20% of activations
                          # This prevents overfitting by forcing redundancy

# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================
# Configure the compute device for training

# NOTE: MPS (Apple Silicon GPU) can cause "bus error" crashes with certain
# PyTorch operations. CPU is more stable, though slower. Set USE_MPS = True
# if you want to try MPS and accept potential instability.
USE_MPS = False

if USE_MPS and torch.backends.mps.is_available():
    # MPS (Metal Performance Shaders) - Apple Silicon GPU acceleration
    # Can be faster but may crash with some operations
    device = "mps"
elif torch.cuda.is_available():
    # CUDA - NVIDIA GPU acceleration
    device = "cuda"
else:
    # CPU - most stable, works everywhere
    device = "cpu"

print(f"Using device: {device}")

# Set random seed for reproducibility
# This ensures the same random numbers are generated each run
torch.manual_seed(1337)

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================

# Read the training data - cooking recipes
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get all unique characters in the text
# For recipes, this includes letters, numbers, punctuation, newlines, etc.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Total characters in dataset: {len(text):,}")

# ==============================================================================
# TOKENIZATION (Character-Level)
# ==============================================================================
# Neural networks work with numbers, not text. We need to convert characters
# to integers and back. This is called tokenization.
#
# Character-level tokenization is simple but effective:
# - Pros: No unknown tokens, works with any text
# - Cons: Longer sequences, harder to learn long-range dependencies

# Create mappings between characters and integers
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

# Encoder: takes a string, outputs a list of integers
encode = lambda s: [stoi[c] for c in s]

# Decoder: takes a list of integers, outputs a string
decode = lambda l: "".join([itos[i] for i in l])

# ==============================================================================
# TRAIN/VALIDATION SPLIT
# ==============================================================================
# We split our data so we can check if the model generalizes to unseen data

# Convert entire text to a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

# Use 80% for training, 20% for validation
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Training tokens: {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")


# ==============================================================================
# BATCH GENERATION
# ==============================================================================

def get_batch(split):
    """
    Generate a batch of input-target pairs for training or validation.
    
    For language modeling, given a sequence like "RECIPE: C", we want to predict:
    - Given "R", predict "E"
    - Given "RE", predict "C"
    - Given "REC", predict "I"
    - etc.
    
    Args:
        split: 'train' or 'val' to select which dataset to sample from
        
    Returns:
        x: Input sequences of shape (batch_size, block_size)
        y: Target sequences of shape (batch_size, block_size)
           y[i] is the next character for each position in x
    """
    data = train_data if split == "train" else val_data
    
    # Randomly sample starting positions for our batch
    # We need to leave room for block_size characters after each start
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Stack the sequences into batches
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    
    # Move data to the appropriate device (GPU/MPS/CPU)
    x, y = x.to(device), y.to(device)
    
    return x, y


@torch.no_grad()  # Disable gradient computation for efficiency during evaluation
def estimate_loss():
    """
    Estimate the average loss on train and validation sets.
    
    We average over multiple batches to get a more stable estimate,
    since individual batch losses can be noisy.
    
    Returns:
        Dictionary with 'train' and 'val' average losses
    """
    out = {}
    model.eval()  # Set model to evaluation mode (disables dropout)
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()  # Set model back to training mode
    return out


# ==============================================================================
# ATTENTION HEAD
# ==============================================================================

class Head(nn.Module):
    """
    Single Head of Self-Attention
    
    Self-attention allows each token to "look at" other tokens in the sequence
    and gather information from them. This is the core mechanism that allows
    transformers to understand context.
    
    How it works:
    1. Each token produces a Query (what am I looking for?)
    2. Each token produces a Key (what do I contain?)
    3. Each token produces a Value (what information do I have?)
    4. Attention scores = Query @ Key^T (how relevant is each token to me?)
    5. Output = softmax(scores) @ Value (weighted sum of values)
    
    The "causal" or "masked" attention means tokens can only attend to
    previous tokens (not future ones) - essential for text generation.
    """
    
    def __init__(self, head_size):
        """
        Args:
            head_size: Dimension of each attention head
                      Total embedding dim = n_head * head_size
        """
        super().__init__()
        
        # Linear projections for Query, Key, Value
        # bias=False is a common choice in attention mechanisms
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
        # Causal mask: prevents attending to future tokens
        # register_buffer means this tensor moves with the model but isn't a parameter
        # tril (lower triangular) allows each position to see itself and previous positions
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
        # Dropout for regularization - randomly zeros attention weights during training
        # This prevents the model from relying too heavily on specific attention patterns
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time/Sequence, Channels/Embedding)
            
        Returns:
            Output tensor of same shape with attention applied
        """
        B, T, C = x.shape
        
        # Compute queries, keys, values
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Compute attention scores ("affinities" between tokens)
        # We scale by 1/sqrt(head_size) to prevent softmax from becoming too peaky
        # Without scaling, dot products grow with dimension, making gradients tiny
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5  # (B, T, T)
        
        # Apply causal mask: set future positions to -inf so softmax gives 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax normalizes attention weights to sum to 1
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        
        # Apply dropout to attention weights (not to values)
        wei = self.dropout(wei)
        
        # Weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        
        return out


# ==============================================================================
# MULTI-HEAD ATTENTION
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention
    
    Instead of having one attention mechanism, we have multiple "heads"
    that each learn to attend to different aspects of the input.
    
    For cooking recipes, different heads might learn:
    - Head 1: Ingredient relationships (flour -> add butter)
    - Head 2: Action sequences (preheat -> bake -> cool)
    - Head 3: Measurement patterns (2 cups -> tablespoons)
    - Head 4: Recipe structure (title -> ingredients -> steps)
    
    The outputs from all heads are concatenated and linearly projected.
    
    Why multi-head is better than single head:
    - More expressive: can capture multiple types of relationships
    - Same compute cost: if we use n_head heads of size (n_embed/n_head),
      total compute is similar to one head of size n_embed
    """
    
    def __init__(self, num_heads, head_size):
        """
        Args:
            num_heads: Number of parallel attention heads
            head_size: Dimension of each attention head
        """
        super().__init__()
        
        # Create multiple attention heads as a ModuleList
        # ModuleList properly registers all heads as submodules
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # Output projection: combines all head outputs
        # Transforms concatenated heads back to embedding dimension
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        
        # Dropout after the projection for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape (B, T, C)
            
        Returns:
            Output of shape (B, T, C) after multi-head attention
        """
        # Run all heads in parallel and concatenate their outputs
        # Each head produces (B, T, head_size), we concatenate to (B, T, n_embed)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Project and apply dropout
        out = self.dropout(self.proj(out))
        
        return out


# ==============================================================================
# FEED-FORWARD NETWORK
# ==============================================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Applied to each position independently (no mixing between positions).
    This is where the model does "thinking" after gathering information
    via attention.
    
    Architecture: Linear -> ReLU -> Linear -> Dropout
    
    The expansion factor of 4x is from the original Transformer paper.
    This gives the network more capacity to learn complex transformations.
    
    Why FFN is important:
    - Attention only does weighted averaging (linear operation)
    - FFN adds non-linearity, enabling complex function approximation
    - Acts like a "memory" that stores learned patterns
    """
    
    def __init__(self, n_embed):
        """
        Args:
            n_embed: Embedding dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # Expand to 4x the embedding dimension
            # This gives more capacity for learning complex patterns
            nn.Linear(n_embed, 4 * n_embed),
            
            # ReLU activation introduces non-linearity
            # Without this, the whole network would just be linear
            nn.ReLU(),
            
            # Project back to embedding dimension
            nn.Linear(4 * n_embed, n_embed),
            
            # Dropout for regularization
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Simply applies the feed-forward network."""
        return self.net(x)


# ==============================================================================
# TRANSFORMER BLOCK
# ==============================================================================

class Block(nn.Module):
    """
    Transformer Block - The core building unit
    
    Combines multi-head attention and feed-forward network with:
    1. Pre-Layer Normalization (for training stability)
    2. Residual Connections (for gradient flow)
    
    Architecture:
        x -> LayerNorm -> Attention -> + residual
        x -> LayerNorm -> FFN -> + residual
    
    Pre-LayerNorm vs Post-LayerNorm:
    - Original Transformer used Post-LN: x + Attention(LayerNorm(x))
    - We use Pre-LN: x + Attention(LayerNorm(x))
    - Pre-LN is more stable for training deep networks
    
    Residual Connections:
    - Allow gradients to flow directly through the network
    - Enable training of very deep networks (without them, gradients vanish)
    - Let the network learn "refinements" rather than full transformations
    """
    
    def __init__(self, n_embed, n_head):
        """
        Args:
            n_embed: Embedding dimension
            n_head: Number of attention heads
        """
        super().__init__()
        
        # Calculate head size: divide embedding evenly among heads
        head_size = n_embed // n_head
        
        # Multi-head self-attention for information gathering
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # Feed-forward network for processing gathered information
        self.ffwd = FeedForward(n_embed)
        
        # Layer normalization layers (one for each sub-layer)
        # LayerNorm normalizes across the embedding dimension
        # This stabilizes training by keeping activations at consistent scale
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """
        Apply the transformer block with residual connections.
        
        Residual pattern: output = input + transformation(input)
        This means the layers learn "what to add" rather than "what to output"
        """
        # Attention block with residual connection
        # x = x + attention(layernorm(x))
        x = x + self.sa(self.ln1(x))
        
        # Feed-forward block with residual connection
        # x = x + ffn(layernorm(x))
        x = x + self.ffwd(self.ln2(x))
        
        return x


# ==============================================================================
# COOKGPT MODEL
# ==============================================================================

class CookGPT(nn.Module):
    """
    CookGPT - GPT-style Language Model for Cooking Recipes
    
    This is a decoder-only transformer that generates text one character at a time.
    Given a sequence of characters, it predicts the probability distribution
    over the next character.
    
    Architecture:
    1. Token Embedding: Convert character indices to vectors
    2. Position Embedding: Add positional information
    3. Transformer Blocks: Process with attention and FFN
    4. Final LayerNorm: Normalize before output
    5. Linear Head: Project to vocabulary size for next-char prediction
    
    Training Objective:
    - Cross-entropy loss between predicted and actual next characters
    - The model learns to assign high probability to the correct next char
    """
    
    def __init__(self):
        super().__init__()
        
        # Token embedding table: maps each character to a vector
        # Shape: (vocab_size, n_embed)
        # This is the model's "understanding" of each character
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        
        # Position embedding table: encodes position in sequence
        # Shape: (block_size, n_embed)
        # Without this, the model couldn't distinguish "CAT" from "TAC"
        # (attention is permutation-equivariant without position info)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Stack of transformer blocks
        # Each block refines the representation
        # More blocks = deeper understanding, but slower training
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head=n_head) for _ in range(n_layer)
        ])
        
        # Final layer normalization
        # Applied before the output projection for stability
        self.ln_f = nn.LayerNorm(n_embed)
        
        # Language model head: projects embeddings to vocabulary logits
        # For each position, produces a score for each possible next character
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        # Initialize weights with smaller values for better training
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization for better training dynamics.
        
        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - Biases: Zero
        
        These values are from the GPT-2 paper and work well empirically.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.
        
        Args:
            idx: Input token indices of shape (B, T)
            targets: Optional target token indices for computing loss
            
        Returns:
            logits: Prediction scores of shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        
        # Get token embeddings for input characters
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embed)
        
        # Get position embeddings for each position
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)  # (T,)
        )  # (T, n_embed)
        
        # Combine token and position information
        # Broadcasting: (B, T, n_embed) + (T, n_embed) = (B, T, n_embed)
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        x = self.blocks(x)  # (B, T, n_embed)
        
        # Final layer normalization
        x = self.ln_f(x)  # (B, T, n_embed)
        
        # Project to vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy: (B*T, vocab_size) and (B*T,)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            
            # Cross-entropy loss measures how well predictions match targets
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.
        
        Starting from a prompt (idx), repeatedly:
        1. Get predictions for the last position
        2. Sample from the probability distribution
        3. Append the sampled token
        4. Repeat until max_new_tokens reached
        
        Args:
            idx: Starting context of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            
        Returns:
            Extended sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to last block_size tokens
            # (model can't handle longer sequences due to position embeddings)
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, loss = self.forward(idx_cond)
            
            # Focus only on the last time step (next token prediction)
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample from the distribution
            # multinomial samples indices according to their probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx


# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================

print("\n" + "="*60)
print("INITIALIZING COOKGPT MODEL")
print("="*60)

# Create the model and move to device
model = CookGPT()
model = model.to(device)

# Count total parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")
print(f"Model architecture: {n_layer} layers, {n_head} heads, {n_embed} embed dim")

# ==============================================================================
# TRAINING
# ==============================================================================

print("\n" + "="*60)
print("TRAINING")
print("="*60)

# Create optimizer
# AdamW is Adam with proper weight decay (regularization)
# It's the standard optimizer for transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    
    # Periodically evaluate loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Get a batch of training data
    xb, yb = get_batch('train')
    
    # Forward pass: compute predictions and loss
    logits, loss = model(xb, yb)
    
    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)  # Clear old gradients
    loss.backward()                         # Compute new gradients
    optimizer.step()                        # Update parameters

print(f"\nFinal training loss: {loss.item():.4f}")

# ==============================================================================
# GENERATION
# ==============================================================================

print("\n" + "="*60)
print("GENERATING RECIPES")
print("="*60)

# Set model to evaluation mode (disables dropout)
model.eval()

# Start with a recipe-style prompt
prompt = "RECIPE:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

print(f"Generating 10,000 tokens starting with: '{prompt}'")
print("This may take a few minutes...")

# Generate tokens
with torch.no_grad():  # Disable gradients for faster generation
    generated = model.generate(context, max_new_tokens=10000)

# Decode to text
generated_text = decode(generated[0].tolist())

# Save to output file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print(f"\nGenerated text saved to output.txt")
print(f"Total characters generated: {len(generated_text):,}")

# Print a sample of the output
print("\n" + "="*60)
print("SAMPLE OUTPUT (first 1000 characters)")
print("="*60)
print(generated_text[:1000])
