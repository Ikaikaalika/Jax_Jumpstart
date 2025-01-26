import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from model import CNN

# Data Processing
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Create vocab from training data
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1 if x == 'pos' else 0

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(torch.float32), text_list, offsets

# DataLoader
from torch.utils.data import DataLoader
train_iter, test_iter = IMDB(split='train'), IMDB(split='test')
train_dataloader = DataLoader(train_iter, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=64, collate_fn=collate_batch)

# Model and Training Setup
model = CNN(num_embeddings=len(vocab))
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 200))  # Assuming a max sequence length of 200 for example
variables = model.init(rng, dummy_input)

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(variables['params'])

@jax.jit
def compute_loss(params, x, y):
    logits = model.apply(params, x)
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()

@jax.jit
def update(params, opt_state, x, y):
    grads = jax.grad(compute_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Training loop
for epoch in range(10):
    for batch in train_dataloader:
        labels, texts, offsets = batch
        texts_jax = jnp.array(texts.numpy())
        labels_jax = jnp.array(labels.numpy())
        variables['params'], opt_state = update(variables['params'], opt_state, texts_jax, labels_jax)
    
    print(f"Epoch {epoch} completed")

# Evaluation
correct = 0
total = 0
for batch in test_dataloader:
    labels, texts, offsets = batch
    texts_jax = jnp.array(texts.numpy())
    labels_jax = jnp.array(labels.numpy())
    predictions = model.apply(variables['params'], texts_jax)
    binary_predictions = (predictions > 0.5).astype(jnp.int32)
    correct += jnp.sum(jnp.equal(binary_predictions, labels_jax))
    total += labels_jax.size

accuracy = correct / total
print(f"Test Accuracy: {accuracy}")