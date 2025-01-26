import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from model import CNN
from data_preprocessing import preprocess_data, load_and_prepare_data
import matplotlib.pyplot as plt
import logging


# Check device(s)
print(jax.devices())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and prepare data
try:
    train_data, test_data = load_and_prepare_data()
    logger.info("Data loaded and preprocessed successfully.")
except Exception as e:
    logger.error(f"Error loading or preprocessing data: {e}")
    raise

# Initialize model
model = CNN()
rng = jax.random.PRNGKey(0)
x = jnp.ones((1, 200))  # Dummy input based on max_length from preprocessing
variables = model.init(rng, x)

# Setup training
learning_rate = 0.001
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(variables['params'])

@jax.jit
def compute_loss(params, x, y):
    logits = model.apply(params, x)
    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss

@jax.jit
def update(params, opt_state, x, y):
    grads = jax.grad(compute_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Lists to store losses for plotting
train_losses = []

# Training loop
for epoch in range(10):
    epoch_loss = 0.0
    for i in range(0, len(train_data['X']), 64):  # Batch size of 64
        batch = train_data['X'][i:i+64]
        targets = jnp.array(train_data['y'][i:i+64])
        variables['params'], opt_state = update(variables['params'], opt_state, batch, targets)
        epoch_loss += compute_loss(variables['params'], batch, targets).item()
    avg_loss = epoch_loss / (len(train_data['X']) // 64)
    train_losses.append(avg_loss)
    logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

# Evaluation
try:
    predictions = model.apply(variables['params'], test_data['X'])
    binary_predictions = (predictions > 0.5).astype(int)
    accuracy = jnp.mean(binary_predictions.flatten() == jnp.array(test_data['y']))
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    logger.error(f"Error during evaluation or plotting: {e}")
