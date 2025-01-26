import jax
import jax.numpy as jnp

def batch_iterate(data, batch_size):
    for i in range(0, len(data['X']), batch_size):
        yield {'X': data['X'][i:i+batch_size], 'y': data['y'][i:i+batch_size]}

# Placeholder for additional utility functions