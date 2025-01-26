from flax import linen as nn

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Embed(num_embeddings=10000, features=16)(x)  # Embedding layer
        x = x.reshape((x.shape[0], x.shape[1], -1))  # Reshape for convolution
        x = nn.Conv(features=32, kernel_size=(3,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return nn.sigmoid(x)