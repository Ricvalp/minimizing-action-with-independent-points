import jax
import jax.numpy as jnp
import flax
# import diffrax
from typing import Sequence
import flax.linen as nn


class encoder(nn.Module):

    num_hidden: Sequence[int]

    def setup(self):

        layer1 = nn.Dense(features=self.num_hidden[0], kernel_init=nn.initializers.glorot_normal())
        layer2 = nn.Dense(features=self.num_hidden[1], kernel_init=nn.initializers.glorot_normal())
        layer3 = nn.Dense(features=self.num_hidden[2], kernel_init=nn.initializers.glorot_normal())
    
    def __call__(self, x):

        x = self.layer1(x)
        x = nn.selu(x)
        x = self.leyer2(x)
        x = nn.selu(x)
        x = self.layer3(x)

        return x

class decoder(nn.Module):

    num_hidden: Sequence[int]

    def setup(self):

        self.layer1 = nn.Dense(features=self.num_hidden[0], kernel_init=nn.initializers.glorot_normal())
        self.layer2 = nn.Dense(features=self.num_hidden[1], kernel_init=nn.initializers.glorot_normal())
        self.layer3 = nn.Dense(features=self.num_hidden[2], kernel_init=nn.initializers.glorot_normal())
    
    def __call__(self, x):

        x = self.layer1(x)
        x = nn.tanh(x)
        x = self.layer2(x)
        x = nn.tanh(x)
        x = self.layer3(x)

        return x
