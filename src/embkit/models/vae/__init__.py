"""
This file initializes the Variational Autoencoder (VAE) models in the embkit library.
It includes two classes, BaseVAE and NetVae, which are used for different types of VAEs respectively.
"""
from .vae import VAE, BaseVAE
from .net_vae import NetVAE
from .rna_vae import RNAVAE
from .encoder import  Encoder
from .decoder import  Decoder
