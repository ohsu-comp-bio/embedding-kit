import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import time

class VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder_fc1 = nn.Linear(feature_dim, latent_dim)
        self.encoder_bn1 = nn.BatchNorm1d(latent_dim)
        self.encoder_relu = nn.ReLU()
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, feature_dim)
        self.decoder_sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.encoder_relu(self.encoder_bn1(self.encoder_fc1(x)))
        return self.fc_mean(h1), self.fc_log_var(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_sigmoid(self.decoder_fc1(z))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.encoder_fc1.in_features))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[1]), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta * KLD

def train_vae(model, train_loader, optimizer, epoch, beta=1.0):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def synth_latent_3(latent_object, synth_index_name):
    print('Start synth sample gen from latent')
    synth_in_count = 3
    synth_sub_len = 200
    synth_ndx_strt = 0
    synth_full_frame = pd.DataFrame(columns = latent_object.columns)
    for subtype in sorted(latent_object.Labels.unique()):
        print(subtype)
        sub = latent_object[latent_object.Labels == subtype]
        print(synth_sub_len)
        synth_index = ['SYNTH-' + synth_index_name + '-' + jtem for jtem in [str(
            item).zfill(5) for item in list(range(synth_ndx_strt,
                                                  synth_sub_len + synth_ndx_strt))]]
        synth_sub_frame = pd.DataFrame(index = synth_index)
        synth_sub_frame.insert(0, 'Labels', sub.Labels[0])
        synth_dict = {}
        for synth_sample in synth_sub_frame.index:
            input_sample_set = sub.sample(synth_in_count)
            new_samp_vec = []
            for col in input_sample_set.iloc[:, 1:]:
                vals_inpt = input_sample_set.loc[:, col]
                choosen_val = vals_inpt.sample(1)
                new_samp_vec.append(choosen_val.values[0])
            synth_dict[synth_sample] = new_samp_vec
        synth_sub_frame = pd.concat([synth_sub_frame, pd.DataFrame(synth_dict).T], axis = 1)
        synth_full_frame = pd.concat(
            [synth_full_frame, synth_sub_frame], axis = 0)
        synth_ndx_strt = synth_ndx_strt + synth_sub_len
    print('Synthetic from latent done, '+str(synth_sub_len)+' samples generated for each subtype')
    return(synth_full_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--feature_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=250)
    parser.add_argument('--beta_start', type=float, default=0.0)
    parser.add_argument('--beta_end', type=float, default=1.0)
    parser.add_argument('--beta_epochs', type=int, default=50)
    args = parser.parse_args()

    # Create output directory
    output_dir = f'i_o/{args.version}/{os.path.basename(args.feature_file).split(".")[0]}/'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data_df = pd.read_csv(args.feature_file, sep='\t', index_col=0)
    labels = data_df.iloc[:, 0]
    features = data_df.iloc[:, 1:]
    feature_dim = features.shape[1]

    # Normalize features (MinMax scaling to [0, 1] as expected by BCE loss)
    features_normalized = (features - features.min()) / (features.max() - features.min())
    
    # Create DataLoader
    tensor_data = torch.tensor(features_normalized.values, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model and optimizer
    model = VAE(feature_dim=feature_dim, latent_dim=args.latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop with beta annealing
    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta_end, args.beta_start + (args.beta_end - args.beta_start) * (epoch / args.beta_epochs))
        train_vae(model, train_loader, optimizer, epoch, beta)

    # Generate synthetic data
    model.eval()
    with torch.no_grad():
        # Encode the original data to get latent representations
        mu, _ = model.encode(tensor_data)
        latent_object = pd.DataFrame(mu.numpy(), index=data_df.index)
        latent_object.insert(0, 'Labels', labels)
        
        # Generate synthetic latent vectors
        synth_latent_frame = synth_latent_3(latent_object, os.path.basename(args.feature_file).split(".")[0])
        
        # Decode synthetic latent vectors
        synth_latent_tensor = torch.tensor(synth_latent_frame.iloc[:, 1:].values, dtype=torch.float32)
        decoded_synth = model.decode(synth_latent_tensor)
        
        # Create final synthetic data DataFrame
        synth_df = pd.DataFrame(decoded_synth.numpy(), index=synth_latent_frame.index, columns=features.columns)
        synth_df.insert(0, 'Labels', synth_latent_frame['Labels'])

        # Save synthetic data
        output_file = os.path.join(output_dir, f'synthetic_data_{args.version}.tsv')
        synth_df.to_csv(output_file, sep='\t')
        print(f"Synthetic data saved to {output_file}")
