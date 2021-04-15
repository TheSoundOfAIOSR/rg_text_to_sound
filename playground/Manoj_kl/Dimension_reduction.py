from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn 

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo

class DR(object):

    def __init__(self, embeddings, data):
        super().__init__()
        self.embeddings = embeddings
        self.data = data
    
    def PCA(self,n_components=None):

        n_samples = len(self.embeddings)
        n_features = len(self.embeddings[0])
        n_components_check = min(n_samples, n_features)
        feature_flag = [len(feature) for feature in self.embeddings]

        if n_components == None:
            n_components = min(n_samples, n_features) - 1

        if (n_components <= n_components_check) and (len(set(feature_flag))==1): 
            pca = PCA(n_components=n_components)
            x = pca.fit_transform(self.embeddings)
            return x

        elif not (len(set(feature_flag))==1):
            print("Length of the all the samples needs to be same")
        
        else:
            print("Number of components need to be less than or equal to minimum of number of samples and number of features")
            return None

    def AUTOENC(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        model = AE(input_shape=784).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        return None
    
    def VARAUTOENC(self, data):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = self.data
        vae = VAE()
        trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
        trainer.fit(vae, data)

        return None

embeddings = np.array([[0.19731526, 0.12105352,-0.25813273,-0.267],
                       [0.19731526, 0.12105352,-0.25813273, -0.267]])
dr = DR(embeddings)
pca = dr.PCA()
print(pca)
