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

class DR(object):

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
    
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
    
    def VARAUTOENC(self):
        return None

embeddings = np.array([[0.19731526, 0.12105352,-0.25813273,-0.267],
                       [0.19731526, 0.12105352,-0.25813273, -0.267]])
dr = DR(embeddings)
pca = dr.PCA()
print(pca)
