from sklearn.decomposition import PCA
import numpy as np

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
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # model = AE(input_shape=784).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        return None
    
    def VARAUTOENC(self, data):

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # data = self.data
        # vae = VAE()
        # trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
        # trainer.fit(vae, data)

        return None