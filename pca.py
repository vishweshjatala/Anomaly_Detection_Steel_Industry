from sklearn.decomposition import IncrementalPCA, PCA
import torch


class PCAModel:
    def __init__(self, variance_range, scaled_data, batch_size):
        base_model = PCA(n_components=variance_range)
        base_model.fit(scaled_data)
        n_components = base_model.n_components_
        self.n_components = n_components
        del base_model
        self.model = IncrementalPCA(n_components=self.n_components,batch_size=batch_size)
        self.fit_mode = True

    def eval(self):
        self.fit_mode = False

    def __call__(self, x):
        device = x.device
        x = x.squeeze(0).detach().cpu().numpy()
        print(x.shape)
        if self.fit_mode:
            self.model.partial_fit(x)
        encoded = self.model.transform(x)
        decoded = self.model.inverse_transform(encoded)

        x = torch.from_numpy(decoded).unsqueeze(0).to(device)
        return x
