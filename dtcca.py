import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from tensorly.decomposition import parafac
import tensorly as tl
from tensorly.tenalg import multi_mode_dot
from sqrtm import sqrtm

tl.set_backend('pytorch')


class TCCA:
    def __init__(self, outdim_size, tcca_reg):
        self.outdim_size = outdim_size
        self.tcca_reg = tcca_reg
        self.mus = None
        self.projs = None

    def fit(self, Hs):
        num_views = len(Hs)
        n = list(Hs[0].size())[0]
        self.mus = []
        Hs_center = []
        C_tildes_inverse_half = []
        for v in range(num_views):
            d = list(Hs[v].size())[1]
            mu = torch.mean(Hs[v], dim=0)
            self.mus.append(mu)
            H_center = (Hs[v] - mu).t()  # d x n
            Hs_center.append(H_center)
            C_tilde = torch.mm(H_center, H_center.t()) / n + self.tcca_reg * torch.eye(d)
            C_tildes_inverse_half.append(torch.inverse(sqrtm(C_tilde)))

        # tensor decomposition
        # print('decomposition')
        Cor = tl.kruskal_to_tensor((torch.ones(n), Hs_center)) / n
        M = multi_mode_dot(Cor, C_tildes_inverse_half)
        weights, u = parafac(M, self.outdim_size, normalize_factors=True)

        # reconstruct projection matrix
        # print('projection')
        self.projs = []
        for v in range(num_views):
            proj = torch.mm(C_tildes_inverse_half[v], u[v])
            self.projs.append(proj)

    def test(self, Hs):
        num_views = len(Hs)
        embeddings = []
        for v in range(num_views):
            H_center = Hs[v] - self.mus[v]
            embedding = torch.mm(H_center, self.projs[v])
            embeddings.append(embedding.detach())
        return embeddings

    def loss(self, Hs):
        num_views = len(Hs)
        n = list(Hs[0].size())[0]
        Hs_center = []
        C_tildes_inverse_half = []
        for v in range(num_views):
            d = list(Hs[v].size())[1]
            mu = torch.mean(Hs[v], dim=0)
            H_center = (Hs[v] - mu).t()  # d x n
            Hs_center.append(H_center)
            C_tilde = torch.mm(H_center, H_center.t()) / n + self.tcca_reg * torch.eye(d)
            C_tildes_inverse_half.append(torch.inverse(sqrtm(C_tilde)))

        # tensor decomposition
        Cor = tl.kruskal_to_tensor((torch.ones(n), Hs_center)) / n
        M = multi_mode_dot(Cor, C_tildes_inverse_half)
        weights, u = parafac(M, self.outdim_size, normalize_factors=True)

        # compute loss
        M_hat = tl.kruskal_to_tensor((weights.detach(), [x.detach() for x in u]))
        loss = torch.norm(M - M_hat) ** 2
        return loss


class MLPNet(nn.Module):
    def __init__(self, layer_sizes):
        """ layer_sizes include input-latent-output sizes"""
        super(MLPNet, self).__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for lid in range(len(layer_sizes) - 1):
            if lid == len(layer_sizes) - 2:
                layers.append(nn.Linear(layer_sizes[lid], layer_sizes[lid+1]))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[lid], layer_sizes[lid+1]),
                    nn.Dropout(p=0.1),
                    nn.Sigmoid(), # Caltech101-7,
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DCCANet(nn.Module):
    def __init__(self, layer_sizes_list):
        super(DCCANet, self).__init__()
        num_views = len(layer_sizes_list)
        model_list = []
        for v in range(num_views):
            model = MLPNet(layer_sizes_list[v]).float()
            model_list.append(model)
        self.model_list = nn.ModuleList(model_list)

    def forward(self, X_list):
        outputs = []
        for v in range(len(X_list)):
            X = X_list[v]
            outputs.append(self.model_list[v](X))
        return outputs


class DTCCA:
    def __init__(self, layer_sizes_list, outdim_size, tcca_reg, epochs):
        self.model = DCCANet(layer_sizes_list).float()
        self.outdim_size = outdim_size
        self.tcca_reg = tcca_reg
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=0, lr=1e-3)
        self.tcca_model = None

    def fit(self, X_list, y=None, vX_list=None, vy=None, knn=1, apply_knn=True,
               checkpoint='checkpointv2_dtcca.model'):
        val_best_acc = None
        for epoch in range(self.epochs):
            self.model.train()
            Hs = self.model(X_list)
            tcca_loss = TCCA(self.outdim_size, self.tcca_reg).loss(Hs)
            self.optimizer.zero_grad()
            tcca_loss.backward()
            self.optimizer.step()

            if vX_list is not None:
                with torch.no_grad():
                    self.model.eval()
                    Hs = self.model(X_list)
                    tcca_model = TCCA(self.outdim_size, self.tcca_reg)
                    tcca_model.fit(Hs)
                    Xtrs_embeddings = tcca_model.test(Hs)
                    Xvals_embeddings = tcca_model.test(self.model(vX_list))
                    acc = evaluate_knn(Xtrs_embeddings, y, Xvals_embeddings, vy, knn=knn, apply_knn=apply_knn)
                    if val_best_acc is None or val_best_acc < acc:
                        val_best_acc = acc
                        print(f"epoch={epoch}, loss={tcca_loss.item()}, val_acc={val_best_acc}")
                        self.tcca_model = tcca_model
                        torch.save(self.model.state_dict(), checkpoint)
                    # else:
                    #     print(f"epoch={epoch}, loss={tcca_loss.item()}")

        # reset the model using the best one selected by validation set
        self.model.load_state_dict(torch.load(checkpoint))

    def test(self, X_list):
        with torch.no_grad():
            self.model.eval()
            Hs = self.model(X_list)
            embeddings = self.tcca_model.test(Hs)
            return embeddings


def evaluate_knn(Xtrs, ytr, Xtes, yte, knn=1, apply_knn=True, apply_concat=True):
    # concatenate features
    knn_classifier = KNeighborsClassifier(n_neighbors=knn)
    if apply_concat:
        Xtr = np.concatenate(Xtrs, axis=1)
        Xte = np.concatenate(Xtes, axis=1)
    else:
        num_view = len(Xtrs)
        Xtr = Xtrs[0]
        Xte = Xtes[0]
        for i in range(1,num_view):
            Xtr += Xtrs[i]
            Xte += Xtes[i]
        Xtr = Xtr / num_view
        Xte = Xte / num_view

    if apply_knn:
        knn_classifier.fit(Xtr, ytr)
        ypred = knn_classifier.predict(Xte)
    else:
        clf = svm.LinearSVC(C=knn, dual=False)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

    acc = 1 - np.count_nonzero(ypred - yte)/yte.shape[0]
    return acc


def run_dcca_inductive(dcca_cls, Xs, ys, tr_idxs, te_idxs, outdim_size, layer_sizes, cca_reg,
             epoches=20, knn=1, apply_knn=True, apply_pca=True):
    num_views = len(Xs)
    splits = tr_idxs.shape[0]
    accs = []
    for i in range(splits):
        # split training/testing data
        tr_idx = tr_idxs[i, :].tolist()
        te_idx = te_idxs[i, :].tolist()
        ytr = ys[tr_idx]
        yte = ys[te_idx]
        Xtrs = []
        Xtes = []
        for v in range(num_views):
            Xtr_v = Xs[v][tr_idx, :]
            Xte_v = Xs[v][te_idx, :]

            if apply_pca:
                n_v = Xtr_v.shape[1]
                n_comp = min(n_v, outdim_size)
                if dcca_cls == TCCA:
                    pca = PCA(n_components=n_comp)
                    pca.fit(Xtr_v)
                else:
                    pca = PCA(n_components=0.95)
                    pca.fit(Xtr_v)
                    if pca.n_components < n_comp:
                        pca = PCA(n_components=n_comp)
                        pca.fit(Xtr_v)

                Xtr_v = pca.transform(Xtr_v)
                Xte_v = pca.transform(Xte_v)

                layer_sizes[v][0] = Xtr_v.shape[1]

            Xtrs.append(Xtr_v)
            Xtes.append(Xte_v)

        torch.manual_seed(1)
        Xs_list = [torch.tensor(X.astype('float32')) for X in Xtrs]
        vXs_list = [torch.tensor(X.astype('float32')) for X in Xtes]
        dcca = dcca_cls(layer_sizes, outdim_size, cca_reg, epoches)
        dcca.fit(Xs_list, y=ytr, vX_list=vXs_list, vy=yte, knn=knn, apply_knn=apply_knn)

        Xtrs_embeddings = dcca.test(Xs_list)
        Xtes_embeddings = dcca.test(vXs_list)

        # classification on the embedded data
        acc = evaluate_knn(Xtrs_embeddings, ytr, Xtes_embeddings, yte, knn=knn, apply_knn=apply_knn)

        print(f"{dcca_cls.__name__},split={i}, acc={acc}")
        accs.append(acc)
    return accs


if __name__ == "__main__":
    from scipy.io import loadmat
    from sklearn import preprocessing
    torch.manual_seed(1)

    data = loadmat('data/mfeat.mat')
    num_views = data["X"].shape[0]

    tr_idxs = data["tr_idxs"][0, 3] - 1  # index start from zero
    te_idxs = data["te_idxs"][0, 3] - 1

    Xs = []
    for i in range(num_views):
        tmpX = preprocessing.scale(data["X"][i, 0])
        Xs.append(tmpX)
    ys = data["y"].reshape(data["y"].shape[0])

    cca_reg = 1e-8
    epoches = 20
    outdim_size = 5
    latent_sizes = [256, 1024, 5]
    layer_sizes_list = []
    for v in range(num_views):
        layer_sizes = [Xs[v].shape[1]] + latent_sizes
        layer_sizes_list.append(layer_sizes)
    print(layer_sizes_list)

    Xs_list = [torch.tensor(X.astype('float32')) for X in Xs]
    accs = run_dcca_inductive(DTCCA, Xs, ys, tr_idxs, te_idxs, outdim_size, layer_sizes_list, cca_reg,
                              epoches=epoches, knn=1, apply_knn=False, apply_pca=False)

    print(accs)
    print(sum(accs)/len(accs))