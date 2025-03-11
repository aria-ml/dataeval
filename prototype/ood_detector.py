from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FT

from scipy.linalg import null_space as null
from scipy.stats import binom, foldnorm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from vae_models import VAEOutput, normdot, vae_loss
from PIL import Image

from dataeval.utils._array import as_numpy


def intrinsic_dimension(X, k=None):
    # in X, each row is an example, each column is a feature (i.e. a pixel if X contains images)
    # Create a NearestNeighbors object and fit it to the data
    k = 5 if k is None else k  # Number of neighbors to find
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(as_numpy(X))

    # Find the k nearest neighbors of each point
    distances, indices = nbrs.kneighbors(as_numpy(X))

    distances = np.sort(distances, axis=1)
    Tk = distances[:, k - 1 : k]  # kth neighbor distance, preserve trailing dim of 1
    Tj = distances[:, 1:-1]  # all but last

    mi = 1.0 / np.mean(np.log(Tk / Tj), axis=1)  # average over each point's neighbors
    mbar = 1.0 / (np.mean(1.0 / mi))  # average over all points

    return mbar

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None):
        if type(data) is torch.Tensor:
            self.data = data.float()
        else:
            self.data = torch.from_numpy(data).float()

        self.labels = None
        if labels is not None:
            if type(labels) is torch.Tensor:
                self.labels = labels.long()
            else:
                self.labels = (
                    torch.from_numpy(labels).long() if labels is not None else None
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]  # , [None] * len(self.data[idx]) # how to handle spots for labels and metadata?


def batch_apply(  # very preliminary. Not suitable for anything that builds a tree. 
    loader: torch.utils.data.DataLoader,
    func: Callable,
    nbatch: int | None = None,
    do_all: bool = False,
    device=None,
    reduction: Callable | None = None
):
    if do_all:
        if nbatch is None:
            nbatch = loader.__len__()
        else:
            raise ValueError("Cannot set do_all to True and also specify number of batches.")

    nbatch = 2 if nbatch is None else nbatch

    device = (
        torch.device("cuda")
        if (device is None and torch.cuda.is_available())
        else torch.device("cpu")
    )

    results, data = torch.zeros(1), torch.zeros(1)
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        this_result = func(data)
        if batch_idx == 0:
            results = torch.zeros(
                (nbatch, *this_result.shape), dtype=this_result.dtype, device=device
            )
        results[batch_idx] = this_result

    if reduction == "mean":  
        results = results.mean()
    elif reduction == "batchmean":  
        results = results.sum() / data.size(0)
    elif reduction == "sum":
        results = results.sum()
    
    return results


class OODdetector:
    def __init__(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor
        | torch.utils.data.Dataset
        | torch.utils.data.DataLoader,
        test_data: torch.Tensor
        | torch.utils.data.Dataset
        | torch.utils.data.DataLoader,
        recon_error: Callable | None = None,
        verbose: bool | None = None,
        criterion=None,
        beta: float | None = None,
        batch_size: int | None = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = loss_function if criterion is None else criterion

        self.beta = 1.0 if beta is None else beta
        self.batch_size = 64 if batch_size is None else batch_size

        self.model.eval()

        self.training_example, self.training_dataset, self.train_loader = (
            self.unpack_data(training_data)
        )

        self.test_example, self.test_dataset, self.test_loader = self.unpack_data(
            test_data
        )

        tdata = torch.tensor(0)
        for tdata in self.test_loader:
            if len(tdata) > 1:
                tdata = tdata[0]
            break
        self.test_pred = self.model(tdata.to(torch.float).to(self.device))
        if isinstance(self.test_pred, tuple):
            self.test_pred = self.test_pred[0]
        elif isinstance(self.test_pred, VAEOutput):
            self.test_pred = self.test_pred.x_recon
        else:
            print("I do not recognize the prediction type of your model.")
            return None
        
        self.previous_test_loss = np.nan
        self.train_loss = np.nan

        self.recon_error = normdot if recon_error is None else recon_error
        try:
            assert self.recon_error == normdot
        except AssertionError:
            print(f"you set {self.recon_error} but only normdot works so far")

        self.verbose = True if verbose is None else verbose

    # If user gives a --
    #    Tensor: wrap it in a dataset and make a loader.
    #    Dataset: extract a tensor with __getitem__, and then make a loader
    #    DataLoader: Call the loader to get a tensor, and then extract the dataset attribute.
    #
    # Finally, return the example, the dataset, and the loader.
    #
    def unpack_data(
        self,
        data: torch.Tensor | torch.utils.data.Dataset | torch.utils.data.DataLoader,
    ):
        if isinstance(data, torch.Tensor):
            example = data
            dataset = CustomDataset(data)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        elif isinstance(data, torch.utils.data.Dataset):
            dataset = data
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
            example = next(iter(loader))
        elif isinstance(data, torch.utils.data.DataLoader):
            loader = data
            dataset = loader.dataset
            example = next(iter(loader))
        else:
            raise TypeError(
                f"You must provide either a Tensor, a Dataset, or a DataLoader, not {type(data)}."
            )
        
        if isinstance(example, (tuple, list)):
            example = example[0]

        return torch.as_tensor(example).to(torch.float).to(self.device), dataset, loader

    # Training data are stored in each OODdetector instance. When these are projected into
    #   the latent space, what is the dimension of the implied manifold?
    def manifold_dimension(self):
        tdata = self.training_example
        vdata = self.test_example
        etdata, _ = self.model.embed(tdata)
        tdim = intrinsic_dimension(etdata.reshape((self.training_example.shape[0], -1)))

        evdata, _ = self.model.embed(vdata)
        vdim = intrinsic_dimension(evdata.reshape((self.test_example.shape[0], -1)))

        return (tdim, vdim)

    def empirical_pvalue(self, test, null):
        table = np.sort(as_numpy(null).reshape((-1)))
        pvals = 1.0 - np.interp(as_numpy(test), table, np.linspace(0, 1, len(table)))
        return pvals

    # The input model will have an embed() method, but we still need to stack it and turn off gradient and go
    #   numpy. So here's the method for that.
    def get_latent_space_vectors(self, x):
        if len(x.shape) < 4:
            x = x.reshape(1, *x.shape)
        xlat, dxlat = self.model.embed(torch.as_tensor(x).to(self.device))
        xlat, dxlat = (
            xlat.detach().cpu().numpy().reshape((x.shape[0], -1)),
            dxlat.detach().cpu().numpy().reshape((x.shape[0], -1)),
        )
        return xlat, dxlat

    # see Out-of-Distribution Detection with Deep Nearest Neighbors by Y. Sun, Y. Ming, X. Zhu, and Y. Li, 2022. They try to work
    #  with distance from manifold near neighbors, but find it works best if all vectors are normalized i.e. directional comparisons.
    #  But they still use Euclidean distance between those unit vectors, oddly.
    #
    def manifold_distance_normalized(
        self, Xinput
    ):  # Xinput is incoming input data to be tested for OOD, embed it as Xtest
        Xtest, _ = self.get_latent_space_vectors(Xinput)

        # X_on_manifold, _  = self.get_latent_space_vectors(self.training_data) # including test data did not improve things noticeably
        X_on_manifold, _ = self.get_latent_space_vectors(
            torch.concatenate((self.training_example, self.test_example))
        )
        X_on_manifold = X_on_manifold / np.sqrt(
            np.sum(X_on_manifold**2, axis=1, keepdims=True)
        )

        # distances, indices = self.neighbors.kneighbors(Xtest)
        distances, indices = self.normal_neighbors.kneighbors(
            Xtest
        )  # did I forget this step?
        Xnn = X_on_manifold[
            indices
        ]  # k near neighbors of each test point, from embedded train data

        Xtest = Xtest / np.sqrt(np.sum(Xtest**2, axis=1, keepdims=True))

        off_manifold_dists = np.zeros(len(Xinput))
        for itest, xnn in enumerate(Xnn):
            off_manifold_displacement = Xtest[itest, :] - xnn[0]
            off_manifold_dists[itest] = np.sqrt(np.sum(off_manifold_displacement**2))

        return off_manifold_dists

    def one_nn_ecdf(
        self, Xinput, normalized=None
    ):  # Xinput is incoming input data to be tested for OOD, embed it as Xtest
        normalized = False if normalized is None else normalized

        X_on_manifold, _ = self.get_latent_space_vectors(
            self.training_example
        )  # including test data did not improve things noticeably
        Xtest, _ = self.get_latent_space_vectors(Xinput)

        if normalized:
            Xtest_norm = np.sqrt(np.sum(Xtest**2, axis=1, keepdims=True))
            Xtest /= Xtest_norm
            X_on_manifold_norm = np.sqrt(
                np.sum(X_on_manifold**2, axis=1, keepdims=True)
            )
            X_on_manifold /= X_on_manifold_norm

            dists, _ = self.normal_neighbors.kneighbors(Xtest, n_neighbors=1)
            refdists, _ = self.normal_neighbors.kneighbors(X_on_manifold, n_neighbors=2)
        else:
            dists, _ = self.neighbors.kneighbors(Xtest, n_neighbors=1)
            refdists, _ = self.neighbors.kneighbors(X_on_manifold, n_neighbors=2)

        refdists = np.sort(refdists[:, 1])
        dists = dists[:, 0]

        return self.empirical_pvalue(dists, refdists)

    def one_nn_distance_pval(
        self, Xinput
    ):  # Xinput is incoming input data to be tested for OOD, embed it as Xtest
        Xtest, scales = self.get_latent_space_vectors(Xinput)

        X_on_manifold, _ = self.get_latent_space_vectors(
            self.training_example
        )  

        _, indices = self.neighbors.kneighbors(Xtest)

        Xnn1 = X_on_manifold[
            indices[:, 0]
        ]  # k near neighbors of each test point, from embedded train data
        dX = Xtest - Xnn1

        pvals_all = 1.0 - foldnorm.cdf(np.sqrt(2) * np.abs(dX) / scales, 0.0)
        not_zero = pvals_all > 0.0
        non_zero_counts = np.sum(not_zero, axis=1)
        pvals_all[np.logical_not(not_zero)] = (
            1.0  # avoid log(zero), then exclude from average with non_zero_counts
        )
        
        pvals_min = np.min(pvals_all, axis=1)

        return pvals_min

    def manifold_distance(
        self, Xinput
    ):  
        # Xinput is incoming input data to be tested for OOD, embed it as Xtest
        Xtest, _ = self.get_latent_space_vectors(Xinput)

        X_on_manifold, _ = self.get_latent_space_vectors(
            self.training_example
        )

        _, indices = self.get_dists_and_indices(Xtest)
        Xnn = X_on_manifold[
            indices
        ] 

        X0 = np.mean(Xnn, axis=1, keepdims=True)
        dXtest = Xtest - X0.squeeze()
        dXnn = Xnn - X0

        off_manifold_dists = np.zeros(len(Xinput))
        for itest, dxnn in enumerate(dXnn):
            off_manifold_dir = null(dxnn)
            off_manifold_disp = dXtest[itest, :].dot(off_manifold_dir)
            off_manifold_dists[itest] = np.sqrt(np.sum(off_manifold_disp**2))

        return off_manifold_dists

    def manifold_distance_weighted(
        self, Xinput
    ):  # Xinput is incoming input data to be tested for OOD, embed it as Xtest
        Xtest, Xtest_std = self.get_latent_space_vectors(Xinput)

        X_on_manifold, X_on_manifold_std = self.get_latent_space_vectors(
            self.training_example
        )

        # distances, indices = self.neighbors.kneighbors(Xtest)
        _, indices = self.get_dists_and_indices(Xtest)
        Xnn = X_on_manifold[
            indices
        ]  # k near neighbors of each test point, from embedded train data
        Xnn_std = X_on_manifold_std[indices]
        dXtest = Xtest - Xnn[:, 0]
        dXnn = Xnn[:, 1:] - np.expand_dims(Xnn[:, 0], 1)
        Xnn_std = Xnn_std[:, 1:]

        dXtest /= Xtest_std
        dXnn /= Xnn_std

        off_manifold_dists = np.zeros(len(Xinput))
        for itest, dxnn in enumerate(dXnn):
            off_manifold_dir = null(dxnn)
            off_manifold_disp = dXtest[itest, :].dot(off_manifold_dir)
            off_manifold_dists[itest] = np.sqrt(np.sum(off_manifold_disp**2))

        return off_manifold_dists

    def get_dists_and_indices(self, X):
        etdata, _ = self.model.embed(self.training_example)
        data4knn = etdata.detach().cpu().reshape((self.training_example.shape[0], -1))
        mdim = int(np.max(np.round(self.manifold_dimension())))
        neighbors = NearestNeighbors(n_neighbors=mdim, algorithm="ball_tree").fit(
            data4knn
        )
        if len(X.shape) == 1 and X.shape[0] == self.model.latent_dim:
            X = X.reshape((1, -1))

        dists, indices = neighbors.kneighbors(X)

        return dists, indices

    def manifold_distance_full_svd(self, Xinput):

        Xtest, _ = self.get_latent_space_vectors(Xinput)
        X_on_manifold, _ = self.get_latent_space_vectors(
            torch.concatenate((self.training_example, self.test_example))
        )

        _, indices = self.get_dists_and_indices(Xtest)

        Xnn = X_on_manifold[
            indices
        ]  # k near neighbors of each test point, from embedded train data

        X0 = np.mean(Xnn, axis=1, keepdims=True)
        dXtest = Xtest - X0.squeeze()
        dXnn = Xnn - X0

        # u, s, vh = np.linalg.svd(dXnn[0]), say.
        # last m rows of vh @ off_manifold_dir make an identity matrix, with m being the difference between the
        #    latent space dimension and the intrinsic dimension.
        off_manifold_dists = np.zeros(len(Xinput))
        for itest, dxnn in enumerate(dXnn):
            u, s, vh = np.linalg.svd(dxnn)

            off_manifold_dir = vh[-1, :].T
            off_manifold_dists[itest] = np.abs(dXtest[itest, :].dot(off_manifold_dir))

        return off_manifold_dists

    def p_ID(self, Xinput, k=None):
        # p-values for the null hypotheses that each sample lies in-distribution.
        k = 3 if k is None else k
        N = self.model.latent_dim
        X_on_manifold, _ = self.get_latent_space_vectors(self.training_example)
        Xtest, _ = self.get_latent_space_vectors(Xinput)

        _, indices = self.neighbors.kneighbors(Xtest, n_neighbors=k)

        pvals = np.zeros(Xinput.shape[0])
        n_out = np.zeros_like(pvals)

        # k near neighbors of each test point, from embedded train data, shape (n_examples, k, latent_dim)
        Xnn = X_on_manifold[indices]

        is_min = Xtest < np.min(Xnn, axis=1)
        is_max = Xtest > np.max(Xnn, axis=1)

        nleft = np.sum(is_min, axis=1)
        nright = np.sum(is_max, axis=1)
        n_out = nleft + nright
        p_out = 2 / (k + 1)

        pvals = 1 - binom.cdf(n_out, N, p_out)
        return pvals

    def p_spread(self, X, k=None):
        # p-values for the null hypotheses that each sample lies in-distribution. I want to look at the
        #  1D ranges generated by each point and its two nearest neighbors. The training data will provide
        #  an empirical distribution for this. Then we can assess the test point and get a p-value from that distribution.
        k = 3 if k is None else k
        N = self.model.latent_dim

        X_on_manifold, _ = self.get_latent_space_vectors(self.training_example)
        _, tindices = self.neighbors.kneighbors(X_on_manifold)
        Xnn = X_on_manifold[tindices[:, :k]]  # train point and 2 neighbors.

        # mspreads is a spread for each latent space component on manifold, for each training example.
        mspreads = np.max(Xnn, axis=1) - np.min(Xnn, axis=1)

        Xtest, _ = self.get_latent_space_vectors(X)

        # _, indices = self.get_dists_and_indices(Xtest)
        _, indices = self.neighbors.kneighbors(Xtest, n_neighbors=k)
        Xcheck = np.concatenate(
            (X_on_manifold[indices[:, : (k - 1)]], np.expand_dims(Xtest, 1)), axis=1
        )

        # tspreads is a spread for each latent space component, for each point being queried for OOD.
        tspreads = np.max(Xcheck, axis=1) - np.min(Xcheck, axis=1)

        N = tspreads.shape[1]
        pvals = np.zeros_like(tspreads)
        for i in range(N):
            pvals[:, i] = self.empirical_pvalue(tspreads[:, i], mspreads[:, i])

        return np.min(pvals, axis=1)

    def p_xy(self, X, k=None, normalize=None):
        k = 3 if k is None else k

        X_on_manifold, _ = self.get_latent_space_vectors(self.training_example)
        tdists, tindices = self.neighbors.kneighbors(X_on_manifold, n_neighbors=k)
        Xnn = X_on_manifold[tindices]  # train point and 2 neighbors.
        NN0, NN1, NN2 = Xnn[:, k - 3], Xnn[:, k - 2], Xnn[:, k - 1]
        X0 = (NN1 + NN2) / 2.0

        Nhat = NN2 - NN1
        Nhat /= np.sqrt(
            np.sum(Nhat**2, axis=1, keepdims=True)
        )  # unit vector from NN1 to NN2
        DX = NN0 - X0
        DXnormsq = np.sum(DX**2, axis=1)
        ye = np.abs(np.sum(Nhat * DX, axis=1))
        xe = np.sqrt(DXnormsq - ye**2)

        if normalize:
            denom = np.sqrt(np.sum((NN1 - NN2) ** 2, axis=1))
            xe /= denom
            ye /= denom

        Xtest, _ = self.get_latent_space_vectors(X)
        _, indices = self.neighbors.kneighbors(Xtest, n_neighbors=k)

        nn0, nn1, nn2 = (
            Xtest,
            X_on_manifold[indices[:, k - 3]],
            X_on_manifold[indices[:, k - 2]],
        )
        x0 = (nn1 + nn2) / 2.0

        nhat = nn1 - x0
        nhat /= np.sqrt(np.sum(nhat**2, axis=1, keepdims=True))
        dx = nn0 - x0
        dxnormsq = np.sum(dx**2, axis=1)
        yc = np.abs(np.sum(nhat * dx, axis=1))
        xc = np.sqrt(np.clip(dxnormsq - yc**2, 0.0, torch.inf))

        if normalize:
            denom = np.sqrt(np.clip(np.sum((nn1 - nn2) ** 2, axis=1), 0.0, np.inf))
            xc /= denom
            yc /= denom

        px = self.empirical_pvalue(xc, xe)
        py = self.empirical_pvalue(yc, ye)

        return px * py

    def p_dist_normalized(self, X):
        mdist = self.manifold_distance_normalized(self.test_example)
        dist = self.manifold_distance_normalized(X)

        return self.empirical_pvalue(dist, mdist)

    # Make a simplex of nearest neighbors (among training examples) to each test point.
    # Compare "off-manifold" distance of each test point to what is typical in test data.
    def simplex_detect(self, xcheck):
        # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
        # distances, indices = nbrs.kneighbors(X)

        # is_ood =

        # return is_ood
        pass

    def ood_solid_angle(self):
        # draw unit vectors from test point to kNN
        #
        # find volume of convex hull of those vectors
        #
        # compare to known hull volume of unit hypersphere.
        #
        # If test point is ID, these will be close. If kNN hull is
        #   relatively small, it means the test point is viewing
        #   the mainfold from outside.
        pass

    def detect_OOD(self, xcheck):  # still using recon error
        recon = self.recon_error(self.test_example, self.test_pred)  # type: ignore

        xcheck_pred, _, _ = self.model(torch.as_tensor(xcheck).to(torch.float).to(self.device))  # type: ignore
        check_recon = self.recon_error(xcheck_pred, xcheck)  # type: ignore

        min_nd = np.percentile(recon, 1)
        is_ood = check_recon < min_nd

        return is_ood

    def set_learning_rate(self, new_rate):
        for g in self.optimizer.param_groups:
            g["lr"] = new_rate

    def show(self, xcheck):
        recon = self.recon_error(self.test_example, self.test_pred)
        xcheck_pred, _, _ = self.model(xcheck)  # type: ignore
        check_recon = self.recon_error(xcheck_pred, xcheck)  # type: ignore

        plt.ecdf(recon)
        plt.ecdf(check_recon)
        plt.legend(["control", "test"], loc="upper left")
        plt.title("reconstruction errors")
        plt.xlabel("cosine(input|output)")
        plt.ylabel("eCDF")

    def show_dists(self, xcheck, method : Callable | None = None):
        if len(xcheck.shape) < 4:
            xcheck = xcheck.reshape(1, *xcheck.shape)
        method = self.manifold_distance if method is None else method
        vdist = method(self.test_example)
        cdist = method(xcheck)

        # want a score that reports the fraction of true positives at a threshold that gives 0.01 FP.
        cutoff = np.percentile(vdist, 95)
        score = 1.0 - np.interp(cutoff, np.sort(cdist), np.linspace(0, 1, len(cdist)))

        vline = plt.ecdf(vdist)
        cline = plt.ecdf(cdist)

        meth = str(method)
        meth = meth[meth.find("detector.") + len("detector.") :]
        meth = meth[: meth.find(" ")]

        plt.title(f"{meth}: {score:.3f}")
        plt.legend([vline, cline], ["control", "test"], loc="upper left")
        plt.xlabel("off-manifold distance")
        plt.ylabel("eCDF")

    def show_rk(self, xcheck, K=None):
        K = 0 if K is None else K

        method = self.ood_knn
        vdist = method(self.test_example)[:, K]
        cdist = method(xcheck)[:, K]

        # want a score that reports the fraction of true positives at a threshold that gives 0.05 FP.
        cutoff = np.percentile(vdist, 95)
        score = 1.0 - np.interp(cutoff, np.sort(cdist), np.linspace(0, 1, len(cdist)))

        vline = plt.ecdf(vdist)
        cline = plt.ecdf(cdist)
        xall, yall = plt.xlim(), plt.ylim()
        plt.plot(xall, [0.95, 0.95], "k:", [cutoff, cutoff], yall, "k:")

        meth = str(method)
        meth = meth[meth.find("detector.") + len("detector.") :]
        meth = meth[: meth.find(" ")]

        plt.title(f"{meth}: {score:.3f} (TPR@FP05)")
        plt.legend([vline, cline], ["control", "test"], loc="upper left")
        plt.xlabel("off-manifold distance")
        plt.ylabel("eCDF")

    def show_interpolated_images(self, X, use_self=None):
        if len(X.shape) < 4:
            X = X.reshape(1, *X.shape)

        use_self = False if use_self is None else use_self
        input_shape = X.shape
        maximg = 10
        nbtwn = 8  # display x0, nbtwn interpolates, and x1
        if len(X) > maximg:
            print(f"please do {maximg} or less at a time.")

        X_on_manifold, _ = self.get_latent_space_vectors(self.training_example)
        Xtest, _ = self.get_latent_space_vectors(X)

        interp_img = torch.zeros((nbtwn + 2, *input_shape[1:]))
        is_rgb = len(input_shape[1:]) == 3 and input_shape[1] > 1

        for ix, x in enumerate(Xtest):
            _, indices = self.get_dists_and_indices(x)
            alpha = np.linspace(0, 1, nbtwn + 2, dtype=Xtest.dtype)
            if use_self:
                x0, x1 = x, X_on_manifold[indices[0][0:1]]
                title = "self to NN1"
            else:
                x0, x1 = X_on_manifold[indices[0][0:2]]
                title = "NN1 to NN2"

            dx = x1 - x0
            dxnorm = np.sqrt(np.sum(dx**2))
            # angle between x0 and x1
            # cos_ang = np.sum(x0*x1)/(np.sqrt(np.sum(x0**2))*np.sqrt(np.sum(x1**2)))

            # angle between nn1 - x0 and nn2 - x0
            dnn1, dnn2 = (
                X_on_manifold[indices[0][0:1]] - x,
                X_on_manifold[indices[0][1:2]] - x,
            )

            cos_ang = np.clip(np.sum(dnn1 * dnn2) / (
                np.sqrt(np.sum(dnn1**2)) * np.sqrt(np.sum(dnn1**2))
            ), -1, 1)

            tri_area = 0.5 * np.sqrt(
                np.clip(np.sum(dnn1 * dnn1) * np.sum(dnn2 * dnn2) - np.sum(dnn1 * dnn2), 0, np.inf)
            )

            ang = np.arccos(cos_ang) * 180 / np.pi
            title += f", distance: {dxnorm:.2f}, angle: {ang:.1f} degrees, area: {tri_area:.1f}"

            mdev = next(self.model.parameters()).device
            interp_img[0] = (
                self.model.decode(torch.as_tensor(x0.reshape((1, -1))).to(mdev))
                .cpu()
                .detach()
                .reshape((1, *input_shape[1:]))
            )
            interp_img[-1] = (
                self.model.decode(torch.as_tensor(x1.reshape((1, -1))).to(mdev))
                .cpu()
                .detach()
                .reshape((1, *input_shape[1:]))
            )
            for i in range(nbtwn):
                xint = torch.as_tensor(x0 + dx * alpha[i]).to(self.device)
                this_img = (
                    self.model.decode(xint.reshape((1, -1)))
                    .detach()
                    .reshape((1, *input_shape[1:]))
                )

                interp_img[i + 1] = this_img

            if is_rgb:
                img_row = np.concatenate(
                    tuple(
                        [
                            np.transpose(img, axes=(1, 2, 0))
                            for img in interp_img.squeeze()
                        ]
                    ),
                    axis=1,
                )
            else:
                img_row = np.concatenate(
                    tuple([img for img in interp_img.squeeze()]), axis=1
                )

            fig = plt.figure(figsize=(4.8*3, 6.4*3))
            plt.imshow(img_row)
            plt.title(title)
            plt.show()

    def train_old(self, epochs=None):
        optimizer = self.optimizer
        self.model.train()
        xtrain = self.training_example  # vestigial from MNIST....stop doing this!

        epochs = (
            5000 if epochs is None else epochs
        )  # just training on subset x, so one iteration per "epoch".

        update_display_freq = max(int(epochs / 10), 1)
        for epoch in (pbar := tqdm(range(epochs))):
            # for epoch in range(1000):
            # Zero the gradients
            optimizer.zero_grad()

            # Calculate forward pass and loss
            pred_tuple = self.model(xtrain)
            if isinstance(self.criterion, tuple):
                recon_loss, reg_loss = self.criterion
                loss = recon_loss(xtrain, *pred_tuple) + self.beta * reg_loss(
                    xtrain, *pred_tuple
                )
            else:
                loss = self.criterion(xtrain, *pred_tuple)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Print progress
            if epoch % update_display_freq == 0 and self.verbose:
                pbar.set_description(f"Epoch: {epoch}, loss: {loss.item():.3f}")

        self.model.eval()
        self.test_pred = self.model(self.test_example)[0]
        if isinstance(self.test_pred, tuple):
            self.test_pred = self.test_pred[0]
        elif isinstance(self.test_pred, VAEOutput):
            self.test_pred = self.test_pred.x_recon

        mdim = int(np.round(np.max(self.manifold_dimension())))
        self.load_knn_model(mdim)
        self.load_normalized_knn_model(mdim)

    # ============================================================================================
    def train(self, num_epochs=None):
        num_epochs = 35 if num_epochs is None else num_epochs
        one_item = self.training_dataset.__getitem__(0)
        if isinstance(one_item, (tuple, list)):
            _, nx, ny = one_item[0].shape
        else:
            _, nx, ny = one_item.shape

        for epoch in (pbar := tqdm(range(num_epochs))):
            self.train_one_epoch(epoch, pbar)
            self.test(epoch=epoch, nx=nx, ny=ny)

        self.model.eval()
        test_data = next(iter(self.test_loader))
        if isinstance(test_data, (tuple, list)):
            test_data = test_data[0]
        test_data = torch.as_tensor(test_data).to(torch.float).to(self.device)
        self.test_pred = self.model(torch.as_tensor(test_data).to(torch.float).to(self.device))[0]

        mdim = int(np.round(np.max(self.manifold_dimension())))
        self.load_knn_model(mdim)
        self.load_normalized_knn_model(mdim)

    def train_one_epoch(self, epoch, pbar, nx=None, ny=None):
        trainds_len = self.train_loader.dataset.__len__()
        self.model.train().to(self.device)
        optimizer = self.optimizer
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            if isinstance(data, (tuple, list)):
                data = data[0]
            data = torch.as_tensor(data).to(torch.float).to(self.device)
            optimizer.zero_grad()

            # Run VAE
            recon_batch, mu, logvar = self.model(data)
            # Compute loss
            rec, kl = self.criterion(
                recon_batch,
                data,
                mu,
                logvar,
            )  # make this a parameter, not attribute of self

            total_loss = rec + kl
            total_loss.backward()
            train_loss += total_loss.item()
            optimizer.step()

            n_updates = 100
            update_period = max(int(len(self.train_loader)/n_updates), 1)
            if batch_idx % update_period == 0:
                logsigma_show = (
                    self.model.log_sigma.item()
                    if type(self.model.log_sigma) is torch.nn.Parameter
                    else self.model.log_sigma
                )
                
                # pbar.set_description(
                #     f"Train Epoch: {epoch:3} [{(batch_idx+1) * len(data):4}/{trainds_len:4} ({100.0 * (batch_idx + 1) / len(self.train_loader):3.0f}%)] test_loss: {self.previous_test_loss:8.2f} MSE: {rec.item() / len(data):10.1f}   KL: {kl.item() / len(data):8.1f}   log_sigma: {logsigma_show:.2f}"
                # )
                pbar.set_description(
                    f"Train Epoch: {epoch:3} train: {self.train_loss:8.2f} test: {self.previous_test_loss:8.2f} "
                )

        train_loss /= trainds_len
        self.train_loss = train_loss
        # print(f"====> Epoch: {epoch} Average loss: {train_loss:.4f}")


    def test(self, epoch, nx=None, ny=None, nchan=None):
        self.model.eval()
        nx = 28 if nx is None else nx
        ny = 28 if ny is None else ny
        nchan = 3 if nchan is None else nchan
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = torch.as_tensor(data).to(torch.float).to(self.device)
                recon_batch, mu, logvar = self.model(data)
                # Pass the second value from posthoc VAE
                rec, kl = self.criterion(recon_batch, data, mu, logvar)
                test_loss += rec + kl
 
        test_loss /= len(self.test_loader.dataset)
        self.previous_test_loss = test_loss
        # print("====> Test set loss: {:.4f}".format(test_loss), flush=True)
        # summary_writer.add_scalar("test/elbo", test_loss, epoch)

    # ============================================================================================

    def load_VAE(self, path=None):
        devmodel_name = "model_state.pth"
        path = devmodel_name if path is None else path
        try:
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict)

            ID = int(max(self.manifold_dimension()))
            self.load_knn_model(ID)
            self.model.eval()
        except FileNotFoundError:
            if path == devmodel_name:
                print("Training new dev model...")
                self.train()
                torch.save(self.model.state_dict(), devmodel_name)
            else:
                raise FileNotFoundError(f"No file named {path}")

        self.model.eval()
        self.test_pred = self.model(self.test_example)[0]

        mdim = int(np.round(np.max(self.manifold_dimension())))
        self.load_knn_model(mdim)
        self.load_normalized_knn_model(self.model.latent_dim)

    def load_knn_model(self, ID):
        etdata, _ = self.model.embed(self.training_example)
        data4knn = etdata.detach().cpu().reshape((self.training_example.shape[0], -1))
        self.neighbors = NearestNeighbors(n_neighbors=ID, algorithm="ball_tree").fit(
            data4knn
        )

    def load_normalized_knn_model(self, ID):
        tdata = self.training_example
        manifold = tdata
        data4knn, _ = self.model.embed(manifold)
        data4knn = data4knn.detach().cpu().reshape((manifold.shape[0], -1))
        norm = torch.sqrt(torch.sum(data4knn**2, 1, keepdim=True))
        data4knn /= norm

        self.normal_neighbors = NearestNeighbors(
            n_neighbors=ID, algorithm="ball_tree"
        ).fit(data4knn)

    def ood_knn(self, xstar):
        phistar, _ = self.model.embed(torch.as_tensor(xstar).to(torch.float).to(self.device))
        phistar = phistar.reshape((xstar.shape[0], -1))
        phistar_norm = torch.sqrt(torch.sum(phistar**2, 1, keepdim=True))
        zstar = (phistar / phistar_norm).cpu().detach().numpy()

        rk, _ = self.normal_neighbors.kneighbors(zstar)

        return rk

    def cdf_max_diff(self, val, test, small_null=None):
        small_null = False if small_null is None else small_null

        x, y = np.sort(val.reshape((-1))), np.sort(test.reshape((-1)))
        Fx, Fy = np.linspace(0, 1, len(x)), np.linspace(0, 1, len(y))

        Fxi, Fyi = np.interp(y, x, Fx), np.interp(x, y, Fy)

        allpts = np.concatenate((x, y))
        iord = np.argsort(allpts)
        allpts = allpts[iord]

        Fxall = np.concatenate((Fx, Fxi))[iord]
        Fyall = np.concatenate((Fyi, Fy))[iord]

        absdiff = np.abs(Fxall - Fyall)
        imax = np.argmax(absdiff)

        return absdiff[imax], allpts[imax]

    # get an image that really ought to be OOD. 
    def prepare_naruto(self):

        image = torch.as_tensor(FT.pil_to_tensor(Image.open("naruto_square.jpg")))
        nsquare = image.shape[1]

        n_channels, ny, nx = self.training_example.shape[1:]
        i0, i1, j0, j1 = 0, nsquare, 0, nsquare
        nsmall = int(nsquare*min(ny, nx)/max(ny, nx))
        margin = int((nsquare - nsmall)/2)
        if ny < nx:
            i0, i1 = margin, margin + nsmall
        elif nx > ny:
            j0, j1 = margin, margin + nsmall
    
        resized_image_naruto = FT.resize(image[i0:i1, j0:j1,:], [ny, nx], interpolation=FT.InterpolationMode.BILINEAR).to(torch.float)/255.0

        if n_channels != 3:
            narlayer = torch.mean(resized_image_naruto, 0, keepdim=True )
            naruto = torch.concatenate([narlayer]*n_channels, 0)
        else:
            naruto = resized_image_naruto.unsqueeze(0)
            
        return naruto

#
# The standalones reconstruction_loss, gaussian_nll, loss function and softclip are used in the train() method of 
#    OODdetector. 
#
def reconstruction_loss(x_hat, x):
    """Computes the likelihood of the data given the latent variable,
    in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1"""

    log_sigma = ((x - x_hat) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
    # self.log_sigma = log_sigma.item()

    # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
    # ensures stable training.
    log_sigma = softclip(log_sigma, -6)

    rec = gaussian_nll(x_hat, log_sigma, x).sum()

    return rec


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * np.log(2 * np.pi)
    )


def loss_function(recon_x, x, mu, logvar):
    # Important: both reconstruction and KL divergence loss have to be summed over all element!
    # Here we also sum the over batch and divide by the number of elements in the data later

    rec = reconstruction_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return rec, kl


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

