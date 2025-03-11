import torch
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
from scipy.spatial import ConvexHull
from dataeval.utils.data.datasets import MNIST
from types import SimpleNamespace as blank_object
from typing import NamedTuple
from functools import partial

class MakeDataset(Dataset):
    def __init__(self, namespace):
        self.namespace = namespace
        for d in dir(namespace):
            if d[0:2] == '__' and d[-2:] == '__': 
                continue
            setattr(self, d, getattr(namespace, d))
    
    def __getitem__(self, idx):
        return self.namespace.__getitem__(idx)

    def __len__(self):
        return self.namespace.__len__()

class InstanceMNIST(blank_object):
    """
    Interface to corrupted MNIST, along with self-generated intrinsic metadata. The latter comes from a catalog of
    simple functions that compute something about each image. A user can easily add new functions to compute other 
    quantities of interest if desired. 
    """
    def __init__(self, corruptions=None, size=None, **kwargs):
        MNIST_NUM_IMAGES = 60000

        self.rng = np.random.default_rng(1234)
        ishuff = self.rng.permutation(MNIST_NUM_IMAGES)

        self.corruptions = [
            "identity",
            "shot_noise",
            "impulse_noise",
            "glass_blur",
            "motion_blur",
            "shear",
            "scale",
            "rotate",
            "brightness",
            "translate",
            "stripe",
            "fog",
            "spatter",
            "dotted_line",
            "zigzag",
            "canny_edges"
        ]
        
        if corruptions is None:
            corruptions = ['identity']
        if not isinstance(corruptions, list):
            corruptions = [corruptions]
 
        super().__init__()

        max_size = int(MNIST_NUM_IMAGES/len(corruptions))
        if size is None:
            size = max_size

        if size > max_size:
            raise ValueError(f'size {size} is too big, must bve less than {max_size} for {len(corruptions)} corruptions.')

        for ic, c in enumerate(corruptions):
            if not c in self.corruptions:
                print(f'Unknown corruption type {c}.')
                raise ValueError

            mnist = MNIST(root='./data', corruption=c, size=size, randomize=False, balance=False, verbose=False) # type: ignore
            images, labels = mnist._load_data_inner()
            images, labels = images[ishuff], labels[ishuff]

            images, labels = images[ic*size:ic*size+size], labels[ic*size:ic*size+size]
            images = (np.reshape(images, (size, 1, *images.shape[1:]))/255.0).astype(np.float32)


            nsamp, nchan, ny, nx = images.shape

            self.x, self.y = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))

            self.images, self.labels = images, labels # for use in self.make_metadata
            
            this_getitem = partial(self.__getitem__, c)

            setattr(self, c, MakeDataset(blank_object(corruption=c, images=images, labels=labels, metadata=self.make_metadata(), __getitem__=this_getitem, __len__=self.__len__)))

    def __getitem__(self, corruption, idx):
        myself = getattr(self, corruption)
        img = torch.tensor(myself.images[idx:idx+1, 0, :, :]) # idx:idx+1 yields a leading dimension of 1
        label = myself.labels[idx]
        metadata = myself.metadata[idx]

        return img, label, metadata

    def __len__(self):
        return self.images.shape[0]

    # catalog of functions for making intinsic metadata. 
    def make_metadata(self):
        xcm, ycm = self.avg_over_img(self.x), self.avg_over_img(self.y)
        cm = np.concatenate((xcm, ycm), axis=1)
        metadata_dict_list = [{"cm_x": xy[0], "cm_y": xy[1]} for xy in cm]

        # First metadata feature sets up the dict, so below we then can just 
        # add easy for loops that update it with additional features. 

        iso_count = self.isolated_pixel_count()
        for i, md in enumerate(metadata_dict_list):
            md.update({"isolated_pixels": iso_count[i]})

        bbox = self.bbox() # returns a dict containing numpy arrays
        for i, md in enumerate(metadata_dict_list): 
            for k in bbox:
                md.update({k: bbox[k][i]})

        sum_nzn_diffs = self.sum_abs_nzn_diffs()
        for i, md in enumerate(metadata_dict_list):
            md.update({"spikiness": sum_nzn_diffs[i]})

        fill_frac = self.fill_frac()
        for i, md in enumerate(metadata_dict_list):
            md.update({'fill_frac': fill_frac[i]})

        rando = self.random_normal()
        for i, md in enumerate(metadata_dict_list):
            md.update({'random': rando[i]})

        return metadata_dict_list

    # weighted average of quant, using pixel values as weights.
    def avg_over_img(self, quant):
        num = np.sum(self.images * quant, axis=(2, 3))
        denom = np.sum(self.images, axis=(2, 3))
        avg = num / denom

        return avg

    # per image count of the number of nonzero pixels that are surrounded by zeros. 
    def isolated_pixel_count(self):
        nz = self.images > 0
        nsum = np.zeros(self.images.shape)
        shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        for shift in shifts:
            nsum += np.roll(self.images, shift, axis=(2, 3))

        zero_neighbors = nsum == 0
        iso = np.logical_and(nz, zero_neighbors)
        count = np.sum(iso, axis=(2, 3))
        return count.reshape(-1)
    
    def bbox(self):
        bbox = np.zeros((len(self.images), 4), dtype=np.int8)
        x, y = self.x, self.y
        for i, img in enumerate(self.images):
            nz = (img > 0).squeeze()
            x0, x1 = np.min(x[nz]), np.max(x[nz])
            y0, y1 = np.min(y[nz]), np.max(y[nz])
            bbox[i,:] = (x0 + y0)/2, (y0 + y1)/2, x1 - x0, y1 - y0

        return {'x_ctr': bbox[:,0], 'y_ctr': bbox[:,1], 'width': bbox[:,2], 'height': bbox[:,3]}
    
    # for every nonzero pixel, add up absolute differences with nonzero neighbors. Yes, lots of double counting. 
    def sum_abs_nzn_diffs(self): 
        nz = self.images > 0
        ndiff = np.zeros(self.images.shape)
        shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        for shift in shifts:
            ndiff[nz] += np.abs(np.roll(self.images, shift, axis=(2, 3))[nz] - self.images[nz])
        
        sumnzn = np.sum(ndiff, axis=(2,3))
        return sumnzn.reshape(-1)
        
    def fill_frac(self):
        nz = self.images > 0
        nz_area = np.sum(nz, axis=(2,3))
        hull_area = np.zeros_like(nz_area)

        x, y = self.x, self.y
        for i, img in enumerate(self.images):
            nz = (img > 0).squeeze()
            xp, yp = self.x[nz].reshape((-1,1)), self.y[nz].reshape((-1,1))
            pts = np.concatenate((xp,yp), axis=1)
            hull_area[i] = ConvexHull(pts).volume # in 2D, volume attribute hold the enclosed plane area
      
        return (nz_area/hull_area).reshape(-1)
    
    def random_normal(self):  # valid metadata tests need to find this uninformative. 
        return self.rng.normal(size=len(self.images))
    

    
def collate_fn_2(batch):
    # The batch comes in the format ((x1, y1), (x2, y2), ..., (xn, yn)).
    # Let's split this up into your xs and your ys.
    xs, ys, md = list(zip(*batch)) # md arrives as a tuple of length batch_size, each element a dict
    # Let's create a tensor that concatenates all your images on a new axis.
    # Is there another way to do this?
    # xs = torch.cat([torch.tensor(i).unsqueeze(0) for i in xs], dim=0)
    xs = torch.stack(xs)
    # Let's create another tensor that combines all your class labels.
    ys = torch.tensor([i for i in ys])

    return xs, ys, list(md) # just pass md through and let utils.preprocess_metadata deal with it

def collate_fn_3(batch):
    # The batch comes in the format ((x1, y1), (x2, y2), ..., (xn, yn)).
    # Let's split this up into your xs and your ys.
    xs, ys, md = list(zip(*batch)) # md arrives as a tuple of length batch_size, each element a dict
    # Let's create a tensor that concatenates all your images on a new axis.
    # Is there another way to do this?
    # xs = torch.cat([torch.tensor(i).unsqueeze(0) for i in xs], dim=0)
    xs = torch.cat([torch.tensor(i).unsqueeze(0) for i in xs], dim=0)
    xs = xs.unsqueeze(1)
    # xs = torch.stack(xs)  # FAILS

    # Let's create another tensor that combines all your class labels.
    ys = torch.tensor([i for i in ys])

    return xs, ys, list(md) # just pass md through and let utils.preprocess_metadata deal with it
