# skeleton class for dataloader
# import sys
# sys.path.append('./src')

import os, os.path as osp
import torch
from torch.utils.data import DataLoader
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def plot_point_cloud(points, colors=None):
    import plotly.graph_objects as go

    fig = go.Figure(
    data=[
        go.Scatter3d(
            x=points[:,0], y=points[:,1], z=points[:,2], 
            mode='markers',
            marker=dict(size=1)
        )
    ],
    layout=dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    )
    fig.show()

class AcidDataset(Dataset):
    def __init__(self, root_dir, mode = "val", categories = [], transform=False, all_points=False):

        self.files = []
        self.all_points = all_points
        self.mode = mode

        for category in categories:
            self.files.extend(glob.glob(os.path.join(root_dir, category, "*.npy")))

        # make train and test split with seed
        np.random.seed(0)
        np.random.shuffle(self.files)
        ratio = 0.7
        self.train_files = self.files[:int(ratio * len(self.files))]
        self.test_files = self.files[int(ratio * len(self.files)):]
        
        if self.mode == "train":
            self.files = self.train_files
            self.transform = transform
        elif self.mode == "val":
            self.files = self.test_files
            self.transform = False
        else:
            raise ValueError("mode must be either train or val")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx], allow_pickle=True).item()
        point_cloud = data["point_cloud"]
        coords = data["pts"]
        occ = data["occ"]

        if self.all_points:
            points = data["all_points"]

        # convert to tensor
        point_cloud = torch.from_numpy(point_cloud).float()
        coords = torch.from_numpy(coords).float()
        if self.all_points:
            points = torch.from_numpy(points).float()
        occ = torch.from_numpy(occ).float()
        occ = 2 * occ - 1 # convert occ to -1, 1

        # randomly rotate point_cloud and coords
        if self.transform:
            from scipy.stats import special_ortho_group
            R = special_ortho_group.rvs(3)
            # rotate point cloud and coords
            point_cloud = torch.matmul(point_cloud, torch.from_numpy(R).float())
            coords = torch.matmul(coords, torch.from_numpy(R).float())
            if self.all_points:
                points = torch.matmul(points, torch.from_numpy(R).float())

        if self.all_points:
            return {"point_cloud": point_cloud, "coords": coords, "all_points": points}, {"occ": occ}
        else:
            return {"point_cloud": point_cloud, "coords": coords}, {"occ": occ}


