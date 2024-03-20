# -*- coding: utf-8 -*-
"""dataset processing and loading
"""
import os
import random
import tensorflow as tf
import json
import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange
import csv

PI = np.pi

def change_coordinate_system(pos, center=None, rot_mat=None):
    """Transforms coordinates by applying an affine transformation

    Input
    ------
    pos : [batch_size, 3], float
        Coordinates

    center : [3], float or `None`
        Offset vector.
        Set to `None` for no offset.
        Defaults to `None`.

    rot_mat : [3,3], float or None
        Rotation matrix.
        Set to `None` to not apply a rotation.
        Defaults to `None`.

    Output
    -------
    pos : [batch_size, 3]
        Rotated and centered coordinates
    """
    if center is not None:
        pos -= center
    if rot_mat is not None:
        pos = tf.squeeze(tf.matmul(rot_mat, tf.reshape(pos, [3,1])))
    return pos

def acos_diff(x, epsilon=1e-7):
    r"""
    Implementation of arccos(x) that avoids evaluating the gradient at x
    -1 or 1 by using straight through estimation, i.e., in the
    forward pass, x is clipped to (-1, 1), but in the backward pass, x is
    clipped to (-1 + epsilon, 1 - epsilon).

    Input
    ------
    x : any shape, tf.float
        Value at which to evaluate arccos

    epsilon : tf.float
        Small backoff to avoid evaluating the gradient at -1 or 1.
        Defaults to 1e-7.

    Output
    -------
     : same shape as x, tf.float
        arccos(x)
    """

    x_clip_1 = tf.clip_by_value(x, -1., 1.)
    x_clip_2 = tf.clip_by_value(x, -1. + epsilon, 1. - epsilon)
    eps = tf.stop_gradient(x - x_clip_2)
    x_1 =  x - eps
    acos_x_1 =  tf.acos(x_1)
    y = acos_x_1 + tf.stop_gradient(tf.acos(x_clip_1)-acos_x_1)
    return y

def theta_phi_from_unit_vec(v):
    r"""
    Computes zenith and azimuth angles (:math:`\theta,\varphi`)
    from unit-norm vectors as described in :eq:`theta_phi`

    Input
    ------
    v : [...,3], tf.float
        Tensor with unit-norm vectors in the last dimension

    Output
    -------
    theta : [...], tf.float
        Zenith angles :math:`\theta`

    phi : [...], tf.float
        Azimuth angles :math:`\varphi`
    """
    x = v[...,0]
    y = v[...,1]
    z = v[...,2]

    # Clip to ensure numerical stability
    theta = acos_diff(z)
    phi = tf.math.atan2(y, x)
    return theta, phi

def normalize(v):
    r"""
    Normalizes ``v`` to unit norm

    Input
    ------
    v : [...,3], tf.float
        Vector

    Output
    -------
    : [...,3], tf.float
        Normalized vector

    : [...], tf.float
        Norm of the unnormalized vector
    """
    norm = tf.norm(v, axis=-1, keepdims=True)
    n_v = tf.math.divide_no_nan(v, norm)
    norm = tf.squeeze(norm, axis=-1)
    return n_v, norm

def rotation_matrix(angles):
    r"""
    Computes rotation matrices as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) [TR38901]_ is used.

    Input
    ------
    angles : [...,3], tf.float
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    Output
    -------
    : [...,3,3], tf.float
        Rotation matrices
    """

    a = angles[...,0]
    b = angles[...,1]
    c = angles[...,2]
    cos_a = tf.cos(a)
    cos_b = tf.cos(b)
    cos_c = tf.cos(c)
    sin_a = tf.sin(a)
    sin_b = tf.sin(b)
    sin_c = tf.sin(c)

    r_11 = cos_a*cos_b
    r_12 = cos_a*sin_b*sin_c - sin_a*cos_c
    r_13 = cos_a*sin_b*cos_c + sin_a*sin_c
    r_1 = tf.stack([r_11, r_12, r_13], axis=-1)

    r_21 = sin_a*cos_b
    r_22 = sin_a*sin_b*sin_c + cos_a*cos_c
    r_23 = sin_a*sin_b*cos_c - cos_a*sin_c
    r_2 = tf.stack([r_21, r_22, r_23], axis=-1)

    r_31 = -sin_b
    r_32 = cos_b*sin_c
    r_33 = cos_b*cos_c
    r_3 = tf.stack([r_31, r_32, r_33], axis=-1)

    rot_mat = tf.stack([r_1, r_2, r_3], axis=-2)
    return rot_mat

def get_coordinate_system():
    """Get points of interest as well as coordinate transformation parameters

    Output
    -------
    center : [3]
        Offset vector

    rot_mat : [3,3]
        Rotation matrix

    poi : dict
        Dictionary with points of interest and their coordinates
    """

    # Read and parse coordinates of points of interest (POIs)
    data_dict = {}
    with open('/home/anplus/Documents/GitHub/diff-rt-calibration/data/tfrecords/coordinates.csv', mode ='r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            data_dict[row['Name']] = {k: v for k, v in row.items() if k != 'Name'}
    # print(data_dict)
    poi = {}
    for val in data_dict:
        pos = data_dict[val]
        if not pos["South"]=="noLoS":
            poi[val] = tf.constant([float(pos["West"]),
                                    float(pos["South"]),
                                    float(pos["Height"])], tf.float32)

    # Add antenna array position
    poi["array_1"] = tf.constant([7.480775, -20.9824, 1.39335], tf.float32)
    poi["array_2"] = tf.constant([-6.390425, 24.440075, 1.4197], tf.float32)

    # Center coordinate system
    center = poi["AU"] # This is our new origin
    for p in poi:
        poi[p] = change_coordinate_system(poi[p], center)

    # Compute rotation matrix such that NWU has coordinates [0, 1, ~0]
    nwu_hat, _ = normalize(poi["NWU"])
    theta, phi = theta_phi_from_unit_vec(nwu_hat)
    rot_mat = tf.squeeze(rotation_matrix(tf.constant([[-phi.numpy()+PI/2, 0, 0]], tf.float32)))

    # Rotate all POIs to match the new coordinate system
    for p in poi:
        poi[p] = change_coordinate_system(poi[p], rot_mat=rot_mat)

    return center, rot_mat, poi

# def rssi2amplitude(rssi):
#     """convert rssi to amplitude
#     """
#     return 100 * 10 ** (rssi / 20)


# def amplitude2rssi(amplitude):
#     """convert amplitude to rssi
#     """
#     return 20 * np.log10(amplitude / 100)


def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


def split_dataset(datadir, ratio=0.8, dataset_type='rfid', num_samples=1000):
    """random shuffle train/test set
    """
    index = []
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)
    elif dataset_type == "ble":
        rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        index = pd.read_csv(rssi_dir).index.values
        random.shuffle(index)
    elif dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.pt')
        index = [i for i in range(torch.load(csi_dir).shape[0])]
        random.shuffle(index)
    elif dataset_type == "dichasus-crosslink":
        index = np.arange(0, num_samples)
        random.shuffle(index)
    elif dataset_type == "dichasus-fdd":
        index = np.arange(0, num_samples)
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')




class Spectrum_dataset(Dataset):
    """spectrum dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1) -> None:
        super().__init__()
        self.datadir = datadir
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_info.yml')
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])
        example_spt = imageio.imread(os.path.join(self.spectrum_dir, self.spt_names[0]))
        self.n_elevation, self.n_azimuth = example_spt.shape
        self.rays_per_spectrum = self.n_elevation * self.n_azimuth
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        self.nn_inputs, self.nn_labels = self.load_data()


    def __len__(self):
        return len(self.dataset_index) * self.rays_per_spectrum


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        train_inputs : tensor. [n_samples, 9]. The inputs for training
                  ray_o, ray_d, tx_pos
        """
        ## NOTE! Each spectrum will cost 1.2 MB of memory. Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 9)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        ## Load gateway position and orientation
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_info = yaml.safe_load(f)
            gateway_pos = gateway_info['gateway1']['position']
            gateway_orientation = gateway_info['gateway1']['orientation']

        ## Load transmitter position
        tx_pos = pd.read_csv(self.tx_pos_dir).values
        tx_pos = torch.tensor(tx_pos, dtype=torch.float32)

        ## Load data, each spectrum contains 90x360 pixels(rays)
        for i, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            spectrum = imageio.imread(os.path.join(self.spectrum_dir, idx + '.png')) / 255
            spectrum = torch.tensor(spectrum, dtype=torch.float32).view(-1, 1)
            ray_o, ray_d = self.gen_rays_spectrum(gateway_pos, gateway_orientation)
            tx_pos_i = torch.tile(tx_pos[int(idx)-1], (self.rays_per_spectrum,)).reshape(-1,3)  # [n_rays, 3]
            nn_inputs[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :9] = \
                torch.cat([ray_o, ray_d, tx_pos_i], dim=1)
            nn_labels[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :] = spectrum

        return nn_inputs, nn_labels


    def gen_rays_spectrum(self, gateway_pos, gateway_orientation):
        """generate sample rays origin at gateway with resolution given by spectrum

        Parameters
        ----------
        azimuth : int. The number of azimuth angles
        elevation : int. The number of elevation angles

        Returns
        -------
        r_o : tensor. [n_rays, 3]. The origin of rays
        r_d : tensor. [n_rays, 3]. The direction of rays, unit vector
        """

        azimuth = torch.linspace(1, 360, self.n_azimuth) / 180 * np.pi
        elevation = torch.linspace(1, 90, self.n_elevation) / 180 * np.pi
        azimuth = torch.tile(azimuth, (self.n_elevation,))  # [1,2,3...360,1,2,3...360,...] pytorch 2.0
        elevation = torch.repeat_interleave(elevation, self.n_azimuth)  # [1,1,1,...,2,2,2,...,90,90,90,...]

        x = 1 * torch.cos(elevation) * torch.cos(azimuth) # [n_azi * n_ele], i.e., [n_rays]
        y = 1 * torch.cos(elevation) * torch.sin(azimuth)
        z = 1 * torch.sin(elevation)

        r_d = torch.stack([x, y, z], dim=0)  # [3, n_rays] 3D direction of rays in gateway coordinate
        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        r_d = R @ r_d  # [3, n_rays] 3D direction of rays in world coordinate
        gateway_pos = torch.tensor(gateway_pos, dtype=torch.float32)
        r_o = torch.tile(gateway_pos, (self.rays_per_spectrum,)).reshape(-1, 3)  # [n_rays, 3]

        return r_o, r_d.T





class BLE_dataset(Dataset):
    """ble dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1) -> None:
        super().__init__()
        self.datadir = datadir
        tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_position.yml')
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load gateway position
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_pos_dict = yaml.safe_load(f)
            self.gateway_pos = torch.tensor([pos for pos in gateway_pos_dict.values()], dtype=torch.float32)
            self.gateway_pos = self.gateway_pos / scale_worldsize
            self.n_gateways = len(self.gateway_pos)

        # Load transmitter position
        self.tx_poses = torch.tensor(pd.read_csv(tx_pos_dir).values, dtype=torch.float32)
        self.tx_poses = self.tx_poses / scale_worldsize

        # Load gateway received RSSI
        self.rssis = torch.tensor(pd.read_csv(self.rssi_dir).values, dtype=torch.float32)

        self.nn_inputs, self.nn_labels = self.load_data()


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        nn_inputs : tensor. [n_samples, 978]. The inputs for training
                    tx_pos:3, ray_o:3, ray_d:9x36x3,
        nn_labels : tensor. [n_samples, 1]. The RSSI labels for training
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        ## generate rays origin at gateways
        gateways_ray_o, gateways_rays_d = self.gen_rays_gateways()

        ## Load data
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):
            rssis = self.rssis[idx]
            tx_pos = self.tx_poses[idx].view(-1)  # [3]
            for i_gateway, rssi in enumerate(rssis):
                if rssi != -100:
                    gateway_ray_o = gateways_ray_o[i_gateway].view(-1)  # [3]
                    gateway_rays_d = gateways_rays_d[i_gateway].view(-1)  # [n_rays x 3]
                    nn_inputs[data_counter] = torch.cat([tx_pos, gateway_ray_o, gateway_rays_d], dim=-1)
                    nn_labels[data_counter] = rssi
                    data_counter += 1

        nn_labels = rssi2amplitude(nn_labels)

        return nn_inputs, nn_labels


    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_gateways, 1, 3]. The origin of rays
        r_d : tensor. [n_gateways, n_rays, 3]. The direction of rays, unit vector
        """


        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.n_gateways, self.beta_res * self.alpha_res, 3])  # [n_gateways, 9*36, 3]
        r_o = self.gateway_pos.unsqueeze(1) # [21, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d


    def __len__(self):
        rssis = self.rssis[self.dataset_index]
        return torch.sum(rssis != -100)

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]




class CSI_dataset(Dataset):

    def __init__(self, datadir, indexdir, scale_worldsize=1):
        """ datasets [datalen*8, up+down+r_o+r_d] --> [datalen*8, 26+26+3+36*3]
        """
        super().__init__()
        self.datadir = datadir
        self.csidata_dir = os.path.join(datadir, 'csidata.npy')
        self.bs_pos_dir = os.path.join(datadir, 'base-station.yml')
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load base station position
        with open(os.path.join(self.bs_pos_dir)) as f:
            bs_pos_dict = yaml.safe_load(f)
            self.bs_pos = torch.tensor([bs_pos_dict["base_station"]], dtype=torch.float32).squeeze()
            self.bs_pos = self.bs_pos / scale_worldsize
            self.n_bs = len(self.bs_pos)

        # load CSI data
        csi_data = torch.from_numpy(np.load(self.csidata_dir))  #[N, 8, 52]
        csi_data = self.normalize_csi(csi_data)
        uplink, downlink = csi_data[..., :26], csi_data[..., 26:]
        up_real, up_imag = torch.real(uplink), torch.imag(uplink)
        down_real, down_imag = torch.real(downlink), torch.imag(downlink)
        self.uplink = torch.cat([up_real, up_imag], dim=-1)    # [N, 8, 52]
        self.downlink = torch.cat([down_real, down_imag], dim=-1)    # [N, 8, 52]
        self.uplink = rearrange(self.uplink, 'n g c -> (n g) c')    # [N*8, 52]
        self.downlink = rearrange(self.downlink, 'n g c -> (n g) c')    # [N*8, 52]

        self.nn_inputs, self.nn_labels = self.load_data()


    def normalize_csi(self, csi):
        self.csi_max = torch.max(abs(csi))
        return csi / self.csi_max

    def denormalize_csi(self, csi):
        assert self.csi_max is not None, "Please normalize csi first"
        return csi * self.csi_max


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 1027]. The inputs for training
                    uplink: 52 (26 real; 26 imag), ray_o: 3, ray_d: 9x36x3, n_samples = n_dataset * n_bs
        nn_labels : tensor. [n_samples, 52]. The downlink channels as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 52+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 52)), dtype=torch.float32)

        ## generate rays origin at gateways
        bs_ray_o, bs_rays_d = self.gen_rays_gateways()
        bs_ray_o = rearrange(bs_ray_o, 'n g c -> n (g c)')   # [n_bs, 1, 3] --> [n_bs, 3]
        bs_rays_d = rearrange(bs_rays_d, 'n g c -> n (g c)') # [n_bs, n_rays, 3] --> [n_bs, n_rays*3]

        ## Load data
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):
            bs_uplink = self.uplink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            bs_downlink = self.downlink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            nn_inputs[data_counter*self.n_bs: (data_counter+1)*self.n_bs] = torch.cat([bs_uplink, bs_ray_o, bs_rays_d], dim=-1) # [n_bs, 52+3+3*36*9]
            nn_labels[data_counter*self.n_bs: (data_counter+1)*self.n_bs]  = bs_downlink
            data_counter += 1
        return nn_inputs, nn_labels

    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_bs, 1, 3]. The origin of rays
        r_d : tensor. [n_bs, n_rays, 3]. The direction of rays, unit vector
        """
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.n_bs, self.beta_res * self.alpha_res, 3])  # [n_bs, 9*36, 3]
        r_o = self.bs_pos.unsqueeze(1) # [n_bs, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def __len__(self):
        return len(self.dataset_index) * self.n_bs


class DichasusDC01Dataset_crosslink(Dataset):
    def __init__(self, datadir, indexdir, scale_worldsize=1, calibrate=True, y_filter=None, num_samples=3000):
        self.datadir = datadir
        self.num_samples = num_samples
        self.csidata_dir = os.path.join(datadir, 'dichasus-dc01.tfrecords')
        self.bs_pos_dir = os.path.join(datadir, 'base-station.yml')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        self.calibrate = calibrate
        self.y_filter = y_filter
        # get coordinates
        self.center, self.rot_mat, _ = get_coordinate_system()

        # load base station position
        with open(os.path.join(self.bs_pos_dir)) as f:
            bs_pos_dict = yaml.safe_load(f)
            bs_pos = tf.constant(bs_pos_dict["base_station"])
            self.n_bs = len(bs_pos)
            bos_pos_collection = []
            for i in range(len(bs_pos)):
                bs_pos_ = change_coordinate_system(bs_pos[i, :], center=self.center, rot_mat=self.rot_mat)
                bs_pos_ = torch.from_numpy(bs_pos_.numpy())
                bos_pos_collection.append(bs_pos_)
            bs_pos = torch.stack(bos_pos_collection).squeeze()
            # print("bs_pos", bs_pos)
            self.bs_pos = bs_pos / scale_worldsize
            self.n_bs = len(bs_pos)

        ## generate rays origin at gateways
        bs_ray_o, bs_rays_d = self.gen_rays_gateways()
        self.bs_ray_o = rearrange(bs_ray_o, 'n g c -> n (g c)')   # [n_bs, 1, 3] --> [n_bs, 3]
        self.bs_rays_d = rearrange(bs_rays_d, 'n g c -> n (g c)') # [n_bs, n_rays, 3] --> [n_bs, n_rays*3]

        self._gen_dataset()

    def _gen_dataset(self):
        # load CSI data
        self.dataset = self._load_dataset(self.datadir, self.calibrate, self.y_filter)
        self.dataset = self.dataset.shuffle(seed=42, reshuffle_each_iteration=False, buffer_size=1024).batch(1)
        dataset_iter = iter(self.dataset)

        NUM_data = self.num_samples
        nn_inputs = torch.tensor(np.zeros((NUM_data * self.n_bs, 3 + 3 + 3 * self.alpha_res * self.beta_res)),
                                 dtype=torch.float32)  # n_bs = 2
        nn_labels = torch.tensor(np.zeros((NUM_data * self.n_bs, 1024 * 2)), dtype=torch.float32)
        csi_collection = []
        pos_collection = []
        # Convert TensorFlow dataset to list of tuples (csi, pos) for easier access
        for it_num in tqdm(range(NUM_data)):
            next_item = next(dataset_iter, None)
            # Stop if exhausted
            if next_item is None:
                break
            # Retrieve the position
            csi, pos = next_item
            pos = torch.from_numpy(pos.numpy()).repeat(self.n_bs, 1).squeeze()
            # [1, num_tx (2) * num_tx_ant (32) = 64, num_subcarriers = 1024]
            csi = torch.from_numpy(csi.numpy()).squeeze()
            csi_collection.append(csi)
            pos_collection.append(pos)

        csi_collection = self.normalize_csi(torch.stack(csi_collection, dim=0))
        for it_num in tqdm(range(NUM_data)):
            csi = csi_collection[it_num]
            pos = pos_collection[it_num]
            csi_real, csi_imag = torch.real(csi), torch.imag(csi)
            csi_ = torch.cat([csi_real, csi_imag], dim=-1)
            nn_inputs[it_num * self.n_bs: (it_num + 1) * self.n_bs] = torch.cat([pos, self.bs_ray_o, self.bs_rays_d],
                                                                                dim=-1)  # [n_bs, 52+3+3*36*9]
            nn_labels[it_num * self.n_bs: (it_num + 1) * self.n_bs] = csi_.view(self.n_bs, 2 * 1024)

        self.nn_inputs = nn_inputs[self.dataset_index]
        self.nn_labels = nn_labels[self.dataset_index]

    def normalize_csi(self, csi):
        self.csi_max = torch.max(abs(csi))
        return csi / self.csi_max

    def denormalize_csi(self, csi):
        assert self.csi_max is not None, "Please normalize csi first"
        return csi * self.csi_max

    def _load_dataset(self, tfrecord_path, calibrate=True, y_filter=None):
        raw_dataset = tf.data.TFRecordDataset(self.csidata_dir)


        feature_description = {
            "cfo": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "csi": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "gt-interp-age-tachy": tf.io.FixedLenFeature([], tf.float32, default_value=0),
            "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "snr": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
        }

        def record_parse_function(proto):
            record = tf.io.parse_single_example(proto, feature_description)
            csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (64, 1024, 2))
            csi = tf.signal.fftshift(csi, axes=1)
            csi = tf.complex(csi[..., 0], csi[..., 1])
            pos = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))
            pos = tf.cast(pos, tf.float32)
            pos = change_coordinate_system(pos, center=self.center, rot_mat=self.rot_mat)
            return csi, pos

        def apply_calibration(csi, pos):
            """Apply STO and CPO calibration"""
            sto_offset = tf.tensordot(tf.constant(offsets["sto"]),
                                      2 * np.pi * tf.range(tf.shape(csi)[1], dtype=np.float32) / tf.cast(
                                          tf.shape(csi)[1], np.float32), axes=0)
            cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype=np.float32), axes=0)
            csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))
            return csi, pos

        dataset = raw_dataset.map(record_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if calibrate:
            calibration_file = os.path.join(self.datadir, 'reftx-offsets-dichasus-dc01.json')
            with open(calibration_file, "r") as offsetfile:
                offsets = json.load(offsetfile)
            dataset = dataset.map(apply_calibration)

        if y_filter is not None:
            def position_filter(csi, pos):
                "Limit y-range to certain range to avoid position too close to the receiver"
                return pos[1] > y_filter[0] and pos[1] < y_filter[1]

            dataset = dataset.filter(position_filter)

        return dataset


    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_bs, 1, 3]. The origin of rays
        r_d : tensor. [n_bs, n_rays, 3]. The direction of rays, unit vector
        """
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 2]
        r_d = r_d.expand([self.n_bs, self.beta_res * self.alpha_res, 3])  # [n_bs, 9*36, 3]
        r_o = self.bs_pos.unsqueeze(1) # [n_bs, 1, 2]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d

    def __len__(self):
        return len(self.nn_inputs)

    def __getitem__(self, idx):
        return self.nn_inputs[idx], self.nn_labels[idx]

class DichasusDC01Dataset_fdd(DichasusDC01Dataset_crosslink):
    def __init__(self, datadir, indexdir, scale_worldsize=1, calibrate=True, y_filter=None, num_samples=1000):
        # Initialize the superclass with some of the same parameters
        super().__init__(datadir, indexdir, scale_worldsize, calibrate, y_filter, num_samples=num_samples)

    def _gen_dataset(self):
        print("FDD dataset generation")
        self.dataset = self._load_dataset(self.datadir, self.calibrate, self.y_filter)
        self.dataset = self.dataset.shuffle(seed=42, reshuffle_each_iteration=False, buffer_size=1024).batch(1)
        dataset_iter = iter(self.dataset)

        NUM_data = self.num_samples
        nn_inputs = torch.tensor(np.zeros((NUM_data * self.n_bs, 512 * 2 + 3 + 3 * self.alpha_res * self.beta_res)),
                                 dtype=torch.float32)  # n_bs = 2
        nn_labels = torch.tensor(np.zeros((NUM_data * self.n_bs, 512 * 2)), dtype=torch.float32)

        csi_collection = []
        # Convert TensorFlow dataset to list of tuples (csi, pos) for easier access
        for it_num in tqdm(range(NUM_data)):
            next_item = next(dataset_iter, None)
            # Stop if exhausted
            if next_item is None:
                break
            # Retrieve the position
            csi, pos = next_item
            pos = torch.from_numpy(pos.numpy()).repeat(self.n_bs, 1).squeeze()
            # [num_tx (2) * num_tx_ant (32) = 64, num_subcarriers = 1024]
            csi = torch.from_numpy(csi.numpy()).squeeze()
            csi_collection.append(csi)

        csi_collection = self.normalize_csi(torch.stack(csi_collection, dim=0))
        for it_num in tqdm(range(NUM_data)):
            csi = csi_collection[it_num]
            uplink, downlink = csi[..., :512], csi[..., 512:]
            up_real, up_imag = torch.real(uplink), torch.imag(uplink)
            down_real, down_imag = torch.real(downlink), torch.imag(downlink)
            uplink = torch.cat([up_real, up_imag], dim=-1)  # [64, 512*2]
            downlink = torch.cat([down_real, down_imag], dim=-1)  # [64, 512*2]
            nn_inputs[it_num * self.n_bs: (it_num + 1) * self.n_bs] = torch.cat([uplink,
                                                                                 self.bs_ray_o, self.bs_rays_d],
                                                                                dim=-1)  # [n_bs, 512*2+3+3*36*9]
            nn_labels[it_num * self.n_bs: (it_num + 1) * self.n_bs] = downlink.view(self.n_bs, 2 * 512)

        self.nn_inputs = nn_inputs[self.dataset_index]
        self.nn_labels = nn_labels[self.dataset_index]



dataset_dict = {"rfid": Spectrum_dataset, "ble": BLE_dataset, "mimo": CSI_dataset,
                "dichasus-crosslink": DichasusDC01Dataset_crosslink,
                "dichasus-fdd": DichasusDC01Dataset_fdd}


if __name__ == "__main__":
    datadir = os.path.join("/home/anplus/Documents/GitHub/diff-rt-calibration/data/tfrecords")
    train_index = os.path.join(datadir, "train_index.txt")
    test_index = os.path.join(datadir, "test_index.txt")
    num_samples = 1000
    if not os.path.exists(train_index) or not os.path.exists(test_index):
        split_dataset(datadir, ratio=0.8, dataset_type='dichasus-crosslink', num_samples=num_samples)
    dataset = DichasusDC01Dataset_crosslink(datadir, train_index, scale_worldsize=1, calibrate=True, y_filter=[-5,5],
                                            num_samples=num_samples)
    # dataset = DichasusDC01Dataset_fdd(datadir, train_index, scale_worldsize=1, calibrate=True, y_filter=[-5, 10],
    # num_samples=num_samples)
