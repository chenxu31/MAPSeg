import pdb

import torch.utils.data as data
import os
from .data_utils import *
import torchio as tio
import torch
import numpy as np
import numpy
import random
import h5py
import platform
import sys


if platform.system() == "Windows":
    sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
    sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")
import common_brats_goat as common_brats
import common_metrics


class mae_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.src_f = None
        self.dst_f = None

        if cfg.data.task == "pelvic":
            pass
        elif cfg.data.task == "brats":
            self.subject_groups = common_brats.calc_subject_partitions()
        else:
            assert 0

    def __getitem__(self, index):
        if self.cfg.data.task == "pelvic":
            pass
        elif self.cfg.data.task == "brats":
            if self.src_f is None:
                self.src_f = h5py.File(os.path.join(self.cfg.data.mae_root, "train_%s.h5" % self.cfg.data.src_modality), "r")
                self.dst_f = h5py.File(os.path.join(self.cfg.data.mae_root, "train_%s.h5" % self.cfg.data.dst_modality), "r")

            if index // 2 == 0:
                subject_id = random.randint(0, self.subject_groups[0][1] - 1)
                tmp_scans = numpy.array(self.src_f["data"][self.subject_groups[0][0] + subject_id])
            else:
                subject_id = random.randint(0, self.subject_groups[1][1] - 1)
                tmp_scans = numpy.array(self.dst_f["data"][self.subject_groups[1][0] + subject_id])

            tmp_scans = tmp_scans.astype(numpy.float32) / 255.

        tmp_scans = random_flip(tmp_scans).copy()

        # whether to pad the image to match the patch size
        # and then cast to torch.tensor
        x, y, z = self.cfg.data.patch_size

        if min(tmp_scans.shape) < min(x, y, z):
            x_diff = x-tmp_scans.shape[0]
            y_diff = y-tmp_scans.shape[1]
            z_diff = z-tmp_scans.shape[2]
            tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        else:
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        _, x1, y1, z1 = tmp_scans.shape
        transforms = tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.75, 1.5), degrees=40,
                                      isotropic=True,
                                      default_pad_value=0, image_interpolation='linear')
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_scans = transforms(tmp_scans)

        # if remove_bg, the patch will only be sampled from the foreground (non-zero) region

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    x_idx = int((x1-x)/2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    y_idx = int((y1-y)/2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    z_idx = int((z1-z)/2)
        else:
            # if not remove_bg, the patch will be sampled from the whole image
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)
        # location indicates the sampled patch location
        location = torch.zeros_like(tmp_scans.data)
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                          bound[4]:bound[5]]))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data

        input_dict = {'local_patch': tmp_scans.data[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                      'global_images': down_scan}

        return input_dict

    def __len__(self):
        # we used fixed 2000 as the number of samples in each epoch
        # one can choose max(2000, self.all_img)
        # return max(2000, self.all_img)
        return 2000


class mpl_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # get all image paths (source and target)
        # folder should end with '_train'

        # data from target domain, only img (folder name should end with '_train')

        self.src_f = None
        self.dst_f = None

        if cfg.data.task == "pelvic":
            pass
        elif cfg.data.task == "brats":
            self.subject_groups = common_brats.calc_subject_partitions()
        else:
            assert 0

        print('num of source: ' + str(self.subject_groups[0][1]))
        print('num of target: ' + str(self.subject_groups[1][1]))

    def __getitem__(self, index):
        '''
        getitem for training/validation
        '''
        if self.cfg.data.task == "pelvic":
            pass
        elif self.cfg.data.task == "brats":
            if self.src_f is None:
                self.src_f = h5py.File(os.path.join(self.cfg.data.mae_root, "train_%s.h5" % self.cfg.data.src_modality), "r")
                self.dst_f = h5py.File(os.path.join(self.cfg.data.mae_root, "train_%s.h5" % self.cfg.data.dst_modality), "r")
                self.seg_f = h5py.File(os.path.join(self.cfg.data.mae_root, "train_seg.h5"), "r")

            '''
            load non-labeled data
            '''
            subject_id = random.randint(0, self.subject_groups[1][1] - 1)
            tmp_scansA = numpy.array(self.dst_f["data"][self.subject_groups[1][0] + subject_id])
            tmp_scansA = tmp_scansA.astype(numpy.float32) / 255.

            '''
            load annotated data
            '''
            subject_id = random.randint(0, self.subject_groups[0][1] - 1)
            tmp_scansB = numpy.array(self.src_f["data"][self.subject_groups[0][0] + subject_id])
            tmp_scansB = tmp_scansB.astype(numpy.float32) / 255.
            tmp_labelsB = numpy.array(self.seg_f["data"][self.subject_groups[0][0] + subject_id])

        tmp_labelsB[tmp_labelsB > 0] = 1
        x, y, z = self.cfg.data.patch_size

        # padding
        if min(tmp_scansA.shape) < min(x, y, z):
            x_diff = 96-tmp_scansA.shape[0]
            y_diff = 96-tmp_scansA.shape[1]
            z_diff = 96-tmp_scansA.shape[2]
            tmp_scansA = np.pad(tmp_scansA, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))))
            tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        else:
            tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        # augmentation
        _, x1, y1, z1 = tmp_scansA.shape
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.3), degrees=30,
                                                       isotropic=True,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest')

                                      ])
            tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
            tmp_scans = transforms(tmp_scans)
        else:
            tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
        # randomly select patch
        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    if bound[4] + z < z1:
                        z_idx = bound[4]
                    else:
                        z_idx = int((z1 - z) / 2)
        else:
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)

        location = torch.zeros_like(tmp_scans.data).float()
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                          bound[4]:bound[5]]), a_segmentation=tio.LabelMap(
            tensor=location[:, bound[0]:bound[1],
                            bound[2]:bound[3], bound[4]:bound[5]]
        ))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data
        locA = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locA)
        coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)

        patchA = tmp_scans.data[:, x_idx:x_idx + x,
                                y_idx:y_idx + y, z_idx:z_idx + z].float()
        downA = down_scan.float()

        '''
        load annotated data
        '''

        tmp_scans = tmp_scansB
        '''
        WARNING: HERE WE ONLY USE POSITIVE INTENSITY 
        FOR CT, USE PREPROCESSING TO turn negatives to positives 
        
        '''
        tmp_label = tmp_labelsB
        assert tmp_scans.shape == tmp_label.shape, 'scan and label must have the same shape'

        if min(tmp_scans.shape) < min(x, y, z):
            x_diff = x-tmp_scans.shape[0]
            y_diff = y-tmp_scans.shape[1]
            z_diff = z-tmp_scans.shape[2]
            tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location
            tmp_label = np.pad(tmp_label, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=0)  # pad with 0s bc it is label
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
            tmp_label = torch.unsqueeze(torch.from_numpy(tmp_label), 0)

        else:
            tmp_scans = torch.unsqueeze(
                torch.from_numpy(tmp_scans.copy()), 0)
            tmp_label = torch.unsqueeze(
                torch.from_numpy(tmp_label.copy()), 0)

        _, x1, y1, z1 = tmp_scans.shape
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_label = tio.LabelMap(tensor=tmp_label)
        sbj = tio.Subject(one_image=tmp_scans, a_segmentation=tmp_label)
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.4), degrees=30,
                                                       isotropic=True,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest'),
                                      tio.RandomBiasField(
                                      p=self.cfg.data.aug_prob),
                                      tio.RandomGamma(
                                      p=self.cfg.data.aug_prob, log_gamma=(-0.4, 0.4))
                                      ])
            sbj = transforms(sbj)
        tmp_scans = sbj['one_image'].data.float()
        tmp_label = sbj['a_segmentation'].data.float()

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    if bound[4] + z < z1:
                        z_idx = bound[4]
                    else:
                        z_idx = int((z1 - z) / 2)
        else:
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)

        location_B = torch.zeros_like(tmp_scans.data).float()
        location_B[:, x_idx:x_idx + x,
                   y_idx:y_idx + y, z_idx:z_idx + z] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                     bound[4]:bound[5]]),
                          a_segmentation=tio.LabelMap(
            tensor=location_B[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]])
        )
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data.float()
        locB = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locB)
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                     bound[4]:bound[5]]),
                          a_segmentation=tio.LabelMap(
            tensor=tmp_label[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]])
        )
        sbj = transforms(sbj)
        aux_label = sbj['a_segmentation'].data

        coordinates_B = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)
        input_dict = {'imgB': tmp_scans[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                      'labelB': torch.squeeze(tmp_label[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]),
                      'label_B_aux': torch.squeeze(aux_label),
                      'downB': down_scan,
                      'cord_B': coordinates_B,
                      'imgA': patchA,
                      'downA': downA,
                      'cord_A': coordinates_A}

        return input_dict

    def __len__(self):

        # we used fixed 100 steps for each epoch in finetuning
        # THIS PARAM WAS NEVER TUNED
        return 100
