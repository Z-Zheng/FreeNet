from torch.utils.data import dataset
import numpy as np
from simplecv.data.preprocess import divisible_pad
import torch
from torch.utils import data

SEED = 2333


class FullImageDataset(dataset.Dataset):
    def __init__(self,
                 image,
                 mask,
                 training,
                 np_seed=2333,
                 num_train_samples_per_class=200,
                 sub_minibatch=10,
                 ):
        self.image = image
        self.mask = mask
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.preset()

    def preset(self):
        train_indicator, test_indicator = fixed_num_sample(self.mask, self.num_train_samples_per_class,
                                                           self.num_classes, self._seed)

        blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :]], axis=0)], 16, False)
        im = blob[0, :self.image.shape[-1], :, :]

        mask = blob[0, -3, :, :]
        self.train_indicator = blob[0, -2, :, :]
        self.test_indicator = blob[0, -1, :, :]

        if self.training:
            self.train_inds_list = minibatch_sample(mask, self.train_indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.train_indicator, self.sub_minibatch,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 9

    def __getitem__(self, idx):

        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]

        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1


class MinibatchSampler(data.Sampler):
    def __init__(self, dataset: FullImageDataset):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(SEED)

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)


def fixed_num_sample(gt_mask: np.ndarray, num_train_samples, num_classes, seed=2333):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
        seed: int

    Returns:
        train_indicator, test_indicator
    """
    rs = np.random.RandomState(seed)

    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    for i in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == i)[0]
        rs.shuffle(inds)

        train_inds = inds[:num_train_samples]
        test_inds = inds[num_train_samples:]

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1

    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)

    return train_indicator, test_indicator


def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, minibatch_size, seed):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        train_indicator: 2-D array of shape [height, width]
        minibatch_size:

    Returns:

    """
    rs = np.random.RandomState(seed)
    # split into N classes
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = dict()
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        rs.shuffle(inds)

        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            left = cnt * minibatch_size
            if left >= len(inds):
                continue
            # remain last batch though the real size is smaller than minibatch_size
            right = min((cnt + 1) * minibatch_size, len(inds))
            fetch_inds = inds[left:right]
            train_inds[fetch_inds] = 1
        cnt += 1
        if train_inds.sum() == 0:
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))
