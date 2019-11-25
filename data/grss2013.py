from simplecv.data import preprocess
import numpy as np
from data.base import FullImageDataset
from skimage.external import tifffile
from data.base import minibatch_sample
from simplecv.data.preprocess import divisible_pad

SEED = 2333


class NewGRSS2013Dataset(FullImageDataset):
    def __init__(self,
                 image_path,
                 gt_path,
                 training=True,
                 sub_minibatch=10):
        self.im_mat_path = image_path
        self.gt_mat_path = gt_path

        image = tifffile.imread(image_path)
        mask = tifffile.imread(gt_path)

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = training
        self.sub_minibatch = sub_minibatch
        super(NewGRSS2013Dataset, self).__init__(image, mask, training, np_seed=SEED,
                                                 num_train_samples_per_class=None,
                                                 sub_minibatch=sub_minibatch)

    def preset(self):
        indicator = np.where(self.mask != 0, np.ones_like(self.mask), np.zeros_like(self.mask))

        blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              indicator[None, :, :]], axis=0)], 16, False)
        im = blob[0, :self.image.shape[-1], :, :]

        mask = blob[0, -2, :, :]
        self.indicator = blob[0, -1, :, :]

        if self.training:
            self.train_inds_list = minibatch_sample(mask, self.indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.indicator, self.sub_minibatch,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 15

    def __getitem__(self, idx):

        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]

        else:
            return self.pad_im, self.pad_mask, self.indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1
