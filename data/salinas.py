from scipy.io import loadmat
from simplecv.data import preprocess

from data.base import FullImageDataset

SEED = 2333


class NewSalinasDataset(FullImageDataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=200,
                 sub_minibatch=10):
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path

        im_mat = loadmat(image_mat_path)
        image = im_mat['salinas_corrected']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['salinas_gt']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        super(NewSalinasDataset, self).__init__(image, mask, training, np_seed=SEED,
                                                num_train_samples_per_class=num_train_samples_per_class,
                                                sub_minibatch=sub_minibatch)

    @property
    def num_classes(self):
        return 16
