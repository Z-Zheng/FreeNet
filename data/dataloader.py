from torch.utils.data.dataloader import DataLoader
from simplecv import registry
from data.pavia import NewPaviaDataset
from data.base import MinibatchSampler
from data.grss2013 import NewGRSS2013Dataset
from data.salinas import NewSalinasDataset


@registry.DATALOADER.register('NewPaviaLoader')
class NewPaviaLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewPaviaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch)
        sampler = MinibatchSampler(dataset)
        super(NewPaviaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=10
        ))


@registry.DATALOADER.register('NewSalinasLoader')
class NewSalinasLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewSalinasDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.num_train_samples_per_class, self.sub_minibatch)
        sampler = MinibatchSampler(dataset)
        super(NewSalinasLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=10
        ))


@registry.DATALOADER.register('NewGRSS2013Loader')
class NewGRSS2013Loader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewGRSS2013Dataset(self.image_path, self.gt_path, self.training, self.sub_minibatch)
        sampler = MinibatchSampler(dataset)
        super(NewGRSS2013Loader, self).__init__(dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                sampler=sampler,
                                                batch_sampler=None,
                                                num_workers=self.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                timeout=0,
                                                worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_path='',
            gt_path='',
            training=True,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=10
        ))
