import torch.utils.data
from data.base_data_loader import BaseDataLoader
from torch.utils.data.sampler import SequentialSampler

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset(opt)
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
        dataset.initialize(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt):
        super(CustomDatasetDataLoader,self).initialize(opt)
        print("Opt.nThreads = ", opt.nThreads)
        self.phase=opt.phase

        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            #shuffle=not opt.serial_batches,
            sampler=SequentialSampler(self.dataset),
            num_workers=int(opt.nThreads)
        )

        if self.phase=='pre_train':
            from data.aligned_dataset import AlignedValDataset
            self.valdataset=AlignedValDataset(opt)
            self.valdataloader = torch.utils.data.DataLoader(
                self.valdataset,
                batch_size=1,
                shuffle=not opt.serial_batches,
                num_workers=1
            )

    def load_data(self):
        if self.phase=='pre_train':
            return self.dataloader, self.valdataloader
        else:
            return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
