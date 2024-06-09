import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

from .dataset import MVTecDataset
from .mean_std import obj_stats_384


def get_dataloader(args):
    # using the mean and std for normalizing
    obj_mean, obj_std = obj_stats_384(args.obj)
    trainT = T.Compose(
        [
            T.Resize(args.image_size, Image.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    validT = T.Compose(
        [
            T.Resize(args.image_size, Image.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    testT = T.Compose(
        [
            T.Resize(args.image_size, Image.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    train_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=True,
        resize=args.image_size,
        transform_x=trainT,
    )
    valid_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=True,
        resize=args.image_size,
        transform_x=validT,
    )
    # length of train dataset
    img_nums = len(train_dataset)
    # list of all the possible indices
    indices = list(range(img_nums))
    # setting the numpy seed to specific value
    np.random.seed(args.seed)
    # shuffle the indices
    np.random.shuffle(indices)
    # this is the split of the indics
    split = int(np.floor(args.val_ratio * img_nums))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    # the object that we are working on is the "grid"
    # the image size is 384
    test_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=False,
        resize=args.image_size,
        transform_x=testT,
    )
    # i am passing the sampelr with specific indiecs
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=valid_sampler
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
