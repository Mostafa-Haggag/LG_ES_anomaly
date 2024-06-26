import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
# this is all the classses names
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, args, dataset_path, class_name, is_train=True, resize=384, transform_x=T.ToTensor()):
        super(MVTecDataset, self).__init__()
        # check that the class that you passed is consisting part of the clases of the dataset
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.args = args
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        # we are resizing to 384 +384//12 = 384+32
        self.resize = resize + self.args.image_size//12
        self.transform_x = transform_x

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

                                                  
        self.transform_mask = T.Compose([T.Resize(self.args.image_size, Image.NEAREST),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if mask == None:
            mask = torch.zeros([1, self.args.image_size, self.args.image_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []
        # phase indicating what should you do
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        # all the dataset images
        # you find a folder inside
        img_types = sorted(os.listdir(img_dir))
        # you are looping over good only
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)# adding the path of everything

            # load gt labels
            # it will enter in here ONLY if in training or validation
            if img_type == 'good': # you are training
                y.extend([0] * len(img_fpath_list))# you are putting 0 as label for my data
                mask.extend([None] * len(img_fpath_list))# you are putting you mask
            else:
                y.extend([1] * len(img_fpath_list))
                # you are putting 1 indicating that there is an anomley here
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)