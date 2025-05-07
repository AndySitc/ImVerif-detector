import os
import pandas as pd
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import PIL

def load_dataset(datadir):
    deepfake_image_files = ['big_gan', 'cips', 'denoising_diffusion_gan', 'diffusion_gan', 'face_synthetics', 'gansformer', 'lama', 
                        'palette', 'sfhq', 'stable_diffusion', 'taming_transformer']
    real_image_files = ['afhq', 'celebahq', 'coco', 'ffhq', 'imagenet', 'landscape', 'lsun', 'metfaces']
    folders = deepfake_image_files + real_image_files
    # Create dataframe
    train_df, test_df = pd.DataFrame(), pd.DataFrame()

    # Load fake data only if in classification mode
    for subfolder in folders:
        data_tmp = pd.read_csv(datadir+ f"/{subfolder}/metadata.csv")
        idx_max = len(data_tmp)
        data_tmp['image_path'] = data_tmp.apply(lambda x: os.path.join(datadir, subfolder, x['image_path']), axis=1)

        # train
        train_df = pd.concat([train_df, data_tmp[data_tmp['image_path'].str.contains('train', case=False, na=False)]])

        # val/test
        test_df = pd.concat([test_df, data_tmp[data_tmp['image_path'].str.contains('val', case=False, na=False)]])
        test_df = pd.concat([test_df, data_tmp[data_tmp['image_path'].str.contains('test', case=False, na=False)]])

        # split the rest
        ## train
        remain_df = data_tmp[~data_tmp['image_path'].str.contains('train|test|val', case=False, na=False)]

        train_df = pd.concat([train_df, remain_df.sample(frac=1, random_state=42).reset_index(drop=True).head(int(0.8*idx_max))])
        ## test/val
        test_df =  pd.concat([test_df, remain_df.sample(frac=1, random_state=42).reset_index(drop=True).tail(int(0.2*idx_max))])

    
    
    real_train = train_df[train_df['target'] == 0]
    fake_train = train_df[train_df['target'] > 0]
    real_test = test_df[test_df['target'] == 0]
    fake_test = test_df[test_df['target'] > 0]

    real_train_balanced = real_train.sample(n=len(fake_train), random_state=42)
    real_test_balanced = real_test.sample(n=len(fake_test), random_state=42)

    train_df_balanced = pd.concat([real_train_balanced, fake_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df_balanced = pd.concat([real_test_balanced, fake_test]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Train: {len(train_df_balanced[train_df_balanced.target == 0])} real images and {len(train_df_balanced[train_df_balanced.target > 0])} fake images.")
    print(f"Test: {len(test_df_balanced[test_df_balanced.target == 0])} real images and {len(test_df_balanced[test_df_balanced.target > 0])} fake images.")

    return train_df_balanced, test_df_balanced

class FullyGenDB(Dataset):
    def __init__(self, datadir, mode = 'train'):
        self.mode = mode
        self.datadir = datadir
        self.df = load_dataset(datadir = self.datadir)[0] if self.mode == 'train' else load_dataset(datadir=self.datadir)[1]
        self.normalize = True
    
    def _transforms(self):
        transforms_list = [transforms.ToTensor()]
        if self.normalize:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_desc = self.df.iloc[idx]
        filesize = os.path.getsize(img_desc.image_path)
        img = PIL.Image.open(img_desc.image_path).convert('RGB')
        img = img.resize((256, 256))
        W, H = img.size
        bpp = filesize * 8. / (H * W)

        transformed = self._transforms()(img)
        label = img_desc.target > 0


        return transformed, bpp, label

