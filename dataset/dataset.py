import os, glob
import random
import numpy as np
import paddle
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from paddle.io import Dataset
from pathlib import Path
from paddle.vision import transforms
import paddle.vision.transforms.functional as TF
# from RandomCrop_ import RandomCrop
def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    return img


def default_loader_crop(path):
    img = Image.open(path).convert('RGB')
    region = img.crop((156, 156, 660, 660))
    return region

class RandomCrop():
    def __init__(self):
        super(RandomCrop).__init__()
    def get_params(self, img, output_size=(128, 128)):
        # 这里的image对象长什么样子
        img_shape = img.shape # w, h = img.size
        if img_shape[2] > output_size[1] :
            left = np.random.randint(img_shape[2] - output_size[1])
        else:
            left = np.random.randint(img_shape[2])
        if img_shape[1] > output_size[0] :
            top = np.random.randint(img_shape[1] - output_size[0])
        else:
            top = np.random.randint(img_shape[1])
        bbox = (top, left, output_size[0], output_size[1])
        return bbox

class Moire_dataset(Dataset):

    def __init__(self, root, loader=default_loader):
        moire_data_root = os.path.join(root, 'images')
        clear_data_root = os.path.join(root, 'gts')
        image_names = os.listdir(clear_data_root)
        image_names = ['_'.join(i.split('_')[:-1]) for i in image_names]
        self.moire_images = [os.path.join(moire_data_root, x +
            '_source.png') for x in image_names]
        self.clear_images = [os.path.join(clear_data_root, x +
            '_target.png') for x in image_names]
        self.transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)
        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)


# class My_Moire_dataset(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])

#         self.transforms2 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms3 = transforms.Compose([transforms.Resize((32, 32))])
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         if self.crop == True:
#             random_crop = RandomCrop()
#             i, j, h, w = random_crop.get_params(moire,
#                 output_size=(128, 128))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear_list = [clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_test(Dataset):

#     def __init__(self, root, crop=False, loader=default_loader):
#         self.max_len = 100
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         moire_images = [os.path.join(moire_data_root, x) for x in image_names1]
#         self.moire_images = moire_images[1:self.max_len]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         clear_images = [os.path.join(clear_data_root, x) for x in image_names2]
#         self.clear_images = clear_images[1:self.max_len]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.Resize((1024, 1024
#             )), transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((512, 512))])
#         self.transforms3 = transforms.Compose([transforms.Resize((256, 256))])
#         self.loader = loader
#         self.labels = image_names1[1:self.max_len]

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear_list = [clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_1(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms3 = transforms.Compose([transforms.Resize((32, 32))])
#         self.transforms4 = transforms.Compose([transforms.Resize((16, 16))])
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         if self.crop == True:
#             random_crop_ = RandomCrop()
#             i, j, h, w = random_crop_.get_params(moire,
#                 output_size=(128, 128))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_RR_1(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms3 = transforms.Compose([transforms.Resize((32, 32))])
#         self.transforms4 = transforms.Compose([transforms.Resize((16, 16))])
#         self.Resize = transforms.Resize((128, 128))
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         if self.crop == True:
#             max = 200
#             min = 100
#             if max > moire.shape[1]:
#                 max = moire.shape[1]
#             if max > moire.shape[2]:
#                 max = moire.shape[2]
#             else:
#                 max = max
#             size = random.randint(min, max)
#             random_crop = RandomCrop()
#             i, j, h, w = random_crop.get_params(moire,
#                 output_size=(size, size))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#             moire = self.Resize(moire)
#             clear = self.Resize(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_RR_2(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((128, 128))])
#         self.transforms3 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms4 = transforms.Compose([transforms.Resize((32, 32))])
#         self.size = 256
#         self.Resize = transforms.Resize((self.size, self.size))
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         if self.crop == True:
#             max = int(self.size * 2)
#             min = int(self.size * 0.25)
#             if max > moire.shape[1]:
#                 max = moire.shape[1]
#             if max > moire.shape[2]:
#                 max = moire.shape[2]
#             else:
#                 max = max
#             size = random.randint(min, max)
#             random_crop = RandomCrop()
#             i, j, h, w = random_crop.get_params(moire,
#                                                 output_size=(size, size))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#             moire = self.Resize(moire)
#             clear = self.Resize(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


class My_Moire_dataset_RR_2_4(Dataset):

    def __init__(self, root, crop=True, loader=default_loader):
        self.crop = crop
        # 数据集
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clean')

        image_names1 = []
        for file_name in os.listdir(moire_data_root):
            if file_name.endswith('jpg') or file_name.endswith('JPEG'):
                image_names1.append(file_name)
            else:
                pass
        print(len(image_names1))
        image_names1.sort()
        image_names1 = image_names1[:-6]
        self.moire_images = [os.path.join(moire_data_root, x) for x in image_names1]

        image_names2 = []
        for file_name in os.listdir(clear_data_root):
            if file_name.endswith('jpg') or file_name.endswith('JPEG'):
                image_names2.append(file_name)
            else:
                pass
        print(len(image_names2))
        image_names2.sort()
        image_names2 = image_names2[:-20]
        self.clear_images = [os.path.join(clear_data_root, x) for x in image_names2]

        image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.transforms2 = transforms.Compose([transforms.Resize((128, 128))])
        self.transforms3 = transforms.Compose([transforms.Resize((64, 64))])
        self.transforms4 = transforms.Compose([transforms.Resize((32, 32))])
        self.size = 256
        self.Resize = transforms.Resize((self.size, self.size))
        self.loader = loader
        self.labels = image_names1

        print('there are ',len(self.moire_images), ' moire iamges and ', len(self.clear_images), ' clear images.')

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = np.asarray(moire)
        clear = np.asarray(clear)
# print('line 341',type(moire))
        moire = self.transforms(moire)
        clear = self.transforms(clear)
# print('line 344',type(moire))
        if self.crop == True and clear.shape[1] != 256:
            max = int(self.size * 2)
            min = int(self.size * 0.25)
            if max > moire.shape[1]:
                max = moire.shape[1]
            if max > moire.shape[2]:
                max = moire.shape[2]
            else:
                max = max
            size = random.randint(min, max)
            random_crop = RandomCrop()
            i, j, h, w = random_crop.get_params(moire, output_size=(size, size))
            moire = TF.crop(moire, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)
            moire = self.Resize(moire)
            clear = self.Resize(clear)
            if random.random() > 0.1:
                angle = random.randint(-20, 20)
                moire = moire.transpose([1,2,0])
                moire = moire.numpy()
                clear = clear.transpose([1,2,0])
                clear = clear.numpy()
# print('line 369', type(moire))
                moire = TF.rotate(moire, angle)
                clear = TF.rotate(clear, angle)
                moire = self.transforms(moire)
                clear = self.transforms(clear)
        else:
            pass
        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear2)
        clear4 = self.transforms4(clear3)
        clear_list = [clear4, clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)


# class My_Moire_dataset_RR_2_3(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((128, 128))])
#         self.transforms3 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms4 = transforms.Compose([transforms.Resize((32, 32))])
#         self.size = 256
#         self.Resize = transforms.Resize((self.size, self.size))
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         if self.crop == True:
#             max = int(self.size * 2)
#             min = int(self.size * 0.25)
#             if max > moire.shape[1]:
#                 max = moire.shape[1]
#             if max > moire.shape[2]:
#                 max = moire.shape[2]
#             else:
#                 max = max
#             size = random.randint(min, max)
#             random_crop = RandomCrop()
#             i, j, h, w = random_crop.get_params(moire,
#                                                 output_size=(size, size))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#             moire = self.Resize(moire)
#             clear = self.Resize(clear)
#             if random.random() > 0.1:
#                 angle = random.randint(-20, 20)
#                 moire = TF.rotate(moire, angle)
#                 clear = TF.rotate(clear, angle)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_RR_2_1(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((128, 128))])
#         self.transforms3 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms4 = transforms.Compose([transforms.Resize((32, 32))])
#         self.size = 256 // 2
#         self.Resize = transforms.Resize((self.size * 2, self.size * 2))
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         joint_moire = []
#         joint_clear = []
#         for _ in range(3):
#             joint_clear.append(clear)
#             joint_moire.append(moire)
#         if self.crop == True:
#             max = int(self.size * 2)
#             min = int(self.size * 0.25)
#             if max > moire.shape[1]:
#                 max = moire.shape[1]
#             if max > moire.shape[2]:
#                 max = moire.shape[2]
#             else:
#                 max = max
#             size = random.randint(min, max)
#             moire, clear = self.my_rand_crop(moire, clear, size)
#             for i in range(3):
#                 joint_moire[i], joint_clear[i] = self.my_rand_crop(joint_moire
#                     [i], joint_clear[i], size)
#             moire = self.joint_image(moire, joint_moire, size)
#             clear = self.joint_image(clear, joint_clear, size)
#             moire = self.Resize(moire)
#             clear = self.Resize(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def joint_image(self, image, image_list, size):
#         joint_img = paddle.zeros([3, size * 2, size * 2]).requires_grad_(False)
#         joint_img[:, :size, :size] = image
#         joint_img[:, size:, :size] = image_list[0]
#         joint_img[:, size:, size:] = image_list[1]
#         joint_img[:, :size, :size] = image_list[2]
#         return joint_img

#     def my_rand_crop(self, moire, clear, size):
#         random_crop = RandomCrop()
#         i, j, h, w = random_crop.get_params(moire,
#                                             output_size=(size, size))
#         moire = TF.crop(moire, i, j, h, w)
#         clear = TF.crop(clear, i, j, h, w)
#         return moire, clear

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_RR_2_2(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((128, 128))])
#         self.transforms3 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms4 = transforms.Compose([transforms.Resize((32, 32))])
#         self.size = 256 // 2
#         self.Resize = transforms.Resize((self.size * 2, self.size * 2))
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         joint_moire = []
#         joint_clear = []
#         for _ in range(3):
#             dex = random.randint(0, len(self.moire_images) - 1)
#             joint_clear.append(self.transforms(self.loader(self.
#                 clear_images[dex])))
#             joint_moire.append(self.transforms(self.loader(self.
#                 moire_images[dex])))
#         if self.crop == True:
#             max = int(self.size * 2)
#             min = int(self.size * 0.25)
#             if max > moire.shape[1]:
#                 max = moire.shape[1]
#             if max > moire.shape[2]:
#                 max = moire.shape[2]
#             else:
#                 max = max
#             size = random.randint(min, max)
#             moire, clear = self.my_rand_crop(moire, clear, size)
#             for i in range(3):
#                 joint_moire[i], joint_clear[i] = self.my_rand_crop(joint_moire
#                     [i], joint_clear[i], size)
#             moire = self.joint_image(moire, joint_moire, size)
#             clear = self.joint_image(clear, joint_clear, size)
#             moire = self.Resize(moire)
#             clear = self.Resize(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def joint_image(self, image, image_list, size):
#         joint_img = paddle.zeros([3, size * 2, size * 2]).requires_grad_(False)
#         joint_img[3, :size, :size] = image
#         joint_img[3, size:, :size] = image_list[0]
#         joint_img[3, size:, size:] = image_list[1]
#         joint_img[3, :size, :size] = image_list[2]
#         return joint_img

#     def my_rand_crop(self, moire, clear, size):
#         random_crop = RandomCrop()
#         i, j, h, w = random_crop.get_params(moire,
#                                             output_size=(size, size))
#         moire = TF.crop(moire, i, j, h, w)
#         clear = TF.crop(clear, i, j, h, w)
#         return moire, clear

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_RR_3(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((256, 256))])
#         self.transforms3 = transforms.Compose([transforms.Resize((128, 128))])
#         self.transforms4 = transforms.Compose([transforms.Resize((64, 64))])
#         self.size = 512
#         self.Resize = transforms.Resize((self.size, self.size))
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         if self.crop == True:
#             max = int(self.size * 1.5)
#             min = int(self.size * 0.5)
#             if max > moire.shape[1]:
#                 max = moire.shape[1]
#             if max > moire.shape[2]:
#                 max = moire.shape[2]
#             else:
#                 max = max
#             size = random.randint(min, max)
#             random_crop = RandomCrop()
#             i, j, h, w = random_crop.get_params(moire,
#                                                 output_size=(size, size))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#             moire = self.Resize(moire)
#             clear = self.Resize(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class My_Moire_dataset_test_1(Dataset):

#     def __init__(self, root, crop=False, loader=default_loader):
#         self.max_len = 100
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         moire_images = [os.path.join(moire_data_root, x) for x in image_names1]
#         self.moire_images = moire_images[1:self.max_len]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         clear_images = [os.path.join(clear_data_root, x) for x in image_names2]
#         self.clear_images = clear_images[1:self.max_len]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.Resize((1024, 1024
#             )), transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((512, 512))])
#         self.transforms3 = transforms.Compose([transforms.Resize((256, 256))])
#         self.transforms4 = transforms.Compose([transforms.Resize((128, 128))])
#         self.loader = loader
#         self.labels = image_names1[1:self.max_len]

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear4 = self.transforms4(clear3)
#         clear_list = [clear4, clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


class My_Moire_dataset_test_val_1(Dataset):

    def __init__(self, root, crop=False, loader=default_loader):
        self.max_len = 100
        self.crop = crop
        moire_data_root = os.path.join(root, 'images')
        clear_data_root = os.path.join(root, 'gts')
        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        moire_images = [os.path.join(moire_data_root, x) for x in image_names1]
        self.moire_images = moire_images[1:self.max_len]
        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        clear_images = [os.path.join(clear_data_root, x) for x in image_names2]
        self.clear_images = clear_images[1:self.max_len]
        image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.loader = loader
        self.labels = image_names1[1:self.max_len]

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = np.asarray(moire)
        clear = np.asarray(clear)
        moire = self.transforms(moire)
        clear = self.transforms(clear)
        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)


class My_Moire_dataset_test_mode_1(Dataset):

    def __init__(self, root, crop=False, loader=default_loader):
        self.crop = crop
        moire_data_root = os.path.join(root, 'images')
        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        moire_images = [os.path.join(moire_data_root, x) for x in image_names1]
        self.moire_images = moire_images
        image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = np.asarray(moire)
        moire = self.transforms(moire)
        label = self.labels[index]
        return moire, label

    def __len__(self):
        return len(self.moire_images)


# class AIMMoire_dataset(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms3 = transforms.Compose([transforms.Resize((32, 32))])
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         if self.crop == True:
#             random_crop = RandomCrop()
#             i, j, h, w = random_crop.get_params(moire,
#                 output_size=(128, 128))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear_list = [clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class AIMMoire_dataset_test(Dataset):

#     def __init__(self, root, crop=False, loader=default_loader):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'moire')
#         clear_data_root = os.path.join(root, 'clean')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((512, 512))])
#         self.transforms3 = transforms.Compose([transforms.Resize((256, 256))])
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear_list = [clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


# class TIP2018moire_dataset_train(Dataset):

#     def __init__(self, root, crop=True, loader=default_loader_crop):
#         self.crop = crop
#         moire_data_root = os.path.join(root, 'source')
#         clear_data_root = os.path.join(root, 'target')
#         image_names1 = os.listdir(moire_data_root)
#         image_names1.sort()
#         self.moire_images = [os.path.join(moire_data_root, x) for x in
#             image_names1]
#         image_names2 = os.listdir(clear_data_root)
#         image_names2.sort()
#         self.clear_images = [os.path.join(clear_data_root, x) for x in
#             image_names2]
#         image_names1 = ['.'.join(i.split('.')[:-1]) for i in image_names1]
#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.transforms2 = transforms.Compose([transforms.Resize((64, 64))])
#         self.transforms3 = transforms.Compose([transforms.Resize((32, 32))])
#         self.loader = loader
#         self.labels = image_names1

#     def __getitem__(self, index):
#         moire_img_path = self.moire_images[index]
#         clear_img_path = self.clear_images[index]
#         moire = self.loader(moire_img_path)
#         clear = self.loader(clear_img_path)
#         moire = self.transforms(moire)
#         clear = self.transforms(clear)
#         h, w = clear.shape[-2:]
#         if self.crop == True:
#             random_crop = RandomCrop()
#             i, j, h, w = random_crop.get_params(moire,
#                 output_size=(128, 128))
#             moire = TF.crop(moire, i, j, h, w)
#             clear = TF.crop(clear, i, j, h, w)
#         clear2 = self.transforms2(clear)
#         clear3 = self.transforms3(clear2)
#         clear_list = [clear3, clear2, clear]
#         label = self.labels[index]
#         return moire, clear_list, label

#     def __len__(self):
#         return len(self.moire_images)


def random_scale_for_pair(moire, clear, mask, is_val=False):
    if is_val == False:
        is_global = np.random.randint(0, 2)
        if is_global == 0:
            resize = transforms.Resize((256, 256))
            moire, clear, mask = resize(moire), resize(clear), resize(mask)
        else:
            resize = transforms.Resize((286, 286))
            moire, clear, mask = resize(moire), resize(clear), resize(mask)
            random_x = np.random.randint(0, moire.size[0] - 256)
            random_y = np.random.randint(0, moire.size[1] - 256)
            moire = moire.crop((random_x, random_y, random_x + 256, 
                random_y + 256))
            clear = clear.crop((random_x, random_y, random_x + 256, 
                random_y + 256))
            mask = mask.crop((random_x, random_y, random_x + 256, random_y +
                256))
        is_flip = np.random.randint(0, 2)
        if is_flip == 0:
            moire = moire.transpose(Image.FLIP_LEFT_RIGHT)
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            pass
    else:
        resize = transforms.Resize((256, 256))
        moire, clear, mask = resize(moire), resize(clear), resize(mask)
    return moire, clear, mask


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[
                1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[
                1] / y), order=0)
        image = paddle.to_tensor(image.astype(np.float32)).unsqueeze(0)
        label = paddle.to_tensor(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):

    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')
            ).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + '/{}.npy.h5'.format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class Synapse_dataset_te(Dataset):

    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')
            ).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + '/{}.npy.h5'.format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
