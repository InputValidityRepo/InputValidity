import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
import sys
sys.path.append('./util')
from ylib.dataloader.tinyimages_80mn_loader import TinyImages
from ylib.dataloader.imagenet_loader import ImageNet

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]],
    #                     [x/255.0 for x in [63.0, 62.1, 66.7]]),
])

transform_train = transforms.Compose([
    # transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomResizedCrop(size=imagesize, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize([x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                      [x / 255.0 for x in [63.0, 62.1, 66.7]]),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}
num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'imagenet': 1000, 'MNIST': 10}

def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
        "eval": {
            'transform_train': transform_test,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_test_largescale,
        },
    })[config_type]

    train_loader, val_loader, lr_schedule, num_classes = None, None, [50, 75, 90], 0
    if args.in_dataset == "CIFAR-10":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(root='./datasets/data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root='./datasets/data', train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(root='./datasets/data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root='./datasets/data', train=False, download=True, transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)

    elif args.in_dataset == 'MNIST':
        if 'train' in split:
            trainset = torchvision.datasets.MNIST(root='/home/jzhang2297/anomaly/data/', train=True,
                                                      transform=transforms.ToTensor(), download=False)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.MNIST(root='/home/jzhang2297/anomaly/data/', train=False, transform=transforms.ToTensor(), download=False)
            val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=config.batch_size, shuffle=True, **kwargs)

    elif args.in_dataset == "imagenet":
        root = args.imagenet_root
        # Data loading code
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)

    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes_dict[args.in_dataset],
    })

def get_loader_out(args, dataset=('tim', 'noise'), config_type='default', split=('train', 'val')):

    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': args.batch_size
        },
    })[config_type]
    train_ood_loader, val_ood_loader = None, None

    if 'train' in split:
        if dataset[0].lower() == 'imagenet':
            train_ood_loader = torch.utils.data.DataLoader(
                ImageNet(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif dataset[0].lower() == 'tim':
            train_ood_loader = torch.utils.data.DataLoader(
                TinyImages(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)

    if 'val' in split:
        val_dataset = dataset[1]  #dataset=(None, ood_dataset)
        print('val dataset is', val_dataset)
        batch_size = args.batch_size
        if args.in_dataset in {'imagenet'}:
            imagesize = 224
        elif args.in_dataset in {'MNIST'}:
            imagesize = 28
        else:
            imagesize = 32
        if True:
            import imageio
            if val_dataset == 'DX_occl': #MNIST
                imgs_list = []
                gen_img_folder = './occl_test_suite'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'DX_light': #MNIST
                imgs_list = []
                gen_img_folder = './light_test_suite'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'DX_light_SVDD': #MNIST
                imgs_list = []
                gen_img_folder = './DX_SVDD0/light/0_0.1/'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img / 255  # preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'DX_blackout': #MNIST
                imgs_list = []
                gen_img_folder = './blackout_test_suite'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'DX_black_DAIV': #MNIST
                imgs_list = []
                gen_img_folder = './DX_DAIV0/blackout/-2708.34_0.01'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'SINVAD':
                import numpy as np
                imgs_list = []
                gen_img_file = './SINVAD/bound_imgs_MNIST.npy'
                gen_imgs = np.load(gen_img_file)[:100]  # we only take 100 test inputs
                for img in gen_imgs:
                    uint_img = img.reshape(28, 28) * 255
                    proc_img = uint_img.astype(np.float64) / 255.
                    imgs_list.append(proc_img)
            elif val_dataset == 'DX_light_SVDD_drive':
                imgs_list = []
                gen_img_folder = './DX_SVDD_driving0/light/0.1'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                img_paths.sort()
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'DX_blackout_SVDD_drive':
                imgs_list = []
                gen_img_folder = './DX_SVDD_driving0/blackout/0.1'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                img_paths.sort()
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'DX_occl_SVDD_drive':
                imgs_list = []
                gen_img_folder = './DX_SVDD_driving0/occl/0.1'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                img_paths.sort()
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)
            elif val_dataset == 'DLFuzz_SVDD_drive':
                imgs_list = []
                gen_img_folder = './DLFuzz/Udacity/test_inputs_DLFuzz_SVDD/Udacity_0.1'
                img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
                img_paths.sort()
                for img_path in img_paths:
                    full_path = gen_img_folder + '/' + img_path
                    img = imageio.imread(full_path)  # img 0~255
                    img = img/255 #preprocessing
                    imgs_list.append(img)

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        #"val_ood_loader": val_ood_loader,
        "val_ood_loader": imgs_list,
    })
