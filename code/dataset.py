import os, random, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# got mean frrom : 
# https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
mean = [0.485, 0.456, 0.406] 
std  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((128, 128)),
    T.ColorJitter(0.2, 0.2, 0.2), # for  change 20%  up and down ,  bright , contrsst abd saturation
    T.ToTensor(),
    T.Normalize(mean, std)
])

eval_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean, std)
])


class baseDatasetCalss(Dataset):
    def __init__(self, root, split='train', transform=None, seed=42):
        self.transform = transform
        self.samples = []    # (path, label_idx)
        self.labels  = []
        self.class_to_idx = {} 
        self.class_to_samples = {}   # label_idx -> list of paths

        classes = sorted(os.listdir(root)) # as we know folder name is our class name 
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            folder = os.path.join(root, cls)
            imgs = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
            rng = random.Random(seed + idx)   # shuffle all images with fix seed
            rng.shuffle(imgs) 
            # split dat
            n = len(imgs)
            n_train = int(0.70 * n)
            n_val   = int(0.15 * n)

            if split == 'train':
                data = imgs[:n_train]
            elif split == 'val':
                data = imgs[n_train:n_train + n_val]
            else:
                data = imgs[n_train + n_val:] # rest is test
 
            self.class_to_samples[idx] =  data
            for p in data :
                self.samples.append((p, idx))
                self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_samples(self, label_idx):
        return self.class_to_samples[label_idx]

    def num_classes(self):
        return len(self.class_to_idx)


class ContrastiveDataset(Dataset):
    # gives (img1, img2, label)  label=1 same class, label=0 different
    def __init__(self, dataset):
        self.ds = dataset
        self.all_labels = list(self.ds.class_to_samples.keys())

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        positive_pair = random.random() < 0.5
        path1, label1 = self.ds.samples[index]
      
        if positive_pair:
            pool = self.ds.class_to_samples[label1]
            path2 = random.choice([p for p in pool if p != path1])
            label = 1
        else:
            neg_class = random.choice([l for l in self.all_labels if l != label1]) # 2nd class for neh img
            path2 = random.choice(self.ds.class_to_samples[neg_class])    # get image in that class
            label = 0


        img1 = Image.open(path1).convert('RGB')
        if self.ds.transform:
            img1 = self.ds.transform(img1)
        img2 = Image.open(path2).convert('RGB')
        if self.ds.transform:
            img2 = self.ds.transform(img2)

        return img1, img2, label


class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.ds = dataset
        self.all_labels = list(self.ds.class_to_samples.keys())

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index): # get pos negetives  , all images as a anchor once approch 
        path_a, label_a = self.ds.samples[index]
        positive_all = self.ds.class_to_samples[label_a]
        path_positive = random.choice([p for p in positive_all if p != path_a]) 
    

        neg_class = random.choice([l for l in self.all_labels if l != label_a])
        path_negative       = random.choice(self.ds.class_to_samples[neg_class])

        def load(p):
            img = Image.open(p).convert('RGB')
            if self.ds.transform:
                img = self.ds.transform(img)
            return img

        return load(path_a), load(path_positive), load(path_negative)

# get all loaders wrt to the model
def get_dataloaders(root, mode='triplet', batch_size=32, num_workers=2):
    tr_base  = baseDatasetCalss(root, split='train', transform=train_transform)
    val_base = baseDatasetCalss(root, split='val',   transform=eval_transform)
    te_base  = baseDatasetCalss(root, split='test',  transform=eval_transform)

    if mode == 'contrastive':
        train_dataset = ContrastiveDataset(tr_base)
        val_dataset = ContrastiveDataset(val_base)
        test_dataset = ContrastiveDataset(te_base)
    else:
        train_dataset = TripletDataset(tr_base)
        val_dataset = TripletDataset(val_base)

        test_dataset = TripletDataset(te_base)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, tr_base.num_classes()


class CachedTripletDataset(Dataset):
    def __init__(self, pt_path):
        data                   = torch.load(pt_path, map_location='cpu')
        self.features          = data['features']
        self.labels            = data['labels']
        self.class_indices  = data['c2i']
        self.all_classes       = list(self.class_indices.keys())

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        anchor_label  = self.labels[idx].item()
        positive_pool      = [i for i in self.class_indices[anchor_label] if i != idx] or self.class_indices[anchor_label]
        idx_positive  = random.choice(positive_pool)
        neg_class     = random.choice([c for c in self.all_classes if c != anchor_label])
        idx_negative  = random.choice(self.class_indices[neg_class])
        return self.features[idx], self.features[idx_positive], self.features[idx_negative]


class CachedContrastiveDataset(Dataset):
    def __init__(self, pt_path):
        data                   = torch.load(pt_path, map_location='cpu')
        self.features          = data['features']
        self.labels            = data['labels']
        self.class_indices  = data['c2i']
        self.all_classes       = list(self.class_indices.keys())

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        anchor_label = self.labels[idx].item()
        if random.random() < 0.5:
            positive_pool = [i for i in self.class_indices[anchor_label] if i != idx] or self.class_indices[anchor_label]
            return self.features[idx], self.features[random.choice(positive_pool)], 1
        neg_class = random.choice([c for c in self.all_classes if c != anchor_label])
        return self.features[idx], self.features[random.choice(self.class_indices[neg_class])], 0
