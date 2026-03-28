import os, sys, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model   import DeepMetricLearning
from dataset import baseDatasetCalss, eval_transform


def build_cache(backbone, root, split, save_path, device):
    ds     = baseDatasetCalss(root, split=split, transform=eval_transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    backbone.eval()
    all_features = []
    with torch.no_grad():
        for imgs, _ in loader:
            print(f'Processing {len(all_features)}/{len(ds)} samples...', end='\r')
            all_features.append(backbone(imgs.to(device)).flatten(1).cpu())

    features = torch.cat(all_features)
    labels   = torch.tensor(ds.labels)

    class_to_indices = {}
    for idx, (_, label) in enumerate(ds.samples):
        class_to_indices.setdefault(label, []).append(idx)

    image_paths = [path for path, _ in ds.samples]

    torch.save({
        'features':     features,
        'labels':       labels,
        'c2i':          class_to_indices,
        'image_paths':  image_paths,
    }, save_path)
    print(f'{split}: {len(features)} samples -> {save_path}')


if __name__ == '__main__':
    os.makedirs('feature_cache', exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = DeepMetricLearning(output_dim=128).backbone.to(device)

    root = 'data/caltech-101'
    build_cache(backbone, root, 'train', 'feature_cache/train.pt', device)
    build_cache(backbone, root, 'val',   'feature_cache/val.pt',   device)
    build_cache(backbone, root, 'test',  'feature_cache/test.pt',  device)
