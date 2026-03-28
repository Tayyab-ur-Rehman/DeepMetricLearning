import argparse, os, sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model   import DeepMetricLearning
from dataset import eval_transform


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--image_path', required=True)
    args = p.parse_args()

    device = 'cpu'
    model  = DeepMetricLearning(output_dim=128).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    img   = Image.open(args.image_path).convert('RGB')
    with torch.no_grad():
        query_emb = model(eval_transform(img).unsqueeze(0).to(device)).cpu().numpy()

    cache = torch.load('feature_cache/test.pt', map_location=device)
    with torch.no_grad():
        all_embs = model.project(cache['features'].to(device)).cpu().numpy()
    labels      = cache['labels'].numpy()
    image_paths = cache['image_paths']
    idx_to_class = {int(labels[i]): os.path.basename(os.path.dirname(image_paths[i]))
                    for i in range(len(labels))}

    dists = np.linalg.norm(all_embs - query_emb, axis=1)
    for rank, idx in enumerate(np.argsort(dists)[:5], 1):
        print(f'top{rank}  class={idx_to_class[int(labels[idx])]}  dist={dists[idx]:.4f}')


if __name__ == '__main__':
    main()
