import argparse, os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL  import Image
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model   import DeepMetricLearning
from dataset import baseDatasetCalss











def plot_tsne(embeddings, labels, save_path):
    reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=6)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()




def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--mode',       default='triplet')
    args = p.parse_args()
    data='data/caltech-101'
    out_dir = os.path.dirname(args.model_path)
    device  =  'cpu'

    model = DeepMetricLearning(output_dim=128).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    cache  = torch.load('feature_cache/test.pt', map_location=device)
    with torch.no_grad():
        embs = model.project(cache['features'].to(device)).cpu().numpy()
    labels  = cache['labels'].numpy()
    te_base = baseDatasetCalss(data, split='test')

   

    plot_tsne(embs, labels,
              save_path=os.path.join(out_dir, f'tsne_{args.mode}.png'))


if __name__ == '__main__':
    main()
