import argparse, os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL  import Image
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeepMetricLearning











def recall_at_k(embeddings, labels, k):
    dists = np.linalg.norm(embeddings[:, None] - embeddings[None], axis=2)
    np.fill_diagonal(dists, np.inf)
    hits = 0
    for i in range(len(labels)):
        top_k = np.argsort(dists[i])[:k]
        if labels[i] in labels[top_k]:
            hits += 1
    return hits / len(labels)


def show_retrieval(query_indices, embeddings, labels, image_paths, idx_to_class, save_path, k=5):
    distances = np.linalg.norm(embeddings[:, None] - embeddings[None], axis=2)
    np.fill_diagonal(distances, np.inf)

    fig, axes = plt.subplots(len(query_indices), k+1, figsize=(2*(k+1), 2*len(query_indices)))
    for row, query_idx in enumerate(query_indices):
        top_neighbors = np.argsort(distances[query_idx])[:k]
        axes[row, 0].imshow(Image.open(image_paths[query_idx]).convert('RGB').resize((128, 128)))
        axes[row, 0].set_title(idx_to_class[int(labels[query_idx])], fontsize=6)
        axes[row, 0].axis('off')
        for col, neighbor_idx in enumerate(top_neighbors):
            axes[row, col+1].imshow(Image.open(image_paths[neighbor_idx]).convert('RGB').resize((128, 128)))
            color = 'green' if labels[neighbor_idx] == labels[query_idx] else 'red'
            axes[row, col+1].set_title(idx_to_class[int(labels[neighbor_idx])], fontsize=6, color=color)
            axes[row, col+1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
    out_dir = os.path.dirname(args.model_path)
    out_dir=os.path.join(out_dir, 'graphs')
    print(f'Output directory: {out_dir}')
    device  =  'cpu'

    model = DeepMetricLearning(output_dim=128).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    cache  = torch.load('feature_cache/test.pt', map_location=device)
    with torch.no_grad():
        embs = model.project(cache['features'].to(device)).cpu().numpy()
    labels  = cache['labels'].numpy()
    image_paths = cache['image_paths']


    r1 = recall_at_k(embs, labels, k=1)
    r5 = recall_at_k(embs, labels, k=5)
    print(f'Recall@1={r1:.4f}   Recall@5={r5:.4f}')

    plot_tsne(embs, labels,
              save_path=os.path.join(out_dir, f'tsne.png'))

    idx_to_class  = {int(labels[i]): os.path.basename(os.path.dirname(image_paths[i])) # to show the image class on the retrival 
                     for i in range(len(labels))}
    query_indices = [indices[0] for indices in cache['c2i'].values()][5:10]
    show_retrieval(query_indices, embs, labels, image_paths, idx_to_class,
                   save_path=os.path.join(out_dir, f'retrieval.png'))


if __name__ == '__main__':
    main()
