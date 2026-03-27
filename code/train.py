import argparse, os, sys
import time
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model   import DeepMetricLearning
from loss    import ContrastiveLoss, TripletLoss
from dataset import CachedTripletDataset, CachedContrastiveDataset, PKDataset


def batch_hard_mining_and_loss(embeddings, labels, margin=0.2):
    dist  = torch.cdist(embeddings, embeddings, p=2)
    total = torch.tensor(0.0, device=embeddings.device)
    n     = 0
    for i in range(len(labels)):
        same     = labels == labels[i] # boolean mask for pos
        same[i]  = False               # not the anchor itself
        diff     = ~(labels == labels[i]) # mask for neg ! was not working so used ~
        if same.sum() == 0 or diff.sum() == 0:
            continue
        hard_pos  = dist[i][same].max()
        hard_neg  = dist[i][diff].min()
        total    += torch.clamp(hard_pos - hard_neg + margin, min=0)
        n += 1
    return total / n


def run_epoch(model, loader, loss_fn, optimizer, device, mode, training=True):
    model.train() if training else model.eval()
    total = 0.0
    ctx   = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            if training:
                optimizer.zero_grad()
            if mode == 'contrastive':
                i1, i2, lbl = [x.to(device) for x in batch]
                loss = loss_fn(model.project(i1), model.project(i2), lbl)

            elif mode == 'triplet':
                a, p, n = [x.to(device) for x in batch]
                loss = loss_fn(model.project(a), model.project(p), model.project(n))

            else:
                features     = batch[0].squeeze(0).to(device)
                batch_labels = batch[1].squeeze(0).to(device)
                loss = batch_hard_mining_and_loss(model.project(features), batch_labels)

            if training:
                loss.backward()
                optimizer.step()
            total += loss.item()

    return total / len(loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--option',       default='1',  choices=['1', '2', '3'])
    p.add_argument('--epochs',     type=int,   default=20)
    args = p.parse_args()
    if(args.option=='1'):
        args.mode = 'contrastive'
    elif(args.option=='2'):
        args.mode = 'triplet'
    else:
        args.mode='hard'
    save_dir= f"weights/{args.mode}_{time.time()}" # folder with time to hanle multiple runs 
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'graphs'),  exist_ok=True)

    batch_size=32
    lr=1e-4
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepMetricLearning(output_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.mode == 'contrastive':
        loss_fn      = ContrastiveLoss()
        train_loader = DataLoader(CachedContrastiveDataset( 'feature_cache/train.pt'),batch_size=batch_size, shuffle=True)
        
        val_loader   = DataLoader(  CachedContrastiveDataset('feature_cache/val.pt'), batch_size=batch_size, shuffle=False)
    elif args.mode == 'triplet':
        loss_fn      = TripletLoss()

        train_loader = DataLoader(CachedTripletDataset('feature_cache/train.pt'), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(CachedTripletDataset('feature_cache/val.pt'),    batch_size=batch_size, shuffle=False)
    else:
        loss_fn      = None
        val_loss_fn  = TripletLoss()
        train_loader = DataLoader(PKDataset('feature_cache/train.pt', p=8, k=8),
                                  batch_size=1, shuffle=True)
        val_loader   = DataLoader(CachedTripletDataset('feature_cache/val.pt'),
                                  batch_size=batch_size, shuffle=False)
        
    tr_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        tr_loss  = run_epoch(model, train_loader, loss_fn,optimizer, device, args.mode, training=True)
        val_loss = run_epoch(model, val_loader,   loss_fn if args.mode != 'hard' else val_loss_fn, optimizer, device, args.mode if args.mode != 'hard' else 'triplet', training=False)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f'epoch {epoch}/{args.epochs}   train={tr_loss:.4f}   val={val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_{args.mode}.pt'))

    torch.save(model.state_dict(), os.path.join(save_dir, f'last_{args.mode}.pt'))

    # loss curve
    plt.figure()
    plt.plot(tr_losses,  label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('epoch');  plt.ylabel('loss')
    plt.title(f'{args.mode}');  plt.legend()
    plt.savefig(os.path.join(save_dir, f'graphs/loss.png'),bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
