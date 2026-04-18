"""Extract ResNet-50 features from thumbnails using MPS (M4)."""
import torch, torchvision, numpy as np, pandas as pd
from torchvision import transforms
from PIL import Image
import os

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
model.eval()
# remove final classifier, keep avgpool → 2048-dim
extractor = torch.nn.Sequential(*list(model.children())[:-1])
extractor = extractor.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_batch(paths, bs=64):
    imgs = []
    valid_idx = []
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img))
            valid_idx.append(i)
        except:
            pass
    if not imgs:
        return np.zeros((len(paths), 2048))
    batch = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        feats = extractor(batch).squeeze(-1).squeeze(-1).cpu().numpy()
    out = np.zeros((len(paths), 2048))
    for j, i in enumerate(valid_idx):
        out[i] = feats[j]
    return out

if __name__ == '__main__':
    train = pd.read_csv('train_n (1).csv')
    test  = pd.read_csv('test_n (1).csv')
    all_ids = list(pd.concat([train['id'], test['id']]).astype(str))
    paths = [f'thumbs/{uid}.jpg' for uid in all_ids]
    BS = 128
    all_feats = []
    print(f'Extracting ResNet50 from {len(paths)} images...')
    for i in range(0, len(paths), BS):
        batch = paths[i:i+BS]
        feats = extract_batch(batch, bs=BS)
        all_feats.append(feats)
        if (i//BS+1) % 5 == 0:
            print(f'  {i+BS}/{len(paths)}')
    feats = np.vstack(all_feats)
    print(f'Feature shape: {feats.shape}')

    # Reduce to 100 dims with SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, random_state=42)
    feats_r = svd.fit_transform(feats)
    print(f'Explained var ratio sum: {svd.explained_variance_ratio_.sum():.3f}')

    cols = [f'rn_{i}' for i in range(100)]
    df_out = pd.DataFrame(feats_r, columns=cols)
    df_out['id'] = all_ids
    df_out.to_csv('resnet_features.csv', index=False)
    print('Saved resnet_features.csv')
