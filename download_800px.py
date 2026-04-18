"""Download 800px images and extract ResNet-50 features at full res via MPS."""
import pandas as pd, numpy as np, requests, os, torch, torchvision
from torchvision import transforms
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import TruncatedSVD
import warnings; warnings.filterwarnings('ignore')

IMG_DIR = 'thumbs_800'
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {DEVICE}')

def fetch(row):
    uid, url = row
    path = f'{IMG_DIR}/{uid}.jpg'
    if os.path.exists(path): return uid, True
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            open(path,'wb').write(r.content)
            return uid, True
    except: pass
    return uid, False

def download_all(df):
    rows = list(zip(df['id'].astype(str), df['img']))
    ok = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs)):
            _, s = fut.result()
            if s: ok += 1
            if (i+1) % 500 == 0: print(f'  {i+1}/{len(rows)} ok={ok}')
    print(f'Downloaded {ok}/{len(rows)}')

# ResNet-50 extractor
model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_batch(uids):
    imgs, valid = [], []
    for i, uid in enumerate(uids):
        p = f'{IMG_DIR}/{uid}.jpg'
        try:
            imgs.append(transform(Image.open(p).convert('RGB')))
            valid.append(i)
        except: pass
    if not imgs: return np.zeros((len(uids), 2048))
    with torch.no_grad():
        feats = extractor(torch.stack(imgs).to(DEVICE)).squeeze(-1).squeeze(-1).cpu().numpy()
    out = np.zeros((len(uids), 2048))
    for j, i in enumerate(valid): out[i] = feats[j]
    return out

if __name__ == '__main__':
    train = pd.read_csv('train_n (1).csv')
    test  = pd.read_csv('test_n (1).csv')
    all_df = pd.concat([train, test], ignore_index=True)
    print(f'Downloading {len(all_df)} 800px images...')
    download_all(all_df)

    all_ids = all_df['id'].astype(str).tolist()
    BS = 64
    all_feats = []
    print('Extracting ResNet50 at 800px...')
    for i in range(0, len(all_ids), BS):
        all_feats.append(extract_batch(all_ids[i:i+BS]))
        if (i//BS+1) % 10 == 0: print(f'  {i+BS}/{len(all_ids)}')
    feats = np.vstack(all_feats)
    print(f'Raw shape: {feats.shape}')

    # normalize before SVD to avoid overflow
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1
    feats_n = feats / norms

    svd = TruncatedSVD(n_components=128, random_state=42)
    feats_r = svd.fit_transform(feats_n)
    print(f'SVD explained var: {svd.explained_variance_ratio_.sum():.3f}')

    cols = [f'rn800_{i}' for i in range(128)]
    df_out = pd.DataFrame(feats_r, columns=cols)
    df_out['id'] = all_df['id'].values
    df_out.to_csv('resnet800_features.csv', index=False)
    print('Saved resnet800_features.csv')
