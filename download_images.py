"""Download all thumbnails for train+test and extract image features."""
import pandas as pd, numpy as np, requests, os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings; warnings.filterwarnings('ignore')

IMG_DIR = 'thumbs'
os.makedirs(IMG_DIR, exist_ok=True)

def fetch_img(row):
    uid, url = row
    path = f'{IMG_DIR}/{uid}.jpg'
    if os.path.exists(path): return uid, True
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            with open(path, 'wb') as f: f.write(r.content)
            return uid, True
    except: pass
    return uid, False

def download_all(df):
    rows = list(zip(df['id'].astype(str), df['iiifthumburl']))
    ok = 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(fetch_img, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs)):
            uid, success = fut.result()
            if success: ok += 1
            if (i+1) % 500 == 0: print(f'  {i+1}/{len(rows)} done, {ok} ok')
    print(f'Downloaded {ok}/{len(rows)}')

N_IMG_FEATS = 43

def extract_features(uid):
    path = f'{IMG_DIR}/{uid}.jpg'
    if not os.path.exists(path):
        return [np.nan]*N_IMG_FEATS
    try:
        img = Image.open(path).convert('RGB').resize((64,64))
        arr = np.array(img, dtype=np.float32)
        feats = []
        # Per-channel mean/std (6)
        for c in range(3):
            feats.append(arr[:,:,c].mean()/255)
            feats.append(arr[:,:,c].std()/255)
        arr_f = arr/255.0
        r,g,b = arr_f[:,:,0], arr_f[:,:,1], arr_f[:,:,2]
        cmax = np.maximum(np.maximum(r,g),b)
        cmin = np.minimum(np.minimum(r,g),b)
        brightness  = cmax.mean()
        saturation  = ((cmax-cmin)/(cmax+1e-7)).mean()
        gray = 0.299*r + 0.587*g + 0.114*b
        feats.append(brightness)                   # 7
        feats.append(saturation)                   # 8
        feats.append(float(gray.std()))            # 9  contrast
        feats.append(float((gray < 0.1).mean()))   # 10 black_frac
        feats.append(float((gray > 0.9).mean()))   # 11 white_frac
        feats.append(float((r-b).mean()))          # 12 warmth
        # Color histogram 8-bin per channel = 24 (13-36)
        for c in range(3):
            hist, _ = np.histogram(arr[:,:,c], bins=8, range=(0,256), density=True)
            feats.extend(hist.tolist())
        # Grayscale percentiles (37-42)
        feats.append(float(gray.mean()))
        for p in [10,25,50,75,90]:
            feats.append(float(np.percentile(gray,p)))
        # Edge density via simple Sobel-like (43)
        gy = np.abs(np.diff(gray, axis=0)).mean()
        gx = np.abs(np.diff(gray, axis=1)).mean()
        feats.append(float((gy+gx)/2))
        assert len(feats) == N_IMG_FEATS, f'got {len(feats)}'
        return feats
    except:
        return [np.nan]*N_IMG_FEATS

if __name__ == '__main__':
    train = pd.read_csv('train_n (1).csv')
    test  = pd.read_csv('test_n (1).csv')
    all_df = pd.concat([train, test], ignore_index=True)
    print(f'Downloading {len(all_df)} images...')
    download_all(all_df)
    print('Extracting features...')
    feats = []
    for uid in all_df['id'].astype(str):
        feats.append(extract_features(uid))
    cols = ['img_r_mean','img_r_std','img_g_mean','img_g_std','img_b_mean','img_b_std',
            'img_brightness','img_saturation','img_contrast','img_black_frac','img_white_frac',
            'img_warmth'] + \
           [f'img_r_h{i}' for i in range(8)] + \
           [f'img_g_h{i}' for i in range(8)] + \
           [f'img_b_h{i}' for i in range(8)] + \
           ['img_gray_mean','img_gray_p10','img_gray_p25','img_gray_p50',
            'img_gray_p75','img_gray_p90','img_edge_mean']
    feat_df = pd.DataFrame(feats, columns=cols)
    feat_df['id'] = all_df['id'].values
    feat_df.to_csv('image_features.csv', index=False)
    print('Saved image_features.csv')
    print(feat_df.head())
    print('null count:', feat_df.isna().any(axis=1).sum())
