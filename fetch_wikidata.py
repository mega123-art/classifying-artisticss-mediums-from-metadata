"""Batch fetch P186 (material used) from Wikidata for all rows."""
import pandas as pd, numpy as np, requests, time, json, re
from collections import defaultdict

SPARQL = 'https://query.wikidata.org/sparql'
HEADERS = {'User-Agent': 'ArtMediumClassifier/1.0 (kaggle competition research)'}
BATCH = 80  # IDs per query

MATERIAL_MAP = {
    # oil signals
    'oil paint': 'oil', 'oil on canvas': 'oil_canvas', 'oil on panel': 'oil_panel',
    'oil on wood': 'oil_wood', 'linseed oil': 'oil',
    # canvas / support
    'canvas': 'canvas_support', 'transferred to canvas': 'canvas_support',
    'transferred on canvas': 'canvas_support',
    # panel / wood supports
    'panel': 'panel_support', 'wood': 'wood_support', 'poplar': 'wood_support',
    'oak': 'wood_support', 'walnut': 'wood_support', 'copper': 'metal_support',
    # watercolor
    'watercolor paint': 'watercolor', 'watercolour': 'watercolor',
    'gouache paint': 'watercolor', 'gouache': 'watercolor',
    'watercolor': 'watercolor',
    # ink
    'ink': 'ink_mat', 'india ink': 'ink_mat', 'chinese ink': 'ink_mat',
    'black ink': 'ink_mat', 'iron gall ink': 'ink_mat',
    'pen': 'ink_mat', 'quill': 'ink_mat',
    # print/etching
    'etching': 'print_mat', 'engraving': 'print_mat', 'lithography': 'print_mat',
    'woodcut': 'print_mat', 'aquatint': 'print_mat', 'drypoint': 'print_mat',
    'intaglio': 'print_mat', 'mezzotint': 'print_mat', 'screenprint': 'print_mat',
    'linocut': 'print_mat', 'lithograph': 'print_mat',
    # tempera / egg tempera
    'tempera': 'tempera_mat', 'egg tempera': 'tempera_mat', 'distemper': 'tempera_mat',
    # acrylic
    'acrylic paint': 'acrylic_mat', 'acrylic': 'acrylic_mat',
    # paper supports
    'paper': 'paper_support', 'wove paper': 'paper_support', 'laid paper': 'paper_support',
    'vellum': 'paper_support', 'parchment': 'paper_support',
    # chalk / graphite / pencil (often with ink or watercolor)
    'graphite': 'graphite', 'chalk': 'chalk', 'black chalk': 'chalk',
    'red chalk': 'chalk', 'pencil': 'graphite',
    'charcoal': 'charcoal',
    # gold leaf (often tempera)
    'gold leaf': 'gold', 'gold': 'gold',
}

def fetch_batch(wids):
    vals = ' '.join(f'wd:{w}' for w in wids)
    q = f"""
SELECT ?item ?matLabel WHERE {{
  VALUES ?item {{ {vals} }}
  ?item wdt:P186 ?mat.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
    for attempt in range(3):
        try:
            r = requests.get(SPARQL, params={'query':q,'format':'json'},
                            headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()['results']['bindings']
            elif r.status_code == 429:
                time.sleep(5 * (attempt+1))
        except Exception as e:
            time.sleep(3)
    return []

def materials_to_features(mat_list):
    cats = defaultdict(int)
    for m in mat_list:
        m_low = m.lower().strip()
        for key, cat in MATERIAL_MAP.items():
            if key in m_low:
                cats[cat] = 1
    return cats

def wikidata_to_pred(cats):
    """Heuristic: map material categories to y class."""
    if not cats: return -1
    # Strong signals
    if cats.get('watercolor'): return 7
    if cats.get('acrylic_mat'): return 0
    if cats.get('tempera_mat') or cats.get('gold'): return 6
    if cats.get('print_mat'): return 5
    # Oil on support
    if cats.get('oil') or cats.get('oil_canvas') or cats.get('oil_panel') or cats.get('oil_wood'):
        if cats.get('canvas_support'): return 2
        if cats.get('panel_support'): return 3
        if cats.get('wood_support'): return 4
        if cats.get('oil_canvas'): return 2
        if cats.get('oil_panel'): return 3
        if cats.get('oil_wood'): return 4
        return 2  # default to canvas
    # Ink
    if cats.get('ink_mat'): return 1
    return -1

if __name__ == '__main__':
    train = pd.read_csv('train_n (1).csv')
    test  = pd.read_csv('test_n (1).csv')
    all_df = pd.concat([train[['id','wikidataid']], test[['id','wikidataid']]], ignore_index=True)
    valid = all_df[all_df['wikidataid'].notna()]
    wids = valid['wikidataid'].tolist()
    ids  = valid['id'].tolist()
    print(f'Fetching Wikidata P186 for {len(wids)} items in batches of {BATCH}...')

    results = {}
    for i in range(0, len(wids), BATCH):
        batch_wids = wids[i:i+BATCH]
        batch_ids  = ids[i:i+BATCH]
        bindings = fetch_batch(batch_wids)
        # group by item
        item_mats = defaultdict(list)
        for b in bindings:
            qid = b['item']['value'].split('/')[-1]
            mat = b['matLabel']['value']
            item_mats[qid].append(mat)
        for wid, rid in zip(batch_wids, batch_ids):
            results[rid] = item_mats.get(wid, [])
        if (i//BATCH + 1) % 5 == 0:
            print(f'  batch {i//BATCH+1}/{(len(wids)+BATCH-1)//BATCH}')
        time.sleep(0.6)

    print(f'Fetched {len(results)} items')
    # Build feature df
    rows = []
    all_ids = all_df['id'].tolist()
    for rid in all_ids:
        mats = results.get(rid, [])
        cats = materials_to_features(mats)
        pred = wikidata_to_pred(cats)
        row = {'id': rid, 'wd_pred': pred, 'wd_materials': '|'.join(mats)}
        for cat in ['oil','canvas_support','panel_support','wood_support',
                    'watercolor','acrylic_mat','tempera_mat','print_mat','ink_mat',
                    'paper_support','graphite','chalk','gold','charcoal']:
            row[f'wd_{cat}'] = cats.get(cat, 0)
        rows.append(row)
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv('wikidata_features.csv', index=False)
    print('Saved wikidata_features.csv')
    # Verify on train
    tr_merge = train.merge(feat_df, on='id')
    has_pred = tr_merge[tr_merge['wd_pred']!=-1]
    acc = (has_pred['wd_pred']==has_pred['y']).mean()
    print(f'Wikidata heuristic accuracy (on rows with pred): {acc:.4f}  ({len(has_pred)}/{len(train)} rows)')
    # On disagreement rows
    cap_pat = re.compile(r'A\s+([a-zA-Z ]+?)\s+artwork titled', re.IGNORECASE)
    INV = {'acrylic':0,'ink':1,'oil on canvas':2,'oil on panel':3,'oil on wood':4,'print':5,'tempera':6,'watercolor':7}
    tr_merge['cap_med'] = tr_merge['cap'].apply(lambda s: (cap_pat.search(s).group(1).strip().lower() if isinstance(s,str) and cap_pat.search(s) else None))
    tr_merge['cap_y'] = tr_merge['cap_med'].map(INV)
    dis = tr_merge[tr_merge['cap_y']!=tr_merge['y']]
    dis_has = dis[dis['wd_pred']!=-1]
    if len(dis_has):
        acc_dis = (dis_has['wd_pred']==dis_has['y']).mean()
        print(f'Wikidata acc on {len(dis_has)}/{len(dis)} disagreement rows: {acc_dis:.4f}')
