"""V3: original features + Wikidata + image histogram + ResNet. LGB only, 3 seeds."""
import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
import lightgbm as lgb

LABEL_MAP = {0:'acrylic',1:'ink',2:'oil on canvas',3:'oil on panel',
             4:'oil on wood',5:'print',6:'tempera',7:'watercolor'}
INV = {v:k for k,v in LABEL_MAP.items()}
N_CLASS = 8
CAP_PAT = re.compile(r'A\s+([a-zA-Z ]+?)\s+artwork titled', re.IGNORECASE)
MED_KW = ['oil','acrylic','tempera','watercolor','gouache','pencil','graphite',
          'pen','ink','chalk','wash','charcoal','etching','woodcut','lithograph',
          'engraving','brushstroke','painted','layered','impasto','etched','plate',
          'engraved','printed','transparent','paper','gold ground','gilded',
          'egg tempera','calligraphic']
TEXT_FIELDS = ['t','txt','tag','note','inscription','creditline','assistivetext']
CAT_COLS    = ['classification','visualbrowserclassification','departmentabbr',
               'viewtype','element','dimensiontype']
RARE_FIELDS = ['volume','eff','subclassification','watermarks','markings',
               'portfolio','series','note','tag','txt']

def extract_cap(s):
    if not isinstance(s,str): return None
    m = CAP_PAT.search(s)
    return m.group(1).strip().lower() if m else None

def parse_dim(s):
    if not isinstance(s,str): return (np.nan, np.nan)
    m = re.search(r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)', s)
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)

def engineer_features(df, top_cat=None):
    df = df.copy()
    df['cap_med'] = df['cap'].apply(extract_cap)
    df['cap_y_int'] = df['cap_med'].map(INV).fillna(-1).astype(int)
    for k,v in LABEL_MAP.items():
        df[f'cap_is_{v.replace(" ","_")}'] = (df['cap_med']==v).astype(int)
    df['y0_f'] = df['y0'].astype(float)
    df['y1_f'] = df['y1'].astype(float)
    df['year_mid']  = (df['y0_f']+df['y1_f'])/2
    df['year_span'] = df['y1_f']-df['y0_f']
    df['century']   = (df['y0_f']//100).fillna(-1).astype(int)
    df['is_pre_1500']  = (df['y0_f']<1500).astype(int)
    df['is_1500_1700'] = ((df['y0_f']>=1500)&(df['y0_f']<1700)).astype(int)
    df['is_1700_1850'] = ((df['y0_f']>=1700)&(df['y0_f']<1850)).astype(int)
    df['is_1850_1940'] = ((df['y0_f']>=1850)&(df['y0_f']<1940)).astype(int)
    df['is_post_1940'] = (df['y0_f']>=1940).astype(int)
    df['anachronism_acrylic'] = ((df['cap_med']=='acrylic')&(df['y0_f']<1940)).astype(int)
    dim = df['dim'].fillna('').str.lower()
    df['has_frame']    = dim.str.contains('framed').astype(int)
    df['has_sheet']    = dim.str.contains('sheet').astype(int)
    df['has_image_dim']= dim.str.contains('image:').astype(int)
    df['has_plate_dim']= dim.str.contains('plate:').astype(int)
    df['has_mount']    = dim.str.contains('mount').astype(int)
    df['n_dim_lines']  = df['dim'].fillna('').str.count(r'\r\n')+1
    pd_dim = df['dim'].apply(parse_dim)
    df['dim_w_cm'] = pd_dim.apply(lambda t: t[0])
    df['dim_h_cm'] = pd_dim.apply(lambda t: t[1])
    df['dim_aspect'] = df['dim_w_cm']/df['dim_h_cm']
    df['dim_area']   = df['dim_w_cm']*df['dim_h_cm']
    df['is_tiny']    = (df['dim_area']<100).astype(int)
    df['is_huge']    = (df['dim_area']>10000).astype(int)
    df['px_aspect']  = df['width']/df['height']
    df['log_px_area']= np.log1p(df['width']*df['height'])
    df['acc_year_n'] = pd.to_numeric(df['acc_id'].fillna('').str[:4], errors='coerce')
    def acc_gift(s):
        if not isinstance(s,str): return np.nan
        p = s.split('.')
        try: return float(p[1]) if len(p)>1 else np.nan
        except: return np.nan
    df['acc_gift_num'] = df['acc_id'].apply(acc_gift)
    for f in TEXT_FIELDS:
        df[f'has_{f}'] = df[f].notna().astype(int)
    concat_text = df[TEXT_FIELDS].fillna('').agg(' '.join, axis=1).str.lower()
    for kw in MED_KW:
        df['kw_'+re.sub(r'\W+','_',kw)] = concat_text.str.contains(re.escape(kw), regex=True).astype(int)
    df['n_rare_fields_present'] = 0
    for f in RARE_FIELDS:
        has = df[f].notna().astype(int)
        df[f'has_{f}'] = has
        df['n_rare_fields_present'] += has
    cat_s = df['cat'].fillna('').astype(str)
    if top_cat is None:
        all_tok = cat_s.str.split('|').explode().str.strip()
        top_cat = all_tok[all_tok.astype(bool)].value_counts().head(30).index.tolist()
    for tok in top_cat:
        df['catk_'+re.sub(r'\W+','_',tok)[:30]] = cat_s.str.contains(re.escape(tok), regex=True).astype(int)
    for c in ['Painting','Drawing','Print','Photograph']:
        df[f'cls_is_{c}'] = (df['classification']==c).astype(int)
    for c in CAT_COLS + ['attribution']:
        df[c] = df[c].fillna('MISSING').astype(str)
    at = df['assistivetext'].fillna('').str.lower()
    df['at_painting'] = at.str.contains('painting').astype(int)
    df['at_drawing']  = at.str.contains('drawing').astype(int)
    df['at_etching']  = at.str.contains('etching').astype(int)
    df['at_ink']      = at.str.contains(r'\bink\b', regex=True).astype(int)
    df['_concat_text'] = concat_text
    return df, top_cat

def oof_artist_encode(tr_df, te_df, y, skf):
    attrs_tr = tr_df['attribution'].values
    attrs_te = te_df['attribution'].values
    oof = np.zeros((len(tr_df), N_CLASS))
    prior = np.bincount(y, minlength=N_CLASS)/len(y)
    for tri,vai in skf.split(tr_df, y):
        counts = pd.DataFrame({'attr':attrs_tr[tri],'y':y[tri]})
        grp = counts.groupby('attr')['y'].value_counts(normalize=True).unstack(fill_value=0)
        for c in range(N_CLASS):
            if c not in grp.columns: grp[c] = 0.0
        grp = grp[list(range(N_CLASS))]
        enc = pd.Series(attrs_tr[vai]).map(lambda a: grp.loc[a].values if a in grp.index else prior)
        oof[vai] = np.vstack(enc.values)
    counts = pd.DataFrame({'attr':attrs_tr,'y':y})
    grp = counts.groupby('attr')['y'].value_counts(normalize=True).unstack(fill_value=0)
    for c in range(N_CLASS):
        if c not in grp.columns: grp[c] = 0.0
    grp = grp[list(range(N_CLASS))]
    te_enc = np.vstack(pd.Series(attrs_te).map(lambda a: grp.loc[a].values if a in grp.index else prior).values)
    return oof, te_enc

def run_v3(seed=42):
    print(f'\n==== V3 SEED {seed} ====')
    train = pd.read_csv('train_n (1).csv')
    test  = pd.read_csv('test_n (1).csv')
    y = train['y'].values

    # external feature tables
    img_df = pd.read_csv('image_features.csv')
    rn_df  = pd.read_csv('resnet_features.csv')
    wd_df  = pd.read_csv('wikidata_features.csv')

    tr_feat, top_cat = engineer_features(train)
    te_feat, _       = engineer_features(test, top_cat=top_cat)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # artist OOF
    art_oof_tr, art_enc_te = oof_artist_encode(tr_feat, te_feat, y, skf)
    for c in range(N_CLASS):
        tr_feat[f'artist_enc_{c}'] = art_oof_tr[:,c]
        te_feat[f'artist_enc_{c}'] = art_enc_te[:,c]

    # merge external features
    for name, df_ext in [('img', img_df), ('rn', rn_df), ('wd', wd_df)]:
        ext_cols = [c for c in df_ext.columns if c not in ['id','wd_materials']]
        n_train  = len(train)
        tr_ext   = df_ext[df_ext['id'].isin(train['id'])].set_index('id').reindex(train['id'])
        te_ext   = df_ext[df_ext['id'].isin(test['id'])].set_index('id').reindex(test['id'])
        for c in ext_cols:
            tr_feat[c] = tr_ext[c].values
            te_feat[c] = te_ext[c].values

    drop = {'y','_concat_text','cap','cap_med','dim','provenancetext','inscription',
            'markings','creditline','acc_id','id','Unnamed: 0','t','txt','tag','note',
            'assistivetext','wikidataid','customprinturl','label','uuid','iiifurl',
            'iiifthumburl','dt','ts','loc','parentid','isvirtual',
            'lastdetectedmodification','created','modified','depictstmsobjectid',
            'sequence','maxpixels','attributioninverted','subclassification',
            'portfolio','series','volume','watermarks','dimension','tp','ord','eff',
            'img','attribution','cat','y0','y1','acc','wd_materials'}
    all_cols = [c for c in tr_feat.columns if c not in drop]
    cat_native = CAT_COLS + ['attribution']
    num_cols = [c for c in all_cols if c not in cat_native]

    for c in all_cols:
        if c not in te_feat.columns: te_feat[c] = 0
    X  = tr_feat[all_cols].copy()
    Xt = te_feat[all_cols].copy()
    for c in num_cols:
        X[c]  = pd.to_numeric(X[c], errors='coerce')
        Xt[c] = pd.to_numeric(Xt[c], errors='coerce')

    # TFIDF
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=3000, sublinear_tf=True)
    svd = TruncatedSVD(n_components=50, random_state=seed)
    Xtr_tfidf = svd.fit_transform(vec.fit_transform(tr_feat['_concat_text']))
    Xte_tfidf = svd.transform(vec.transform(te_feat['_concat_text']))
    tcols = [f'svd_{i}' for i in range(50)]
    X[tcols]  = Xtr_tfidf
    Xt[tcols] = Xte_tfidf

    print(f'Total features: {X.shape[1]}')

    # LGB
    cat_cols_present = [c for c in CAT_COLS if c in X.columns]
    Xlgb  = X.copy()
    Xtlgb = Xt.copy()
    for c in cat_cols_present:
        Xlgb[c]  = Xlgb[c].astype('category')
        Xtlgb[c] = pd.Categorical(Xtlgb[c], categories=Xlgb[c].cat.categories)

    lgb_params = dict(
        objective='multiclass', num_class=N_CLASS, metric='multi_logloss',
        learning_rate=0.015, num_leaves=127, max_depth=9,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=8, lambda_l1=0.05, lambda_l2=0.05,
        verbose=-1, seed=seed
    )

    oof_lgb  = np.zeros((len(X), N_CLASS))
    test_lgb = np.zeros((len(Xt), N_CLASS))

    for fold,(tri,vai) in enumerate(skf.split(Xlgb,y)):
        dtr = lgb.Dataset(Xlgb.iloc[tri], y[tri], categorical_feature=cat_cols_present)
        dva = lgb.Dataset(Xlgb.iloc[vai], y[vai], categorical_feature=cat_cols_present)
        m = lgb.train(lgb_params, dtr, num_boost_round=3000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_lgb[vai] = m.predict(Xlgb.iloc[vai])
        test_lgb += m.predict(Xtlgb)/5
        print(f'  fold {fold+1} best: {m.best_iteration} rounds')

    acc = accuracy_score(y, oof_lgb.argmax(1))
    print(f'LGB OOF acc: {acc:.4f}')

    cap_pat = re.compile(r'A\s+([a-zA-Z ]+?)\s+artwork titled', re.IGNORECASE)
    tr_feat['cap_y'] = tr_feat['cap_med'].map(INV)
    dis_idx = tr_feat[tr_feat['cap_y']!=tr_feat['cap_med'].map(INV)].index
    dis_mask = tr_feat['cap_y'] != pd.Series(y, index=tr_feat.index).map(LABEL_MAP)
    dis_idx2 = tr_feat[dis_mask].index if hasattr(dis_mask,'index') else []
    dis_rows = tr_feat['cap_med'].map(INV).fillna(-1) != pd.Series(y)
    print(f'  acc on {dis_rows.sum()} disagreement rows: {accuracy_score(y[dis_rows], oof_lgb[dis_rows].argmax(1)):.4f}')

    return oof_lgb, test_lgb, y

if __name__ == '__main__':
    all_oof  = []
    all_test = []
    y_ref    = None

    for seed in [42, 7, 123]:
        oof, test_p, y = run_v3(seed=seed)
        all_oof.append(oof)
        all_test.append(test_p)
        if y_ref is None: y_ref = y

    blend_oof  = np.mean(all_oof,  axis=0)
    blend_test = np.mean(all_test, axis=0)

    print(f'\n=== FINAL ===')
    print(f'3-seed blend OOF: {accuracy_score(y_ref, blend_oof.argmax(1)):.4f}')

    test_df = pd.read_csv('test_n (1).csv')
    sub = pd.DataFrame({'id': test_df['id'], 'y': blend_test.argmax(1)})
    sub.to_csv('submission_v3.csv', index=False)
    np.savez('preds_v3.npz', blend_oof=blend_oof, blend_test=blend_test, y=y_ref)
    print('Saved submission_v3.csv')
    print(sub['y'].value_counts().sort_index())
