"""Full pipeline: features + base models + stacking + multi-seed. Runnable end-to-end."""
import pandas as pd, numpy as np, re, warnings, json, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

LABEL_MAP = {0:'acrylic',1:'ink',2:'oil on canvas',3:'oil on panel',
             4:'oil on wood',5:'print',6:'tempera',7:'watercolor'}
INV = {v:k for k,v in LABEL_MAP.items()}
N_CLASS = 8

CAP_PAT = re.compile(r'A\s+([a-zA-Z ]+?)\s+artwork titled', re.IGNORECASE)

MED_KEYWORDS = ['oil','acrylic','tempera','watercolor','gouache','pencil','graphite',
                'pen','ink','chalk','wash','charcoal','etching','woodcut','lithograph',
                'engraving','brushstroke','painted','layered','impasto','etched','plate',
                'engraved','printed','transparent','paper','gold ground','gilded',
                'egg tempera','calligraphic']

RARE_FIELDS = ['volume','eff','subclassification','watermarks','markings','portfolio',
               'series','note','tag','txt']

TEXT_FIELDS = ['t','txt','tag','note','inscription','creditline','assistivetext']

CAT_COLS_NATIVE = ['classification','visualbrowserclassification','departmentabbr',
                   'attribution','viewtype','element','dimensiontype']

def extract_cap(s):
    if not isinstance(s,str): return None
    m = CAP_PAT.search(s)
    return m.group(1).strip().lower() if m else None

def parse_dim(s):
    """Parse first line of dim string for width×height cm."""
    if not isinstance(s,str): return (np.nan, np.nan)
    m = re.search(r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)', s)
    if m: return (float(m.group(1)), float(m.group(2)))
    return (np.nan, np.nan)

def engineer_features(df):
    df = df.copy()
    # cap medium
    df['cap_med'] = df['cap'].apply(extract_cap)
    df['cap_y'] = df['cap_med'].map(INV).fillna(-1).astype(int)
    for k,v in LABEL_MAP.items():
        df[f'cap_is_{v.replace(" ","_")}'] = (df['cap_med']==v).astype(int)

    # temporal
    df['y0_f'] = df['y0'].astype(float)
    df['y1_f'] = df['y1'].astype(float)
    df['year_mid'] = (df['y0_f']+df['y1_f'])/2
    df['year_span'] = df['y1_f']-df['y0_f']
    df['century'] = (df['y0_f']//100).fillna(-1).astype(int)
    df['is_pre_1500'] = (df['y0_f']<1500).astype(int)
    df['is_1500_1700'] = ((df['y0_f']>=1500)&(df['y0_f']<1700)).astype(int)
    df['is_1700_1850'] = ((df['y0_f']>=1700)&(df['y0_f']<1850)).astype(int)
    df['is_1850_1940'] = ((df['y0_f']>=1850)&(df['y0_f']<1940)).astype(int)
    df['is_post_1940'] = (df['y0_f']>=1940).astype(int)
    df['anachronism_acrylic'] = ((df['cap_med']=='acrylic') & (df['y0_f']<1940)).astype(int)

    # dim text flags
    dim = df['dim'].fillna('').str.lower()
    df['has_frame'] = dim.str.contains('framed').astype(int)
    df['has_sheet'] = dim.str.contains('sheet').astype(int)
    df['has_image_dim'] = dim.str.contains('image:').astype(int)
    df['has_plate_dim'] = dim.str.contains('plate:').astype(int)
    df['has_mount'] = dim.str.contains('mount').astype(int)
    df['n_dim_lines'] = df['dim'].fillna('').str.count(r'\r\n')+1
    # physical dims from first line
    pd_dim = df['dim'].apply(parse_dim)
    df['dim_w_cm'] = pd_dim.apply(lambda t: t[0])
    df['dim_h_cm'] = pd_dim.apply(lambda t: t[1])
    df['dim_aspect'] = df['dim_w_cm']/df['dim_h_cm']
    df['dim_area'] = df['dim_w_cm']*df['dim_h_cm']
    df['is_tiny'] = (df['dim_area']<100).astype(int)
    df['is_huge'] = (df['dim_area']>10000).astype(int)

    # image pixel dims (width/height cols are pixels)
    df['px_aspect'] = df['width']/df['height']
    df['px_area'] = df['width']*df['height']
    df['log_px_area'] = np.log1p(df['px_area'])

    # accession
    df['acc_year'] = df['acc_id'].fillna('').str[:4]
    df['acc_year_n'] = pd.to_numeric(df['acc_year'], errors='coerce')
    def acc_gift(s):
        if not isinstance(s,str): return np.nan
        parts = s.split('.')
        try: return float(parts[1]) if len(parts)>1 else np.nan
        except: return np.nan
    df['acc_gift_num'] = df['acc_id'].apply(acc_gift)

    # text-based keyword flags (on concat of text fields)
    for f in TEXT_FIELDS:
        df[f'has_{f}'] = df[f].notna().astype(int)
    concat_text = df[TEXT_FIELDS].fillna('').agg(' '.join, axis=1).str.lower()
    for kw in MED_KEYWORDS:
        col = 'kw_' + re.sub(r'\W+','_',kw)
        df[col] = concat_text.str.contains(re.escape(kw), regex=True).astype(int)

    # rare-field missingness
    df['n_rare_fields_present'] = 0
    for f in RARE_FIELDS:
        has = df[f].notna().astype(int)
        df[f'has_{f}'] = has
        df['n_rare_fields_present'] += has

    # cat pipe-split top 30 tokens
    cat_series = df['cat'].fillna('').astype(str)
    all_tokens = cat_series.str.split('|').explode().str.strip()
    top_cat = all_tokens[all_tokens.astype(bool)].value_counts().head(30).index.tolist()
    for tok in top_cat:
        col = 'catk_' + re.sub(r'\W+','_',tok)[:30]
        df[col] = cat_series.str.contains(re.escape(tok), regex=True).astype(int)

    # visualbrowser/classification flags
    for c in ['Painting','Drawing','Print','Photograph']:
        df[f'cls_is_{c}'] = (df['classification']==c).astype(int)

    # native categoricals — fill MISSING
    for c in CAT_COLS_NATIVE:
        df[c] = df[c].fillna('MISSING').astype(str)

    df['_concat_text'] = concat_text
    return df, top_cat

def add_top_cat_test(df_test, top_cat):
    """Apply same cat tokens fit on train."""
    cat_series = df_test['cat'].fillna('').astype(str)
    for tok in top_cat:
        col = 'catk_' + re.sub(r'\W+','_',tok)[:30]
        df_test[col] = cat_series.str.contains(re.escape(tok), regex=True).astype(int)
    return df_test

def build_tfidf_svd(train_text, test_text, n_components=50, seed=42):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),
                          max_features=3000, sublinear_tf=True)
    Xtr = vec.fit_transform(train_text)
    Xte = vec.transform(test_text)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Xtr_s = svd.fit_transform(Xtr)
    Xte_s = svd.transform(Xte)
    return Xtr_s, Xte_s, vec, svd

def run_pipeline(seed=42):
    print(f"\n==== SEED {seed} ====")
    train = pd.read_csv('train_n (1).csv')
    test = pd.read_csv('test_n (1).csv')
    y = train['y'].values
    tr_feat, top_cat = engineer_features(train)
    te_feat, _ = engineer_features(test)
    te_feat = add_top_cat_test(te_feat, top_cat)
    # align columns
    feat_cols = [c for c in tr_feat.columns if c not in ['y','_concat_text','cap','cap_med','dim',
        'provenancetext','inscription','markings','creditline','acc_id','id','Unnamed: 0',
        't','txt','tag','note','assistivetext','wikidataid','customprinturl','label',
        'uuid','iiifurl','iiifthumburl','dt','ts','loc','parentid','isvirtual',
        'lastdetectedmodification','created','modified','depictstmsobjectid','sequence',
        'maxpixels','attributioninverted','subclassification','portfolio','series','volume',
        'watermarks','dimension','tp','ord','eff','img','attribution','cat']]
    # keep native cats back
    feat_cols += CAT_COLS_NATIVE
    feat_cols = list(dict.fromkeys(feat_cols))
    # Ensure both have
    for c in feat_cols:
        if c not in te_feat.columns: te_feat[c] = 0
    X = tr_feat[feat_cols].copy()
    Xt = te_feat[feat_cols].copy()

    # TFIDF
    tfidf_tr, tfidf_te, _, _ = build_tfidf_svd(tr_feat['_concat_text'], te_feat['_concat_text'], seed=seed)
    tfidf_cols = [f'svd_{i}' for i in range(tfidf_tr.shape[1])]
    X[tfidf_cols] = tfidf_tr
    Xt[tfidf_cols] = tfidf_te

    # numeric cast for non-cat
    num_cols = [c for c in X.columns if c not in CAT_COLS_NATIVE]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        Xt[c] = pd.to_numeric(Xt[c], errors='coerce')

    print(f"feature count: {len(feat_cols)+len(tfidf_cols)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_lgb = np.zeros((len(X), N_CLASS))
    oof_cat = np.zeros((len(X), N_CLASS))
    oof_lr  = np.zeros((len(X), N_CLASS))
    test_lgb = np.zeros((len(Xt), N_CLASS))
    test_cat = np.zeros((len(Xt), N_CLASS))
    test_lr  = np.zeros((len(Xt), N_CLASS))

    # ---- LightGBM ----
    X_lgb = X.copy()
    Xt_lgb = Xt.copy()
    for c in CAT_COLS_NATIVE:
        X_lgb[c] = X_lgb[c].astype('category')
        Xt_lgb[c] = pd.Categorical(Xt_lgb[c], categories=X_lgb[c].cat.categories)

    lgb_params = dict(objective='multiclass', num_class=N_CLASS, metric='multi_logloss',
                      learning_rate=0.02, num_leaves=63, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=5, verbose=-1, seed=seed)
    for fold,(tri,vai) in enumerate(skf.split(X_lgb,y)):
        dtr = lgb.Dataset(X_lgb.iloc[tri], y[tri], categorical_feature=CAT_COLS_NATIVE)
        dva = lgb.Dataset(X_lgb.iloc[vai], y[vai], categorical_feature=CAT_COLS_NATIVE)
        m = lgb.train(lgb_params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_lgb[vai] = m.predict(X_lgb.iloc[vai])
        test_lgb += m.predict(Xt_lgb)/5
    acc_lgb = accuracy_score(y, oof_lgb.argmax(1))
    print(f"LGB OOF acc: {acc_lgb:.4f}")

    # ---- CatBoost ----
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS_NATIVE]
    X_cat = X.copy()
    Xt_cat = Xt.copy()
    for c in CAT_COLS_NATIVE:
        X_cat[c] = X_cat[c].astype(str).fillna('MISSING')
        Xt_cat[c] = Xt_cat[c].astype(str).fillna('MISSING')
    # fill NaN numerics with large sentinel
    for c in num_cols:
        X_cat[c] = X_cat[c].fillna(-9999)
        Xt_cat[c] = Xt_cat[c].fillna(-9999)
    for fold,(tri,vai) in enumerate(skf.split(X_cat,y)):
        m = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6,
                               loss_function='MultiClass', random_seed=seed,
                               early_stopping_rounds=100, verbose=0,
                               cat_features=cat_idx)
        m.fit(X_cat.iloc[tri], y[tri], eval_set=(X_cat.iloc[vai], y[vai]))
        oof_cat[vai] = m.predict_proba(X_cat.iloc[vai])
        test_cat += m.predict_proba(Xt_cat)/5
    acc_cat = accuracy_score(y, oof_cat.argmax(1))
    print(f"CAT OOF acc: {acc_cat:.4f}")

    # ---- Logistic Regression on TFIDF + cap one-hot ----
    cap_cols = [c for c in X.columns if c.startswith('cap_is_')]
    lr_cols = tfidf_cols + cap_cols
    Xlr = X[lr_cols].fillna(0).values
    Xtlr = Xt[lr_cols].fillna(0).values
    for fold,(tri,vai) in enumerate(skf.split(Xlr,y)):
        m = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0,
                               solver='saga', max_iter=500, n_jobs=-1,
                               multi_class='multinomial', random_state=seed)
        m.fit(Xlr[tri], y[tri])
        oof_lr[vai] = m.predict_proba(Xlr[vai])
        test_lr += m.predict_proba(Xtlr)/5
    acc_lr = accuracy_score(y, oof_lr.argmax(1))
    print(f"LR  OOF acc: {acc_lr:.4f}")

    # ---- Stack ----
    top_meta = ['y0_f','has_frame','has_sheet','has_inscription','dim_area',
                'century','anachronism_acrylic'] + \
               [c for c in X.columns if c.startswith('cap_is_')]
    meta_tr = np.hstack([oof_lgb, oof_cat, oof_lr, X[top_meta].fillna(-9999).values])
    meta_te = np.hstack([test_lgb, test_cat, test_lr, Xt[top_meta].fillna(-9999).values])

    meta_oof = np.zeros((len(X), N_CLASS))
    meta_test = np.zeros((len(Xt), N_CLASS))
    meta_params = dict(objective='multiclass', num_class=N_CLASS, metric='multi_logloss',
                       learning_rate=0.05, num_leaves=15, verbose=-1, seed=seed)
    for fold,(tri,vai) in enumerate(skf.split(meta_tr,y)):
        dtr = lgb.Dataset(meta_tr[tri], y[tri])
        dva = lgb.Dataset(meta_tr[vai], y[vai])
        m = lgb.train(meta_params, dtr, num_boost_round=300, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        meta_oof[vai] = m.predict(meta_tr[vai])
        meta_test += m.predict(meta_te)/5
    acc_meta = accuracy_score(y, meta_oof.argmax(1))
    print(f"STACK OOF acc: {acc_meta:.4f}")

    return dict(oof_lgb=oof_lgb, oof_cat=oof_cat, oof_lr=oof_lr,
                test_lgb=test_lgb, test_cat=test_cat, test_lr=test_lr,
                meta_oof=meta_oof, meta_test=meta_test,
                acc_lgb=acc_lgb, acc_cat=acc_cat, acc_lr=acc_lr, acc_meta=acc_meta,
                y=y)

if __name__=='__main__':
    r = run_pipeline(seed=42)
    # save preds for reuse
    np.savez('preds_seed42.npz', **{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
    print("\nDone.")
