"""Assemble the final .ipynb."""
import json

def md(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
def code(src): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}

cells = []

# ---- Title ----
cells.append(md("""# The Art Historian AI: Forensic Medium Attribution

> *"Every brushstroke leaves a fingerprint — if you know where to look."*

**Competition framing (what we figured out before writing any ML):**

The `cap` column on every row reads *"A &lt;medium&gt; artwork titled '...' by ..."* — a 4-line regex
extracts the medium and scores **94.17%** on the training set with zero learning. The organizers did not
make the label a secret; they made it a **weak label**. **5.8% of rows (233 of 4000) have a cap that
disagrees with `y`.** The real competition is not classification. It is **forensic reattribution**:
training a model that knows when to *override* the curator's tag using physical, temporal, and
material evidence hidden elsewhere in the metadata.

This notebook is structured as an investigation — not an ML pipeline.
"""))

# ---- Section 1 ----
cells.append(md("""## 1. The Opening Investigation

Scene of the crime: 4000 training artworks, 57 columns of mixed metadata, 8 medium classes.
Before touching a model we canvass the evidence: shapes, class balance, and how much of the metadata is
actually populated.
"""))

cells.append(code("""import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_PATH = 'train_n (1).csv'
TEST_PATH  = 'test_n (1) (3).csv'

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

LABEL_MAP = {0:'acrylic',1:'ink',2:'oil on canvas',3:'oil on panel',
             4:'oil on wood',5:'print',6:'tempera',7:'watercolor'}
INV = {v:k for k,v in LABEL_MAP.items()}

print('train shape:', train.shape)
print('test  shape:', test.shape)
print('target y distribution (named):')
print(train['y'].map(LABEL_MAP).value_counts())
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9,4))
counts = train['y'].map(LABEL_MAP).value_counts()
ax.bar(counts.index, counts.values, color='#6b7a8f')
ax.set_title('Class distribution (training set)')
ax.set_ylabel('n rows'); plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""miss = (train.isna().mean()*100).sort_values(ascending=False)
print('Top-15 most-null columns (%):')
print(miss.head(15).round(1))

top_null_cols = miss.head(25).index.tolist()
plt.figure(figsize=(10,5))
sns.heatmap(train[top_null_cols].isna().iloc[::20].T, cbar=False, cmap='Greys')
plt.title('Missingness pattern across the 25 most-null columns')
plt.xlabel('row (every 20th)'); plt.tight_layout(); plt.show()
"""))

cells.append(md("""**Observations**
- Class balance is roughly even (412 – 611 rows per medium) — no need for resampling.
- `volume`, `eff`, `subclassification`, `customprinturl`, `watermarks`, `markings` are >95% empty —
  these columns carry signal mainly through **their absence** (we'll encode that explicitly).
- A small zero-null spine (`classification`, `departmentabbr`, `cap`, `iiifthumburl`, `viewtype`,
  `attribution`) will anchor most of our features.
"""))

# ---- Section 2 ----
cells.append(md("""## 2. The 20-Minute Breakthrough

The `cap` field is disarmingly helpful. Every row is a sentence of the form
`A <medium> artwork titled '...' by <artist>.`

A single regex extracts the medium and gives us a baseline.
"""))

cells.append(code("""CAP_PAT = re.compile(r'A\\s+([a-zA-Z ]+?)\\s+artwork titled', re.IGNORECASE)
def extract_cap(s):
    if not isinstance(s,str): return None
    m = CAP_PAT.search(s)
    return m.group(1).strip().lower() if m else None

train['cap_med'] = train['cap'].apply(extract_cap)
train['cap_y']   = train['cap_med'].map(INV)

acc_regex = (train['cap_y']==train['y']).mean()
disagreements = (train['cap_y']!=train['y']).sum()
print(f'Pure-regex training accuracy: {acc_regex:.4f}')
print(f'Disagreements (cap says X, y says Y): {disagreements} / {len(train)}')
"""))

cells.append(md("""94.17%. A naive submission would stop here. **We got suspicious.**

If the cap were ground truth, why would 233 rows disagree? That is either labelling noise (bad hypothesis
— test labels would be equally noisy and nobody could exceed 94.17%) *or* it is an **intentional test of
whether the model can re-attribute using other evidence**. We assume the latter.
"""))

cells.append(code("""from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
mask = train['cap_y'].notna()
cm = confusion_matrix(train.loc[mask,'y'], train.loc[mask,'cap_y'], labels=list(LABEL_MAP.keys()))
fig, ax = plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(cm, display_labels=list(LABEL_MAP.values())).plot(ax=ax, cmap='Greys', colorbar=False)
plt.title('Regex-only predictions: where the curator and reality diverge')
plt.xticks(rotation=40, ha='right'); plt.tight_layout(); plt.show()

print('Disagreement pattern (cap_y → true y):')
print(pd.crosstab(train.loc[mask,'cap_y'].map(LABEL_MAP),
                  train.loc[mask,'y'].map(LABEL_MAP)))
"""))

cells.append(md("""The error concentration is informative:

| Cap says     | Truth         | Count |
|--------------|---------------|-------|
| ink          | watercolor    | 101   |
| oil on panel | oil on canvas | 37    |
| oil on wood  | oil on canvas | 19    |
| tempera      | oil on canvas | 14    |
| print        | ink           | 12    |

The errors are almost entirely **within visually/physically similar groups** (ink↔watercolor, oil-substrate
mix-ups, tempera↔oil). This is exactly where domain features should dominate: framing,
sheet-vs-canvas terminology, century, paint-era.
"""))

# ---- Section 3 Anachronism ----
cells.append(md("""## 3. The Anachronism Detector

Forensic tell #1: **chemistry has a history.**

Acrylic emulsion paint was patented in 1934 and did not enter widespread artistic use until ~1953
(Liquitex). Any painting dated before ~1940 that the curator tagged `acrylic` is materially impossible —
the paint did not exist. The model should *not* trust the label in that scenario.
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(10,5))
sns.violinplot(x=train['y'].map(LABEL_MAP), y=train['y0'], inner='quartile',
               palette='Set2', ax=ax)
ax.axhline(1940, color='red', ls='--', lw=1, label='acrylic patented (1934) / mainstream (~1940)')
ax.set_title('Start-year distribution per medium')
ax.set_xlabel(''); ax.set_ylabel('y0 (earliest year)'); plt.xticks(rotation=30, ha='right')
ax.legend(); plt.tight_layout(); plt.show()

anach = ((train['cap_med']=='acrylic') & (train['y0']<1940)).sum()
print(f'Rows the cap calls "acrylic" but dated pre-1940: {anach}')
"""))

cells.append(md("""We'll turn this into explicit features: `century`, era-buckets, and `anachronism_acrylic`. The model
learns to downweight the cap when the chemistry doesn't fit the calendar.
"""))

# ---- Section 4 Feature Engineering ----
cells.append(md("""## 4. Feature Engineering — Each Feature Has a Reason

Every feature is grounded in either (a) the regex spine, (b) art-historical domain knowledge, or
(c) structural regularities in the NGA metadata. One function applied identically to train and test —
no train/test skew possible.
"""))

cells.append(code("""from pipeline_v3 import engineer_features, oof_artist_encode, CAT_COLS, N_CLASS
from sklearn.model_selection import StratifiedKFold

y = train['y'].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tr_feat, top_cat = engineer_features(train)
te_feat, _       = engineer_features(test, top_cat=top_cat)

print(f'Feature columns generated: {tr_feat.shape[1]}')
print('\\nhas_frame proportion per class:')
print(tr_feat.groupby('y')['has_frame'].mean().rename(index=LABEL_MAP).round(3))
"""))

cells.append(md("""**Feature families and *why* each exists**

| Family | Count | Rationale |
|---|---|---|
| Weak-label spine | 10 | `cap_is_<medium>` one-hots + `cap_y_int` — the 94% baseline as a feature, not an output |
| Anachronism | 7 | `century`, era-buckets, `anachronism_acrylic` — material chemistry vs date |
| Framing vocabulary | 6 | `has_frame`, `has_sheet`, `has_image_dim`, `has_plate_dim`, `has_mount` — oils framed; prints come as sheets |
| Physical dims | 8 | `dim_w_cm`, `dim_h_cm`, `dim_aspect`, `dim_area`, `is_tiny`, `is_huge` — tempera panels are small |
| Pixel dims | 2 | `px_aspect`, `log_px_area` — image aspect ratio correlates with format |
| Accession archaeology | 2 | `acc_year_n`, `acc_gift_num` — donation batches cluster similar works |
| Medium keywords | 30 | Binary flags from concat of all text fields — curators mention "gouache", "etching", "impasto" |
| Rare-field presence | 9 | `has_volume`, `has_portfolio`, …, `n_rare_fields_present` |
| Category tokens | 30 | Top-30 `cat` tokens — high signal for print/drawing/painting type |
| `assistivetext` flags | 4 | `at_painting`, `at_drawing`, `at_etching`, `at_ink` |
| TF-IDF → SVD50 | 50 | Char-3-to-5 TF-IDF on all text → dense medium vocabulary |
| Artist OOF encoding | 8 | Out-of-fold artist→class probability vector (no leakage) |
| Image histogram | 43 | Brightness, saturation, warmth, RGB histograms, edge density from thumbnails |
| ResNet-50 visual | 100 | Pretrained ResNet-50 avgpool → TruncatedSVD(100) on 200px thumbs |
| Wikidata P186 | 14 | Material-used properties fetched via SPARQL for rows with Wikidata IDs |
| Native categoricals | 6 | `classification`, `departmentabbr`, `viewtype`, `element`, `dimensiontype`, `attribution` |

**Total: ~340 columns.** Zero mean-imputation — NaN is information.
"""))

# ---- Section 5 External Features ----
cells.append(md("""## 5. External Feature Sources

Three external feature tables are precomputed and merged at train time.

### 5a. Image Color Histograms (`image_features.csv`)
43 features extracted from the 200px IIIF thumbnails: per-channel means/stds, brightness, saturation,
contrast, black/white fraction, warmth ratio, RGB histogram bins, grayscale percentiles, and Canny edge
density. These capture the physical *feel* of the medium — oils are warm and saturated; watercolors are
pale and low-contrast; prints have high edge density.

### 5b. ResNet-50 Visual Embeddings (`resnet_features.csv`)
Pretrained ResNet-50 (ImageNet1K) with the final FC layer removed, run on 200px thumbnails via Apple MPS.
The 2048-dim avgpool output is L2-normalized and compressed to 100 dims with TruncatedSVD. These features
capture texture and composition patterns invisible to hand-crafted histograms.

### 5c. Wikidata Materials (`wikidata_features.csv`)
For every row with a `wikidataid`, we query P186 (material used) via SPARQL in batches of 80.
Material labels are mapped to 14 binary category flags (`wd_oil`, `wd_watercolor`, `wd_ink_mat`,
`wd_print_mat`, `wd_tempera_mat`, `wd_acrylic_mat`, etc.) plus a heuristic `wd_pred` label.
Coverage is partial (~60% of rows) but the signal is very strong when present — Wikidata effectively
provides a second expert opinion on the medium.
"""))

# ---- Section 6 Pipeline ----
cells.append(md("""## 6. The Final Pipeline

```
raw CSV ─────────────────────────────────────────────────────▶ engineer_features()
                                                                        │
               ┌────────────────────────────────────────────────────────┤
               │                                                        │
   image_features.csv ──▶ merge on id                                  │
   resnet_features.csv ──▶ merge on id          ───────────────────────▶ ~340 columns
   wikidata_features.csv ─▶ merge on id                                │
               │                                                        │
               └────────────────────────────────────────────────────────┘
                                       │
                    OOF artist target encoding (no leakage)
                                       │
                     TF-IDF char(3-5) → SVD50 on text
                                       │
                              LightGBM (5-fold CV)
                          num_leaves=127, lr=0.015, max_depth=9
                          early_stopping=150, ~2000 rounds
                                       │
                        3-seed average (seeds 42, 7, 123)
                                       │
                               submission_v3.csv
```

Key design decisions:
- **LightGBM only** — CatBoost and LogReg were tested but added noise without improving OOF.
  The stacking meta-learner was also removed: LGB alone at 97.9% OOF beat the 3-model stack.
- **Artist OOF encoding** — attribution has 2000+ unique values. OOF target encoding gives each
  fold a calibrated prior over the 8 classes for each artist, preventing leakage.
- **3 seeds, averaged probabilities** — reduces variance on the ~58 test disagreement rows.
"""))

# ---- Section 7 Training ----
cells.append(md("""## 7. Training (5-fold CV)
"""))

cells.append(code("""from pipeline_v3 import engineer_features, oof_artist_encode, CAT_COLS, N_CLASS
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
import lightgbm as lgb

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
y = train['y'].values

img_df = pd.read_csv('image_features.csv')
rn_df  = pd.read_csv('resnet_features.csv')
wd_df  = pd.read_csv('wikidata_features.csv')

tr_feat, top_cat = engineer_features(train)
te_feat, _       = engineer_features(test, top_cat=top_cat)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

art_oof, art_te = oof_artist_encode(tr_feat, te_feat, y, skf)
for c in range(N_CLASS):
    tr_feat[f'artist_enc_{c}'] = art_oof[:,c]
    te_feat[f'artist_enc_{c}'] = art_te[:,c]

for name, df_ext in [('img',img_df),('rn',rn_df),('wd',wd_df)]:
    ext_cols = [c for c in df_ext.columns if c not in ['id','wd_materials']]
    tr_ext = df_ext.set_index('id').reindex(train['id'])
    te_ext = df_ext.set_index('id').reindex(test['id'])
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
num_cols = [c for c in all_cols if c not in CAT_COLS]

X = tr_feat[all_cols].copy(); Xt = te_feat[all_cols].copy()
for c in all_cols:
    if c not in Xt.columns: Xt[c] = 0
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
    Xt[c] = pd.to_numeric(Xt[c], errors='coerce')

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=3000, sublinear_tf=True)
svd = TruncatedSVD(n_components=50, random_state=42)
X[[f'svd_{i}' for i in range(50)]]  = svd.fit_transform(vec.fit_transform(tr_feat['_concat_text']))
Xt[[f'svd_{i}' for i in range(50)]] = svd.transform(vec.transform(te_feat['_concat_text']))

print(f'Total features: {X.shape[1]}')

cat_cp = [c for c in CAT_COLS if c in X.columns]
Xl = X.copy(); Xtl = Xt.copy()
for c in cat_cp:
    Xl[c] = Xl[c].astype('category')
    Xtl[c] = pd.Categorical(Xtl[c], categories=Xl[c].cat.categories)

lgb_params = dict(
    objective='multiclass', num_class=8, metric='multi_logloss',
    learning_rate=0.015, num_leaves=127, max_depth=9,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=8, lambda_l1=0.05, lambda_l2=0.05,
    verbose=-1, seed=42
)

oof = np.zeros((len(X), 8)); test_preds = np.zeros((len(Xt), 8))
for fold, (tri, vai) in enumerate(skf.split(Xl, y)):
    dtr = lgb.Dataset(Xl.iloc[tri], y[tri], categorical_feature=cat_cp)
    dva = lgb.Dataset(Xl.iloc[vai], y[vai], categorical_feature=cat_cp)
    m = lgb.train(lgb_params, dtr, 3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof[vai] = m.predict(Xl.iloc[vai])
    test_preds += m.predict(Xtl) / 5
    print(f'  fold {fold+1} best={m.best_iteration} rounds')

print(f'\\nOOF accuracy (seed=42): {accuracy_score(y, oof.argmax(1)):.4f}')
"""))

# ---- Section 8 Multi-Seed ----
cells.append(md("""## 8. Multi-Seed Averaging

Three seeds reduce variance on the ~58 disagreement rows in the test set (where the model must
override the cap signal). Each seed samples different fold assignments and feature subsets.
"""))

cells.append(code("""# Load pre-computed 3-seed blend (preds_v3.npz)
preds = np.load('preds_v3.npz')
blend_oof  = preds['blend_oof']
blend_test = preds['blend_test']
y_ref      = preds['y']

from sklearn.metrics import accuracy_score
print(f'3-seed blend OOF accuracy: {accuracy_score(y_ref, blend_oof.argmax(1)):.4f}')
print(f'Public LB accuracy: 0.97111  (971 / 1000 test rows correct)')
print()

# Accuracy on the hard disagreement rows only
import re
CAP_PAT = re.compile(r'A\\s+([a-zA-Z ]+?)\\s+artwork titled', re.IGNORECASE)
def extract_cap(s):
    m = CAP_PAT.search(s) if isinstance(s, str) else None
    return m.group(1).strip().lower() if m else None

train2 = pd.read_csv(TRAIN_PATH)
train2['cap_med'] = train2['cap'].apply(extract_cap)
train2['cap_y'] = train2['cap_med'].map(INV)
dis_mask = (train2['cap_y'] != train2['y']).values
print(f'Disagreement rows: {dis_mask.sum()}')
print(f'OOF accuracy on disagreement rows: {accuracy_score(y_ref[dis_mask], blend_oof[dis_mask].argmax(1)):.4f}')
print(f'OOF accuracy on cap-agreeing rows: {accuracy_score(y_ref[~dis_mask], blend_oof[~dis_mask].argmax(1)):.4f}')
"""))

# ---- Section 9 SHAP ----
cells.append(md("""## 9. Interpretability — SHAP

Why does the model beat 94.17%? SHAP on a single-fold LightGBM shows which features override the cap.
"""))

cells.append(code("""import shap, lightgbm as lgb

X_shap = X.copy()
for c in cat_cp:
    X_shap[c] = X_shap[c].astype('category')

model_shap = lgb.LGBMClassifier(
    objective='multiclass', num_class=8, n_estimators=500,
    learning_rate=0.015, num_leaves=127, random_state=42, verbose=-1
)
model_shap.fit(X_shap, y,
               categorical_feature=cat_cp)

explainer = shap.TreeExplainer(model_shap)
shap_vals = explainer.shap_values(X_shap.iloc[:500])

# Show for watercolor class (index 7) — the hardest to detect when cap says ink
shap.summary_plot(
    shap_vals[7] if isinstance(shap_vals, list) else shap_vals[..., 7],
    X_shap.iloc[:500], plot_type='dot', show=True,
    max_display=15
)
"""))

cells.append(md("""`cap_is_watercolor`, `cap_y_int`, `artist_enc_7`, and Wikidata material flags dominate — exactly
the forensic signals we engineered. The artist OOF encoding appears in the top features because certain
artists (German 16th Century, Mark Rothko) are near-deterministically associated with watercolor
regardless of what the cap says.
"""))

# ---- Section 10 Mislabel Gallery ----
cells.append(md("""## 10. Where the Model Overrides the Curator

High-confidence test predictions that *disagree* with the cap — the model's forensic calls.
"""))

cells.append(code("""import requests
from PIL import Image
from io import BytesIO

CAP_PAT2 = re.compile(r'A\\s+([a-zA-Z ]+?)\\s+artwork titled', re.IGNORECASE)
cap_test = test['cap'].apply(lambda s: CAP_PAT2.search(s).group(1).strip().lower()
                              if isinstance(s,str) and CAP_PAT2.search(s) else None)
pred  = blend_test.argmax(1)
conf  = blend_test.max(1)

overrides = pd.DataFrame({
    'id': test['id'],
    'cap_says': cap_test.map(INV).map(LABEL_MAP),
    'model_says': pd.Series(pred).map(LABEL_MAP),
    'conf': conf,
    'thumb': test['iiifthumburl'],
    'title': test['t']
})
mask = (overrides['cap_says'] != overrides['model_says']) & (overrides['conf'] > 0.85)
gallery = overrides[mask].sort_values('conf', ascending=False).head(5)
print(f'{mask.sum()} high-confidence overrides (conf>0.85) in test; showing top 5.')

fig, axes = plt.subplots(1, min(5, len(gallery)), figsize=(18,5))
if len(gallery) == 1: axes = [axes]
for ax, (_, row) in zip(axes, gallery.iterrows()):
    try:
        img = Image.open(BytesIO(requests.get(row['thumb'], timeout=5).content))
        ax.imshow(img)
    except:
        ax.text(0.5, 0.5, '(image fetch failed)', ha='center', va='center')
    ax.axis('off')
    ax.set_title(f"{str(row['title'])[:25]}\\ncap: {row['cap_says']}\\nmodel: {row['model_says']} ({row['conf']:.2f})",
                 fontsize=9)
plt.tight_layout(); plt.show()
"""))

# ---- Section 11 What We Chose Not To Do ----
cells.append(md("""## 11. What We Tried and What We Chose NOT to Do

### Experiments that did NOT improve the score

- ❌ **CatBoost + LogReg stacking** — V1 pipeline (LGB 97.10% + CatBoost 96.77% + LogReg 94.40% → stack
  96.67%) was *worse* than LGB alone. The stacker added model-averaging noise without new signal. Removed
  in V3.
- ❌ **XGBoost** — Tested in V4 (depth=8, 3000 rounds). Runtime >4h on M4 MPS without meaningful OOF
  improvement. Killed.
- ❌ **800px ResNet-50 features** (`resnet800_features.csv`, 128 SVD dims) — Downloaded and extracted
  800px IIIF images; OOF remained 97.925% identical. Higher resolution did not help distinguish
  ink from watercolor at the pixel level — the signal is in the metadata, not the paint texture.
- ❌ **5-seed blend** — Seeds {42, 7, 123, 13, 99} gave 97.875% OOF, slightly *below* the 3-seed
  97.925%. Extra seeds added noise on the hard cases.
- ❌ **Full retrain on all 4000 rows** (V6) — CV-tuned iteration count × 1.05, trained without validation.
  Changed only 1 prediction vs V3. No LB improvement.
- ❌ **Attribution-based post-processing overrides** (V7) — Patterns like *German 16th Century +
  cap=ink → watercolor* were 100% reliable on training data (22/22, 10/10). Applied 9 overrides in
  test. **LB dropped from 97.111% to 96.66%** — the test Rothko rows are genuinely ink/tempera.
  Training patterns do not always generalize when sample sizes are small.
- ❌ **Fine-tuning DeBERTa / BERT on metadata.** 4000 rows + 85% null text fields → catastrophic
  overfitting. Char-n-gram TF-IDF captures the same vocabulary with a thousand-fold less compute.
- ❌ **CleanLab / dropping the 233 cap-disagreement rows.** Those rows *are* the training signal for
  override behaviour. Deleting them collapses the ceiling back to 94.17%.
- ❌ **SMOTE / class rebalancing.** Classes are 5.2–7.6% — nearly uniform. Nothing to fix.

### Hard ceiling analysis

The model achieves **97.925% OOF** and **97.111% LB**. The gap comes from the ~58 test disagreement
rows (cap≠true). We achieve ~66.5% accuracy on these — they require physical inspection or external
provenance knowledge the metadata doesn't encode. The hardest pair (`ink ↔ watercolor`) involves
paper-based, unframed works from overlapping centuries with nearly identical image statistics.
"""))

# ---- Section 12 Submission ----
cells.append(md("""## 12. Final Submission
"""))

cells.append(code("""sub = pd.read_csv('submission_v3.csv')
print('Submission shape:', sub.shape)
print('\\nClass distribution:')
print(sub['y'].map(LABEL_MAP).value_counts().sort_index())
print('\\nSample rows:')
print(sub.head())
"""))

# ---- Section 13 Model Card ----
cells.append(md("""## 13. Model Card

| Field | Value |
|---|---|
| **Intended use** | Classify the medium of an NGA artwork given its metadata record (8-class). |
| **Training data** | 4 000 rows of NGA metadata; 8 balanced classes; weakly labelled via `cap` (5.8% disagreements with `y`). |
| **Architecture** | ~340 engineered features → LightGBM (num_leaves=127, lr=0.015, max_depth=9, early_stopping=150) → 3-seed average (seeds 42, 7, 123). |
| **External features** | Image histograms (200px IIIF thumbnails, 43 features), ResNet-50 embeddings → SVD100, Wikidata P186 material flags (14 binary). |
| **5-fold CV OOF accuracy** | **97.925%** (3-seed blend). Regex-only baseline: 94.17%. |
| **Public LB accuracy** | **97.111%** (submission_v3.csv, 971 / 1000 test rows). |
| **Disagreement row accuracy** | 66.52% on the 233 train rows where cap≠y (the "forensic reattribution" challenge). |
| **Known failure modes** | (a) `ink ↔ watercolor` is the hardest pair — paper-based, unframed, overlapping centuries, nearly identical image statistics. (b) Attribution-based overrides do not reliably generalize from train to test when sample sizes are <25. (c) Rows with all-null text and missing dims collapse to the cap baseline. |
| **Features used** | Regex cap one-hots, temporal anachronism flags, dim-text vocabulary, physical/pixel dims, accession prefix, 30 medium keywords, 30 category tokens, assistivetext flags, TF-IDF-SVD50, artist OOF encoding, image histograms, ResNet-50 SVD100, Wikidata P186 material binary flags. |
| **What it cannot do** | Distinguish sub-media (etching vs drypoint within `print`); operate without `cap` or `classification`; reliably override when attribution sample size is <25. |
| **Runtime** | Feature extraction: ~15 min (images + ResNet on MPS). Training: ~8 min (3 seeds × 5 folds on CPU). |

— End of investigation. —
"""))

nb = {
  "cells": cells,
  "metadata": {
    "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
    "language_info": {"name":"python","version":"3.9"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

with open('artwork_medium_solution.ipynb','w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote notebook with {len(cells)} cells")
