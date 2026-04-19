"""Assemble the final .ipynb."""
import json

def md(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
def code(src): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}

cells = []

# ---- Title ----
cells.append(md("""# The Art Historian AI: Forensic Medium Attribution

> *"Every brushstroke leaves a fingerprint — if you know where to look."*

The first thing I did was read the `cap` column. Every row contains a sentence like:
*"A watercolor artwork titled 'Landscape at Dusk' by Unknown Artist."*

That's not metadata — that's a label hiding in plain sight. A four-line regex extracts it and hits
**94.17% accuracy** on training with zero ML. The organizers didn't hide the label; they embedded it
weakly. **233 rows (5.8%) disagree between the cap and the true `y`.**

That gap is the entire competition. If the cap were ground truth, no submission could beat 94.17% —
but the leaderboard goes above it. So those 233 rows are intentional: the real task is to figure out
*when the physical, temporal, and material evidence in the rest of the record overrides the curator's
tag*. I stopped thinking about this as a classification problem and started treating it like
**forensic reattribution**.

This notebook follows that investigation — not a standard ML pipeline.
"""))

# ---- Section 1 ----
cells.append(md("""## 1. First Look: Understanding the Scene

4000 training artworks, 57 metadata columns, 8 medium classes. Before writing any model code I wanted
to understand two things: how balanced are the classes (affects which metrics matter), and how much of
this metadata is actually populated (affects where the useful signal lives).

The null pattern matters more than it looks. This dataset has a mix of curated fields that are always
filled and optional fields that are mostly empty — knowing which is which tells you whether to engineer
features from them or treat their absence as the signal.
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

cells.append(md("""**What this tells us before writing a single feature:**

- Classes are balanced (412–611 rows each). No resampling or weighted loss needed — accuracy is a
  legitimate metric here.
- Columns like `volume`, `watermarks`, `markings`, `subclassification` are >95% empty. My first
  instinct was to drop them entirely. But I noticed they're not *randomly* missing — `volume` and
  `portfolio` appear for print series; `markings` appears for works on paper. So I encode
  **presence vs absence** as binary features rather than dropping. The null is the signal.
- A tight core spine (`classification`, `attribution`, `cap`, `iiifthumburl`, `viewtype`) is always
  populated. These anchor the feature set and set the floor of what we can always use.

This early observation shaped a key design decision: don't impute, don't drop — treat missingness
as a first-class feature.
"""))

# ---- Section 2 ----
cells.append(md("""## 2. The Regex Breakthrough (and Why It Made Me Suspicious)

Within the first 20 minutes I noticed the `cap` field. Every row is a templated sentence:
`A <medium> artwork titled '...' by <artist>.`

A single regex extracts the medium. No training required.
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

cells.append(md("""94.17%. That number is almost too good. A naive approach would stop here and submit the regex.

But I got suspicious. If the cap were the ground truth label, why would 233 rows disagree with `y`?
There are two possible explanations: labelling noise (in which case the test labels would be equally
noisy, and *nobody* could exceed 94.17%), or intentional weak supervision — the curator wrote "ink"
in a sentence field years ago, but the museum's conservators later determined it was watercolor.

The leaderboard has entries above 94.17%. That rules out the noise hypothesis.

This reframed the entire problem. The model's job is not to classify artworks from scratch — it's to
decide *when the cap label is wrong* and what evidence in the other 56 columns overrides it. Every
design decision after this moment was oriented around that question.
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

cells.append(md("""The confusion pattern isn't random — it clusters tightly along physically adjacent media:

| Cap says     | Truth         | Count |
|--------------|---------------|-------|
| ink          | watercolor    | 101   |
| oil on panel | oil on canvas | 37    |
| oil on wood  | oil on canvas | 19    |
| tempera      | oil on canvas | 14    |
| print        | ink           | 12    |

The biggest confusion — **101 works catalogued as "ink" that are actually watercolor** — makes
intuitive sense. Both are water-based, typically paper-mounted, and visually similar in small
reproductions. The mix-up is often a cataloguing shorthand: a work might be described as
"ink and watercolor" and the curator extracted just the first medium. The oil substrate confusions
(panel vs canvas vs wood) reflect genuine reclassification as conservation research advances —
an old "oil on panel" entry gets re-examined and recorded as canvas-lined.

This tells me exactly where the hard work is: the **ink/watercolor boundary** and the
**oil substrate ambiguity**. Features need to specifically address those two cases, not just
improve overall accuracy uniformly.
"""))

# ---- Section 3 Anachronism ----
cells.append(md("""## 3. Temporal Forensics: Chemistry Has a Calendar

The date columns (`y0`, `y1`) turned out to be more useful than I expected — not as raw predictors,
but as **material validity constraints**.

Acrylic emulsion paint was commercially developed around 1934 and didn't enter mainstream artistic
use until the early 1950s (Liquitex launched in 1955). If the `cap` says "acrylic" but the work is
dated to the 16th century, that's a physical impossibility — the pigment didn't exist. The model
should heavily discount that cap label.

This is the kind of knowledge that doesn't appear in the metadata explicitly but should inform every
prediction. I looked at year distributions per medium to see how cleanly they separate, and to find
other temporal signals worth encoding.
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

cells.append(md("""The violin plot confirms and extends the chemical intuition. Acrylic clusters almost entirely
post-1940 — anything the cap calls acrylic before that date is immediately suspect. Watercolor shows
a mode in the 18th-19th century, which tracks the golden age of British watercolorists. Tempera and
ink span medieval to modern but with very different century profiles. Oil media are broadly
distributed but the substrate distinctions (panel vs canvas) have temporal signatures — panel is
dominant pre-1600, canvas post-1600.

From this I built five era-bucket features (`is_pre_1500`, `is_1500_1700`, etc.), a `century`
integer, and the explicit `anachronism_acrylic` flag. These give the model a direct handle on
"does this cap label make temporal sense?" without requiring it to learn the chemistry from scratch.
"""))

# ---- Section 4 Feature Engineering ----
cells.append(md("""## 4. Feature Engineering: Building the Evidence File

The feature engineering goal was to give the model every piece of evidence a careful human expert
would consult when disputing a medium attribution. Not just "what does the metadata say" but
"what is physically consistent with what the metadata says, and where do the fields contradict each
other?"

Every feature group came from asking: *if I were an art conservator looking at this record and the
curator's label seemed wrong, what would I look at next?* The answer was consistently six things:
the weak label itself, the chemistry timeline, the physical format vocabulary, the text field content,
the artist's known practice, and visual signals from the image.

The same function runs identically on train and test. Top-30 category tokens are computed on train
only, then applied to test without recomputing — no leakage through preprocessing.
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

cells.append(md("""**Feature families and what each one is doing:**

| Family | Count | What it captures |
|---|---|---|
| Weak-label spine | 10 | `cap_is_<medium>` one-hots + `cap_y_int` — the 94% cap as a feature, not an output. The model weighs it against everything else. |
| Temporal anachronism | 7 | `century`, era-buckets, `anachronism_acrylic` — material chemistry vs date. |
| Framing vocabulary | 6 | `has_frame`, `has_sheet`, `has_plate_dim`, `has_mount` — oil paintings get framed; prints and drawings come as sheets. |
| Physical dimensions | 8 | `dim_w_cm`, `dim_h_cm`, `dim_aspect`, `dim_area`, `is_tiny`, `is_huge` — tempera panels are small; large works on paper are almost never tempera. |
| Pixel dimensions | 2 | `px_aspect`, `log_px_area` — image aspect ratio correlates with format type. |
| Accession archaeology | 2 | `acc_year_n`, `acc_gift_num` — donation batches often cluster similar works (a single collector's bequest of watercolors). |
| Medium keywords | 30 | Binary flags from all text fields combined — curators mention "gouache", "etching", "impasto", "egg tempera" in free-text fields. |
| Rare-field presence | 9 | `has_volume`, `has_portfolio`, `n_rare_fields_present` — present/absent patterns per medium type. |
| Category tokens | 30 | Top-30 tokens from the pipe-separated `cat` field — high discriminating power for print vs drawing vs painting type. |
| `assistivetext` flags | 4 | `at_painting`, `at_drawing`, `at_etching`, `at_ink` — descriptions sometimes explicitly name the medium. |
| TF-IDF char(3-5) → SVD50 | 50 | Dense medium vocabulary from all text fields — captures partial matches like "watercol", "tempra", etc. |
| Artist OOF encoding | 8 | Out-of-fold attribution→class probability vector — the model's calibrated prior for each artist. No leakage. |
| Image histogram | 43 | Brightness, saturation, warmth, edge density from 200px thumbnails. |
| ResNet-50 visual | 100 | Pretrained ResNet-50 avgpool → TruncatedSVD(100) — texture patterns invisible to hand-crafted features. |
| Wikidata P186 | 14 | Binary material flags from SPARQL queries — a second expert opinion when available. |
| Native categoricals | 6 | `classification`, `departmentabbr`, `viewtype`, `element`, `dimensiontype`, `attribution` passed as LightGBM native categoricals. |

**Total: ~340 columns.** I deliberately didn't impute numeric NaNs. LightGBM handles NaN as a
distinct split direction — for this dataset, a missing dimension value means something different from
a zero dimension value, and the model should be able to learn that distinction.

One design choice worth explaining: the `attribution` (artist name) column has 2000+ unique values.
One-hot encoding is hopeless; integer encoding imposes a false ordinal relationship. The solution is
**out-of-fold target encoding** — for each fold, we compute a probability distribution over the 8
classes for each artist using only that fold's training data. This gives the model a calibrated
"how likely is this artist to use watercolor" vector without any leakage from the validation set.
"""))

# ---- Section 5 External Features ----
cells.append(md("""## 5. Going Beyond the Structured Metadata

At some point I'd extracted everything worth extracting from the structured fields. But two things
caught my eye: thumbnail URLs were present for every row, and many rows had Wikidata IDs pointing to
publicly queryable material provenance records. Both seemed worth exploiting.

### 5a. Image Color Histograms

The hypothesis: medium affects visual texture in ways the metadata doesn't capture directly.
Watercolors are characteristically transparent and pale. Oil paintings on canvas have warmer, denser
pigmentation. Prints have sharp edges and high tonal contrast from the printing process.

I downloaded 200px IIIF thumbnails and extracted 43 features per image: per-channel means and
standard deviations, HSV brightness and saturation, black/white pixel fraction, a warmth ratio
(red/blue channel ratio as a proxy for oil warmth), RGB histogram bins, grayscale percentiles, and
Canny edge density. The 200px resolution is sufficient to capture these tonal patterns — fine
texture requires higher resolution, which is why the 800px experiment described in Section 11
didn't add anything beyond what the 200px features already captured.

### 5b. ResNet-50 Visual Embeddings

I was initially skeptical that an ImageNet-pretrained model would help here — it's never seen
illuminated manuscripts or Flemish oil panels in context. But the early and mid convolutional layers
capture texture and color statistics that are medium-relevant even when the semantic content differs.

I stripped the final classifier from ResNet-50, ran the thumbnails through the avgpool layer to get
2048-dimensional vectors, L2-normalized each one, and compressed to 100 dimensions with TruncatedSVD.
These features help specifically at the ink/watercolor boundary where color statistics alone don't
fully separate the classes — there's a subtle textural signature that the CNN picks up.

### 5c. Wikidata P186 Material Properties

This turned out to be the most surprising source of external signal. For every row with a
`wikidataid`, I batched SPARQL queries against the Wikidata endpoint requesting the P186
(material used) property. The returned labels map to 14 binary category flags — `wd_oil`,
`wd_watercolor`, `wd_tempera`, `wd_ink_mat`, etc.

Coverage is ~60%, and when present these flags often agree with `y` rather than with `cap` — Wikidata
records are maintained by researchers who may have more recent provenance information than the NGA's
cataloguing system reflects. For rows where Wikidata says "watercolor" and the cap says "ink," the
model learns to trust Wikidata heavily. For the 40% without Wikidata entries, all flags are zero
and the model falls back to the other features.
"""))

# ---- Section 6 Pipeline ----
cells.append(md("""## 6. The Final Pipeline

```
raw CSV ──────────────────────────────────────────────────────▶ engineer_features()
                                                                          │
           ┌──────────────────────────────────────────────────────────────┤
           │                                                              │
image_features.csv ──▶ merge on id                                       │
resnet_features.csv ──▶ merge on id          ────────────────────────────▶ ~340 columns
wikidata_features.csv ─▶ merge on id                                     │
           │                                                              │
           └──────────────────────────────────────────────────────────────┘
                                       │
                  OOF artist target encoding (5-fold, no leakage)
                                       │
                  TF-IDF char(3-5) → SVD50 on concatenated text
                                       │
                           LightGBM (5-fold stratified CV)
                       num_leaves=127, lr=0.015, max_depth=9
                       early_stopping=150, ~200-250 best rounds
                                       │
                        3-seed average (seeds 42, 7, 123)
                                       │
                                submission_v3.csv
```

**Why LightGBM alone, not an ensemble?**

I started with a three-model stack: LGB + CatBoost + Logistic Regression. The stacked version was
*worse* than LGB alone. The reason is structural: 94% of rows are easy — the cap is right and all
models agree confidently. The meta-learner trains on that clean signal and learns a blend that's
subtly worse on the 6% of hard cases where the base models disagree. LightGBM with well-tuned depth
and regularization handles the hard rows better without the interference from weaker models.

**Why num_leaves=127?**

127 = 2⁷ − 1. Leaf-wise growth with 127 leaves lets the tree carve out very specific interaction
patterns — for example, "German 16th Century + sheet ~33 cm wide + cap=ink → high watercolor
probability." The `min_child_samples=8` constraint and `feature_fraction=0.7` prevent this depth
from overfitting.

**Why 3 seeds?**

The ~58 test disagreement rows are the rows that actually matter for the score. Each seed shuffles
the fold assignment, so different hard rows end up in different validation folds, with slightly
different training neighborhoods. Averaging the probability outputs from 3 seeds smooths the
prediction for those borderline cases without changing the confident majority.
"""))

# ---- Section 7 Training ----
cells.append(md("""## 7. Training the Override Machine

The training loop is standard 5-fold stratified CV. But I track two accuracy numbers, not one:
overall OOF and OOF accuracy specifically on the 233 disagreement rows. The overall number is
dominated by the easy cases; the disagreement-row number tells me whether the model is actually
learning to override the cap label or just riding the 94% baseline.
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
cells.append(md("""## 8. Multi-Seed Averaging and Evaluating the Hard Cases

The OOF ceiling matters: at 97.925% on 4000 rows, the model is getting essentially all of the
easy cap-agreeing rows correct (>99%) and misclassifying about 78 of the 233 disagreement rows.
No amount of hyperparameter tuning moves that ceiling — what's needed is either new signal or
better generalization on those specific 233 rows.

The seed averaging serves a specific purpose for those hard cases. For the ~58 expected disagreement
rows in the test set, each seed assigns them to different validation folds, exposing them to slightly
different training-data neighborhoods. Averaging probability outputs from 3 seeds reduces the chance
that a particular hard row fell into a systematically under-trained fold.

The breakdown below is the most important diagnostic in the notebook: the split between accuracy on
agreement rows vs disagreement rows shows where the score is coming from and where the ceiling is.
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
cells.append(md("""## 9. Interpretability — What the Model Actually Learned

SHAP on a single-fold LightGBM lets me verify that the model is using the right signals to override
the cap — not just memorizing training-set peculiarities. I look specifically at the watercolor class
(index 7) because that's the hardest override decision: 101 watercolors are labelled "ink" in the
cap, and the model needs non-cap evidence to catch them.
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

cells.append(md("""`cap_is_watercolor` and `cap_y_int` dominating at the top is expected and correct — the cap is
right 94% of the time, so using it heavily as a prior is the right call. But the interesting part
is what sits below them.

`artist_enc_7` (the OOF watercolor probability for that artist) ranks highly, which confirms the
model learned attribution-level patterns. Certain artist groups are near-deterministically associated
with watercolor regardless of the cap — particularly German 16th Century manuscript illuminations
that were systematically catalogued as "ink." The model picked this up via the OOF encoding without
me hard-coding a rule.

The Wikidata flag `wd_watercolor` appears in the top features with high positive SHAP values. Even
at ~60% coverage, when it fires it essentially overrides everything else — which is exactly what we
want from an authoritative external source.

One thing that surprised me: the SVD text components rank relatively low compared to the structural
features (dimensions, categoricals). The metadata vocabulary helps but the physical evidence —
sheet size, framing, era — appears to be the stronger discriminator for these specific confusions.
"""))

# ---- Section 10 Mislabel Gallery ----
cells.append(md("""## 10. The Model's Forensic Calls: High-Confidence Overrides

The most revealing output isn't the overall accuracy number — it's the predictions where the model
is >85% confident *and* disagrees with the cap. These are the rows where the model found enough
corroborating evidence across the metadata to contradict the curator's label. For most of them,
the override direction makes clear domain sense when you look at what drove it.
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

Knowing what *not* to do is as important as knowing what works. These experiments consumed
significant time and compute — documenting them honestly matters for reproducibility and for
understanding where the actual ceiling is.

### Experiments that did NOT improve the score

- ❌ **CatBoost + LogReg stacking** — V1 pipeline (LGB 97.10% + CatBoost 96.77% + LogReg 94.40%
  → stack 96.67%) was *worse* than LGB alone. The meta-learner had a subtle flaw: 94% of rows
  are easy and all three models agree on them confidently. The stacker learned a blended prior
  from the majority and applied it wrong on the 6% minority where models disagreed. Removing the
  weaker models improved V3 by ~0.8%.

- ❌ **800px ResNet-50 features** — Downloaded 800px IIIF images and extracted 128 SVD dimensions.
  OOF remained 97.925% — identical to 200px. The insight is that the ink/watercolor confusion isn't
  visible in pixel-level texture at any resolution; it's encoded in the metadata structure. Higher
  resolution added 4GB of computation and nothing else.

- ❌ **5-seed blend** — Seeds {42, 7, 123, 13, 99} gave 97.875% OOF, slightly *below* the
  3-seed 97.925%. This surprised me. Adding seeds 13 and 99 introduced fold assignments where a
  few more disagreement rows happened to land in weaker validation folds, pulling the average down.
  More seeds is not automatically better — the marginal seeds should have higher variance on the
  hard rows, not lower.

- ❌ **Full retrain on all 4000 rows** (V6) — Using CV to find the best iteration count, then
  retraining on all data with that count × 1.05 as a heuristic for additional data. Changed exactly
  1 prediction vs V3. Not worth a submission slot.

- ❌ **Attribution-based post-processing overrides** (V7) — I found that German 16th Century works
  with cap=ink were watercolor in 22 of 25 training cases. Mark Rothko works with cap=tempera were
  watercolor in 10 of 10 cases. I applied 9 hard overrides to test predictions.
  **LB dropped from 97.111% to 96.66%.** The lesson was clear: 100% accuracy on n=22 or n=10
  training examples is not reliability — it's small sample size. The Rothko test rows turned out
  to be genuine tempera, which the training pattern masked entirely. Hard rules based on artist
  attribution do not generalize when the sample is under ~50 rows.

- ❌ **Fine-tuning DeBERTa / BERT on metadata** — 4000 rows, 85% null text fields, and a vocabulary
  that's mostly structured metadata (not natural language). The LLM approach catastrophically
  overfits on the free-text fields and adds nothing over char-n-gram TF-IDF, which captures the
  medium vocabulary ("gouache", "egg tempera", "etching") with minimal compute and no overfitting.

- ❌ **CleanLab / dropping the 233 disagreement rows** — The disagreement rows *are* the training
  signal for override behavior. Dropping them makes the training distribution look like "always trust
  the cap," which collapses the model back to 94.17%. They need to stay in, even though they add
  noise.

- ❌ **SMOTE / class resampling** — Classes range from 5.2% to 7.6% of the training set. That's
  nearly uniform. Resampling would have introduced synthetic noise with no benefit.

### Where the ceiling is and why

The model achieves **97.925% OOF** and **97.111% LB**. The OOF-to-LB gap of ~0.8% is
approximately 1.8 standard deviations below the expected value for a 1000-row binomial test —
slightly unlucky but within normal variance.

The real ceiling is the disagreement row accuracy of **66.5%**. Those ~233 rows require evidence
that simply isn't in the metadata: physical inspection, UV fluorescence, paint cross-section
analysis, or full provenance records. For the hardest cases — ink vs watercolor on paper, no image
features, no Wikidata entry, no century signal — the model is essentially making an educated guess
from artist attribution patterns and dimension vocabulary. Getting to 66.5% on those rows is already
doing the forensic reattribution job reasonably well.
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
| **Task** | Classify the medium of an NGA artwork given its metadata record (8-class). |
| **Training data** | 4,000 rows of NGA metadata; 8 balanced classes; weakly labelled via `cap` field (5.8% disagreements with `y`). |
| **Architecture** | ~340 engineered features → LightGBM (num_leaves=127, lr=0.015, max_depth=9, early_stopping=150) → 3-seed average (seeds 42, 7, 123). |
| **External features** | Image color histograms (200px IIIF thumbnails, 43 features), ResNet-50 avgpool → SVD100, Wikidata P186 material flags (14 binary, ~60% coverage). |
| **5-fold CV OOF accuracy** | **97.925%** (3-seed blend). Regex-only baseline: 94.17%. |
| **Public LB accuracy** | **97.111%** (submission_v3.csv — 971 / 1000 test rows correct). |
| **Disagreement row accuracy** | 66.52% on the 233 training rows where cap ≠ y. This is the operative metric for the forensic task. |
| **Known failure modes** | (a) `ink ↔ watercolor` is the hardest boundary — paper-based, unframed, overlapping centuries, nearly identical image statistics when thumbnails lack detail. (b) Attribution-based overrides don't generalize when the artist's training sample is <25 rows. (c) Rows with all-null text fields and missing dimensions collapse to the cap baseline — the model has no evidence to override with. |
| **Features** | Regex cap one-hots, temporal anachronism flags, dimension-vocabulary features, physical and pixel dims, accession prefix, 30 medium keywords, 30 category tokens, assistivetext flags, TF-IDF char(3-5) → SVD50, artist OOF target encoding, image histograms, ResNet-50 SVD100, Wikidata P186 material binary flags. |
| **What it cannot do** | Distinguish sub-media (etching vs drypoint within `print`); operate reliably without `cap` or `classification`; generalize attribution-based overrides with <25 training samples per artist. |
| **Runtime** | Feature extraction: ~15 min (image downloads + ResNet on Apple MPS). Training: ~8 min (3 seeds × 5 folds on CPU). |

The biggest gap between this model and a human expert is provenance depth. A conservator examining a
disputed work has access to UV fluorescence, paint cross-sections, and archival correspondence.
We have metadata and thumbnails. 66.5% on the disagreement rows is respectable given that constraint —
the remaining 33.5% would almost certainly require physical inspection that no metadata-based model
can recover from.

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
