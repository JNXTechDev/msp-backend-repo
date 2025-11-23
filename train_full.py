import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# After predicting with pipeline on test set:
y_pred_baseline = baseline_pipeline.predict(X_test)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
precision_baseline, recall_baseline, f1_baseline, _ = precision_recall_fscore_support(y_test, y_pred_baseline, average='binary')

print("Baseline Pipeline Evaluation:")
print(f"Accuracy: {accuracy_baseline:.4f}")
print(f"Precision: {precision_baseline:.4f}")
print(f"Recall: {recall_baseline:.4f}")
print(f"F1 Score: {f1_baseline:.4f}\n")

y_pred_enhanced = enhanced_pipeline.predict(X_test)
accuracy_enhanced = accuracy_score(y_test, y_pred_enhanced)
precision_enhanced, recall_enhanced, f1_enhanced, _ = precision_recall_fscore_support(y_test, y_pred_enhanced, average='binary')

print("Enhanced Pipeline Evaluation:")
print(f"Accuracy: {accuracy_enhanced:.4f}")
print(f"Precision: {precision_enhanced:.4f}")
print(f"Recall: {recall_enhanced:.4f}")
print(f"F1 Score: {f1_enhanced:.4f}\n")


# Ensure local modules are importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from text_meta_transformer import TextMetaTransformer

BASE = os.path.abspath(os.path.dirname(__file__))
MODELS = os.path.join(BASE, 'models')
OUT = os.path.join(BASE, 'experiments_out')
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


DEFAULT_DATASET = os.path.join(BASE, '..', 'datasets', 'final_dataset.csv')


def preprocess_text(s):
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_dataset(fn=None):
    if fn is None:
        fn = DEFAULT_DATASET
    df = pd.read_csv(fn, encoding='latin-1')
    # build email_content
    if 'Subject' in df.columns and 'Message' in df.columns:
        df['email_content'] = df['Subject'].fillna('') + '\n\n' + df['Message'].fillna('')
    else:
        txt_cols = [c for c in df.columns if c.lower() in ('message','text','body')]
        if txt_cols:
            df['email_content'] = df[txt_cols[0]].astype(str)
        else:
            df['email_content'] = df.iloc[:,1].astype(str)

    # detect label
    label_col = None
    for cand in ['Spam/Ham','spam/ham','label','Label','spam','v1','v2']:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]

    raw_labels = df[label_col]
    # handle numeric labels (0/1) or textual labels
    if pd.api.types.is_numeric_dtype(raw_labels):
        labels = raw_labels.fillna(0).astype(int).map({1: 'spam', 0: 'ham'})
    else:
        labels = raw_labels.astype(str).str.lower().str.strip().map({'spam':'spam','ham':'ham','s':'spam','h':'ham'})
    df = df.assign(label=labels)
    df = df.dropna(subset=['label'])
    df['label_num'] = df['label'].map({'ham':0,'spam':1})
    df['email_content'] = df['email_content'].astype(str).apply(preprocess_text)

    if 'From' in df.columns:
        df['sender_email'] = df['From'].astype(str)
    else:
        df['sender_email'] = df['label'].apply(lambda v: 'spammer@gmail.com' if v=='spam' else 'user@company.com')

    def extract_domain(e):
        if isinstance(e,str) and '@' in e:
            return e.split('@')[-1].lower()
        return ''

    df['domain'] = df['sender_email'].apply(extract_domain)
    FREE = set(['gmail.com','yahoo.com','hotmail.com','outlook.com','aol.com','icloud.com','mail.com','protonmail.com','zoho.com','yandex.com','live.com','msn.com'])
    df['domain_flag'] = df['domain'].apply(lambda d: 1 if d in FREE else 0)
    return df

def train_and_save(target_size=20000, C=1.0, calib_cv=5, dataset_path=None):
    df = load_dataset(dataset_path)
    print('Loaded Dataset rows:', len(df))

    # split first
    from sklearn.model_selection import train_test_split
    try:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_num'])
    except Exception:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # upsample training partition only
    current = len(train_df)
    if current < target_size:
        needed = target_size - current
        print(f'Upsampling train set from {current} to {target_size} (+{needed})')
        up = train_df.sample(n=needed, replace=True, random_state=42)
        train_df = pd.concat([train_df, up], ignore_index=True)

    # pipelines
    vect = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2), min_df=2)
    base_clf = CalibratedClassifierCV(LinearSVC(C=C, dual=False), cv=calib_cv)
    baseline_pipeline = Pipeline([('vect', vect), ('clf', base_clf)])

    textmeta = TextMetaTransformer()
    enh_clf = CalibratedClassifierCV(LinearSVC(C=C, dual=False), cv=calib_cv)
    enhanced_pipeline = Pipeline([('textmeta', textmeta), ('clf', enh_clf)])

    # Fit
    print('Fitting baseline pipeline...')
    baseline_pipeline.fit(train_df['email_content'], train_df['label_num'].values)
    print('Fitting enhanced pipeline...')
    enhanced_pipeline.fit(train_df[['email_content','domain_flag']], train_df['label_num'].values)

    # evaluate on test
    y_test = test_df['label_num'].values
    prob_b = baseline_pipeline.predict_proba(test_df['email_content'])[:,1]
    prob_e = enhanced_pipeline.predict_proba(test_df[['email_content','domain_flag']])[:,1]
    pred_b = (prob_b >= 0.5).astype(int)
    pred_e = (prob_e >= 0.5).astype(int)

    results = {
        'baseline_test_acc': float(accuracy_score(y_test, pred_b)),
        'enhanced_test_acc': float(accuracy_score(y_test, pred_e)),
        'baseline_report': classification_report(y_test, pred_b, output_dict=True),
        'enhanced_report': classification_report(y_test, pred_e, output_dict=True)
    }

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    base_name_b = os.path.join(MODELS, f'baseline_pipeline_calibrated_full_{ts}.pkl')
    base_name_e = os.path.join(MODELS, f'enhanced_pipeline_calibrated_full_{ts}.pkl')
    print('Saving pipelines to', base_name_b, base_name_e)
    joblib.dump(baseline_pipeline, base_name_b)
    joblib.dump(enhanced_pipeline, base_name_e)
    # Save a canonical symlink-name for app.py to find (non-versioned)
    joblib.dump(enhanced_pipeline, os.path.join(MODELS, 'enhanced_pipeline.pkl'))

    # Save metrics and metadata
    with open(os.path.join(OUT, 'train_full_results.json'), 'w', encoding='utf-8') as fh:
        json.dump({'C':C, 'target_size':target_size, 'results':results, 'timestamp':ts}, fh, indent=2)

    meta = {'timestamp': ts, 'C':C, 'calibrated': True, 'seed':42, 'target_size': target_size}
    with open(os.path.join(MODELS, f'meta_{ts}.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)

    print('Training complete. Results saved to', OUT, 'and models saved to', MODELS)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='Path to final_dataset.csv')
    parser.add_argument('--target-size', type=int, default=20000)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--calib-cv', type=int, default=5)
    args = parser.parse_args()
    train_and_save(target_size=args.target_size, C=args.C, calib_cv=args.calib_cv, dataset_path=args.dataset)
