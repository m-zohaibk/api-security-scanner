import os
import sys
import math
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = BASE_DIR / "datasets" / "raw" / "csic_2010" / "csic_database.csv"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_entropy(text):
    if not text or len(str(text).strip()) == 0:
        return 0.0
    text = str(text)
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

def count_special_chars(text):
    if not text or str(text).strip() == 'nan':
        return 0
    special = set("'\";<>(){}[]&|`!@#$%^*\\=+")
    return sum(1 for c in str(text) if c in special)

def encode_method(method):
    mapping = {
        'GET': 1, 'POST': 2, 'PUT': 3,
        'DELETE': 4, 'PATCH': 5, 
        'HEAD': 6, 'OPTIONS': 7
    }
    return mapping.get(str(method).strip().upper(), 0)

def get_url_depth(url):
    if not url or str(url).strip() == 'nan':
        return 0
    try:
        path = str(url).split('HTTP')[0].strip()
        parts = [p for p in path.split('/') if p and 'localhost' not in p]
        return min(len(parts), 10)
    except:
        return 0

def get_url_length(url):
    if not url or str(url).strip() == 'nan':
        return 0
    return min(len(str(url)), 500)

def get_param_count(url):
    if not url or '?' not in str(url):
        return 0
    try:
        query = str(url).split('?')[1]
        query = query.split('HTTP')[0]
        return len([p for p in query.split('&') if '=' in p])
    except:
        return 0

def get_query_length(url):
    if not url or '?' not in str(url):
        return 0
    try:
        query = str(url).split('?')[1]
        query = query.split('HTTP')[0]
        return min(len(query), 500)
    except:
        return 0

def calculate_keyword_risk(payload, url):
    risk_keywords = [
        'select', 'union', 'insert', 'drop',
        'delete', 'update', 'exec', 'script',
        'alert', 'onerror', 'onload', 'eval',
        'base64', 'passwd', 'shadow', 'whoami',
        'null', 'none', 'true', 'admin', 'root'
    ]
    combined_text = (str(payload) + " " + str(url)).lower()
    words = [w for w in combined_text.replace('&', ' ').replace('?', ' ').replace('=', ' ').replace('/', ' ').split() if w]
    total_words = max(len(words), 1)
    keyword_matches = sum(1 for kw in risk_keywords if kw in combined_text)
    return round(min(keyword_matches / total_words, 1.0), 4)

def calculate_param_risk(url):
    risky_params = [
        'id', 'user', 'uid', 'admin', 'debug',
        'test', 'token', 'key', 'pass', 'pwd',
        'cmd', 'exec', 'query', 'sql', 'file'
    ]
    if not url or '?' not in str(url):
        return 0
    try:
        query = str(url).split('?')[1].split('HTTP')[0]
        params = [p.split('=')[0].lower().strip() for p in query.split('&') if '=' in p]
        return sum(1 for p in params if any(rp in p for rp in risky_params))
    except Exception:
        return 0

def calculate_url_encoded_ratio(url):
    if not url or str(url).strip() == 'nan':
        return 0.0
    return round(str(url).count('%') / max(len(str(url)), 1), 4)

def calculate_payload_digit_ratio(payload):
    if not payload or str(payload).strip() == 'nan':
        return 0.0
    digit_count = sum(1 for c in str(payload) if c.isdigit())
    return round(digit_count / max(len(str(payload)), 1), 4)

def check_sql_structure(payload, url):
    sql_patterns = [
        r"'\s*(or|and)\s+'",
        r"\bunion\b.*\bselect\b",
        r"--\s*$",
        r";\s*(drop|delete|insert|update)",
        r"'\s*=\s*'"
    ]
    combined = (str(payload) + " " + str(url)).lower()
    import re
    return 1 if any(re.search(p, combined, re.IGNORECASE) for p in sql_patterns) else 0

def extract_features(row):
    url = str(row.get('URL', ''))
    content = str(row.get('content', ''))
    method = str(row.get('Method', 'GET'))
    cookie = str(row.get('cookie', ''))
    lenght = row.get('lenght', 0)
    
    # Clean NaN values
    content = '' if content == 'nan' else content
    cookie = '' if cookie == 'nan' else cookie
    
    # Payload: use content if exists, else URL params
    payload = content if content else ''
    if not payload and '?' in url:
        payload = url.split('?')[1].split('HTTP')[0]
    
    try:
        payload_len = float(lenght) if str(lenght) != 'nan' else len(payload)
    except:
        payload_len = len(payload)
    
    # Auth header check
    auth_present = 1 if (
        'authorization' in str(row).lower() or
        'bearer' in str(row).lower() or
        (cookie and cookie != 'nan')
    ) else 0
    
    # Header count (non-null columns)
    header_cols = [
        'User-Agent', 'Pragma', 'Cache-Control',
        'Accept', 'Accept-encoding', 
        'Accept-charset', 'language',
        'cookie', 'content-type', 'connection'
    ]
    header_count = sum(
        1 for col in header_cols 
        if str(row.get(col, 'nan')) != 'nan' and str(row.get(col, '')).strip() != ''
    )
    
    return {
        'encoded_method': encode_method(method),
        'path_depth': get_url_depth(url),
        'url_length': get_url_length(url),
        'query_param_count': get_param_count(url),
        'query_string_length': get_query_length(url),
        'payload_length': min(float(payload_len), 1000),
        'payload_entropy': calculate_entropy(payload),
        'special_char_count': count_special_chars(payload + url),
        'header_count': header_count,
        'auth_header_present': auth_present,
        'status_code': 200,
        'response_size': min(float(payload_len) * 2, 5000),
        'keyword_risk_score': calculate_keyword_risk(payload, url),
        'param_name_risk': calculate_param_risk(url),
        'url_encoded_ratio': calculate_url_encoded_ratio(url),
        'payload_digit_ratio': calculate_payload_digit_ratio(payload),
        'has_sql_structure': check_sql_structure(payload, url),
    }

def main():
    # Load real CSV
    print('Loading CSIC 2010 CSV...')
    df = pd.read_csv(RAW_CSV_PATH)
    print(f'Loaded: {df.shape[0]} rows, {df.shape[1]} columns')

    # Check classification column
    print('Classification values:')
    print(df['classification'].value_counts())

    # Use classification column as label
    # 0 = normal, 1 = anomalous
    df['label'] = df['classification'].astype(int)

    print(f'Normal samples: {(df["label"]==0).sum()}')
    print(f'Attack samples: {(df["label"]==1).sum()}')

    # Extract features for every row
    print('Extracting features...')
    feature_list = []
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f'  Processing row {idx}...')
        feat = extract_features(row.to_dict())
        feat['label'] = row['label']
        feature_list.append(feat)

    features_df = pd.DataFrame(feature_list)
    
    # Merge with self-generated telemetry scans if available
    vampi_scans_path = BASE_DIR / "datasets" / "self_generated" / "vampi_scans.csv"
    if vampi_scans_path.exists():
        print('Merging self-generated VAmPI telemetry dataset...')
        vampi_df = pd.read_csv(vampi_scans_path)
        feature_cols = [c for c in features_df.columns if c in vampi_df.columns]
        if 'label' in vampi_df.columns and len(feature_cols) >= 12:
            # Sample up to 5000 records to maintain class balance
            sample_size = min(len(vampi_df), 5000)
            vampi_sampled = vampi_df[feature_cols].sample(n=sample_size, random_state=42)
            features_df = pd.concat([features_df, vampi_sampled], ignore_index=True)
            print(f'Merged {len(vampi_sampled)} self-generated samples into combined dataset.')

    print(f'Combined features extracted: {features_df.shape}')
    features_df = features_df.fillna(0.0)

    # Save features CSV
    features_csv_path = PROCESSED_DIR / "features.csv"
    features_df.to_csv(features_csv_path, index=False)
    print(f'Saved features.csv to {features_csv_path}')

    # Train/test split 80/20
    X = features_df.drop('label', axis=1).fillna(0.0)
    y = features_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save as DataFrames with label
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['label'] = y_train.values
    train_csv_path = PROCESSED_DIR / "train.csv"
    train_df.to_csv(train_csv_path, index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['label'] = y_test.values
    test_csv_path = PROCESSED_DIR / "test.csv"
    test_df.to_csv(test_csv_path, index=False)

    # Save scaler for use during scanning
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print('\n=== DATASET STATISTICS ===')
    print(f'Total samples: {len(features_df)}')
    print(f'Normal samples: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)')
    print(f'Attack samples: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)')
    print(f'Train size: {len(X_train)}')
    print(f'Test size: {len(X_test)}')
    print(f'Features: {X.columns.tolist()}')
    print('Done. Real CSIC 2010 data prepared.')

if __name__ == "__main__":
    main()
