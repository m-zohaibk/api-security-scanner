# API Security Platform

[![CI](https://github.com/rahimahkhan/api-security-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rahimahkhan/api-security-platform/actions/workflows/ci.yml)

A standalone AI-powered API security testing, structure inspection, and anomaly detection platform built with **Python 3.11**, **Flask**, **scikit-learn (Isolation Forest)**, **PyTorch (LSTM + Autoencoder)**, and **SQLAlchemy**.

Extending OWASP OFFAT under the MIT License, this platform analyzes live API target URLs without requiring OpenAPI specification documents.

---

## Detection Architecture

1. **Layer 1 — Signature Engine**: Evaluates HTTP request/response headers, parameters, and bodies against deterministic regex rules, missing security header patterns, stack trace error exposures, and PII leaks.
2. **Layer 2 — ML Anomaly Detection**: Extracts 17 numerical HTTP features (`runtime-http-17-v1`) (method, path depth, URL length, entropy, status code, response size, header count, auth status) and scores anomalies using a trained **Isolation Forest** model.
3. **Layer 3 — Deep Learning**: Combines character-level **LSTM** sequence modeling with a PyTorch **Autoencoder** feature reconstruction error benchmark.

---

## Quick Start (Docker)

Launch the platform dashboard along with the local VAmPI target app in one command:

```bash
docker compose up --build
```

- **Web Dashboard**: Access at [http://localhost:5000](http://localhost:5000)
- **VAmPI Vulnerable API**: Running at [http://localhost:5001](http://localhost:5001)

---

## Manual Setup

1. **Clone & Virtual Environment**:
   ```bash
   git clone https://github.com/rahimahkhan/api-security-platform.git
   cd api-security-platform
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Baseline Machine Learning & Deep Learning Models**:
   ```bash
   python training/prepare_dataset.py
   python training/train_isolation_forest.py
   python training/train_lstm.py
   python training/train_autoencoder.py
   ```

4. **Launch Application**:
   - **CLI Inspection Scan**:
     ```bash
     python main.py --url http://localhost:5001
     ```
   - **Web Dashboard**:
     ```bash
     python main.py --dashboard
     ```

---

## Dataset Instructions

- **Primary Dataset**: HTTP CSIC 2010 Web Application Attacks ([Kaggle Download](https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks)). Place downloaded raw text files inside `datasets/raw/csic_2010/`.

---

## Test Target Applications

The scanner can be evaluated against these vulnerable targets:
- **VAmPI** (`erev0s/vampi`): Vulnerable REST API (OWASP API Top 10) running locally on port 5001.
- **DVWA** (`digininja/dvwa`): Damn Vulnerable Web Application.
- **WebGoat** (`webgoat/webgoat`): OWASP Java vulnerable application.

---

## License

Distributed under the [MIT License](LICENSE).
