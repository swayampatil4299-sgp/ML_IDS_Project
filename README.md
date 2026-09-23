# 🛡️ SentinelShield IDS: Machine Learning Network Intrusion Detection System

An end-to-end, beginner-friendly **Intrusion Detection System (IDS)** web application built using **Python**, **Scikit-Learn**, and **Streamlit**. 

SentinelShield simulates real-time network traffic telemetry, preprocesses multi-modal packet data, and utilizes machine learning classification models (**Random Forest Classifier** and **Logistic Regression**) to accurately intercept cyberattacks such as **Denial of Service (SYN Flood)**, **SSH Brute Force**, and **Port Scanning Reconnaissance**.

---

## 🚀 Key Features

- **Synthetic Network Traffic Generator (`generate_data.py`)**:
  - Generates 6,000 realistic connection records with normal traffic patterns and common cyberattack signatures.
  - Automatically exports training and testing sets, plus a ready-to-use 100-packet sample batch test file (`data/sample_batch_test.csv`).
- **Robust Machine Learning Pipeline (`train_model.py`)**:
  - End-to-end `scikit-learn` `Pipeline` utilizing `ColumnTransformer`.
  - Feature scaling with `StandardScaler` for numerical metrics and categorical encoding with `OneHotEncoder`.
  - Trains and benchmarks **Random Forest Classifier** against **Logistic Regression**.
  - Evaluates models with Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
  - Automatically identifies the best-performing model and serializes the complete bundle with `joblib`.
- **Interactive Streamlit Web Dashboard (`app.py`)**:
  - **⚡ Real-Time Packet Inspector**: Live interactive form with sliders, number inputs, dropdowns, and **4 One-Click Attack Presets** (Normal Web Browsing, SYN Flood, SSH Brute Force, Port Scan).
  - **📁 Batch Traffic Simulator**: Drag-and-drop CSV log analyzer with 1-click demo loader, instant threat scoring, KPI metric cards, Donut Charts, and downloadable classified CSV reports.
  - **📊 Model Performance Dashboard**: Side-by-side metric comparison, interactive Confusion Matrix heatmaps, and Top Feature Importance visualizations.
  - **📖 Presentation & Educational Guide**: Built-in reference explaining IDS fundamentals, False Positives vs. False Negatives, and a 3-minute group presentation script.

---

## 📂 Project Architecture

```
ML_IDS_Project/
├── app.py                      # Interactive Streamlit Web Application
├── train_model.py              # ML Training, Evaluation & Serialization Pipeline
├── generate_data.py            # Synthetic Network Traffic Dataset Generator
├── requirements.txt            # Python Dependencies
├── README.md                   # Complete Documentation & Presentation Guide
├── data/
│   ├── network_traffic.csv     # 6,000-record training/testing dataset
│   └── sample_batch_test.csv   # 100-record test dataset for batch simulation
└── models/
    └── ids_model_payload.joblib # Serialized model pipeline & evaluation metadata
```

---

## ⚙️ Installation & Quickstart Guide

### 1. Clone or Open the Project
```bash
cd /path/to/ML_IDS_Project
```

### 2. Set Up Virtual Environment & Dependencies

You can use standard Python `venv` or `uv`:

#### Option A: Using standard Python (`venv` and `pip`)
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate       # On macOS/Linux
# .venv\Scripts\activate        # On Windows

# Install requirements
pip install -r requirements.txt
```

#### Option B: Using `uv` (Ultra-fast package manager)
```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt --python .venv/bin/python
```

---

## 🏃 Running the Application

### Step 1: Generate the Datasets
```bash
# Creates data/network_traffic.csv and data/sample_batch_test.csv
python generate_data.py
```

### Step 2: Train and Benchmark the Models
```bash
# Preprocesses data, trains Random Forest & Logistic Regression, saves models
python train_model.py
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```

The web dashboard will automatically open in your default browser at:
👉 **`http://localhost:8501`**

---

## 📊 Dataset Features & Attack Types

The dataset simulates core connection telemetry inspired by benchmark intrusion detection datasets (such as NSL-KDD):

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `duration` | Numerical | Connection duration in seconds (0 for rapid probes, >0 for sessions). |
| `protocol_type` | Categorical | Network transport protocol (`tcp`, `udp`, `icmp`). |
| `service` | Categorical | Destination service (`http`, `ftp`, `smtp`, `ssh`, `dns`, `telnet`, `other`). |
| `flag` | Categorical | TCP connection status flag (`SF` normal, `S0` SYN flood, `REJ` rejected, `RSTO` reset). |
| `src_bytes` | Numerical | Number of payload bytes sent from source to destination. |
| `dst_bytes` | Numerical | Number of payload bytes sent from destination to source. |
| `failed_logins` | Numerical | Count of failed authentication attempts (0 for normal, 2–6 for brute force). |
| `traffic_count` | Numerical | Number of connections to the same host in the past 2 seconds. |
| `logged_in` | Binary | 1 if successfully authenticated; 0 otherwise. |
| `label` | Target | Binary classification: `Normal` vs. `Intrusion/Attack`. |

### Simulated Attack Profiles:
1. **Normal Traffic (~65%)**: Standard HTTP/HTTPS, DNS, and SMTP requests with normal byte sizes, `SF` flag, and low connection frequencies.
2. **SYN Flood (DoS) (~15%)**: Massive bursts of TCP connections (`traffic_count` between 120 and 500), `S0` flag (unanswered SYN packets), zero destination bytes.
3. **SSH/FTP Brute Force (~10%)**: Elevated failed authentication attempts (`failed_logins` $\ge$ 2), targeting administrative protocols (`ssh`, `ftp`, `telnet`).
4. **Port Scan / Reconnaissance (~10%)**: Rapid probes across multiple destination services with high `REJ` and `S0` flags and minimal byte transfers.

---

## 🤖 Machine Learning Model Benchmarking

The system trains two distinct algorithms to illustrate model differences:

| Metric | Random Forest Classifier | Logistic Regression | Winner |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **99.92%** | 99.75% | 🌲 **Random Forest** |
| **Precision (Attack)** | **100.00%** | 99.52% | 🌲 **Random Forest** |
| **Recall (Attack)** | **99.76%** | 99.76% | **Tie** |
| **F1-Score** | **0.9988** | 0.9964 | 🌲 **Random Forest** |

### Why Random Forest Outperformed Logistic Regression:
- Network intrusions typically involve **non-linear compound rules** (for example: *If `traffic_count` is high AND `flag` is `S0` AND `src_bytes` is near zero $\rightarrow$ SYN Flood*).
- Random Forest naturally builds decision trees that isolate these multi-feature combinations without requiring manual feature interaction engineering.
- Logistic Regression provides a fast linear baseline, but struggles slightly with non-linear boundary intersections.

---

## 🧩 Confusion Matrix & Cybersecurity Metrics

| Term | In Plain English | Cybersecurity Impact |
| :--- | :--- | :--- |
| **True Positive (TP)** | Actual attack correctly identified as an attack. | Threat blocked; infrastructure protected. |
| **True Negative (TN)** | Benign packet correctly recognized as normal. | Legitimate traffic allowed without disruption. |
| **False Positive (FP)** | Benign packet falsely flagged as an attack. | **False Alarm / Alert Fatigue**: Security operations team wastes time investigating harmless traffic. |
| **False Negative (FN)** | Real attack incorrectly classified as normal. | **Critical Security Breach**: Intrusion goes undetected, allowing attackers inside the perimeter. |

> **Presentation Tip:** In cybersecurity, minimizing **False Negatives (FN)** is usually prioritized over minimizing False Positives, because the cost of a missed breach is far greater than the cost of a false alarm.

---

## 🗣️ 3-Minute Group Presentation Script

Use this script to divide speaking roles and guide your demonstration during academic presentations:

- **Speaker 1 (Introduction & Problem Statement - 45s):**
  > "Hello everyone. Today we are presenting our Machine Learning Intrusion Detection System, **SentinelShield**. In modern enterprise networks, traditional signature-based firewalls struggle to detect novel or high-frequency zero-day attacks. Our project applies supervised machine learning to classify network connection records in real-time."

- **Speaker 2 (Data Preprocessing & Model Architecture - 45s):**
  > "We simulated realistic network connection records with features like duration, protocol type, status flags, failed logins, and connection counts. Using Scikit-Learn pipelines, we scaled numeric metrics using `StandardScaler` and encoded protocol flags with `OneHotEncoder`. We compared a Random Forest Classifier against Logistic Regression, and Random Forest achieved a top F1-score of 0.9988."

- **Speaker 3 (Live Demonstration & Conclusion - 90s):**
  > *(Switch to browser showing Streamlit app)*
  > - *"Here is our web interface. Under the **Live Packet Inspector**, we can click 'SYN Flood' preset and click 'Detect Activity'. The model immediately flags this as an attack with >99% confidence, highlighting the abnormal connection burst and S0 flag."*
  > - *"Now we click 'Normal Web Browsing' — the system confirms safe benign traffic in green."*
  > - *"Next, under **Batch Traffic Simulator**, we can load a batch of 100 packets and see instant visual threat analytics with a donut chart and downloadable classified log."*
  > - *"Thank you! We'd be glad to take any questions."*

---

## 📜 License & Credits

Developed as an educational and demonstration project for Machine Learning in Network Security. Built with Python, Streamlit, Scikit-Learn, and Plotly.
# ML_IDS_Project
# ML_IDS_Project
