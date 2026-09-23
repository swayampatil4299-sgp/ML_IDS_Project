"""
generate_data.py
Synthetic Network Traffic Dataset Generator for Intrusion Detection System (IDS).

Simulates realistic network connection records with normal traffic and
common cyberattack patterns:
1. Normal Traffic: Web (HTTP/HTTPS), DNS, SMTP, SSH, video/download bursts.
2. SYN Flood (DoS): High connection burst, TCP, S0 flags, minimal bytes.
3. Brute Force: Failed logins, SSH/FTP/Telnet services, unauthenticated.
4. Port Scan / Recon: Quick probe packets across diverse ports.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_ids_data(num_records: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a DataFrame of synthetic network packet/connection records with
    realistic distribution and subtle real-world variance.
    """
    np.random.seed(random_seed)
    
    records = []
    
    # Proportions: ~65% Normal, ~15% SYN Flood, ~10% Brute Force, ~10% Port Scan
    n_normal = int(num_records * 0.65)
    n_syn_flood = int(num_records * 0.15)
    n_brute_force = int(num_records * 0.10)
    n_port_scan = num_records - n_normal - n_syn_flood - n_brute_force
    
    # 1. Normal Traffic
    for _ in range(n_normal):
        protocol = np.random.choice(['tcp', 'udp', 'icmp'], p=[0.72, 0.24, 0.04])
        if protocol == 'tcp':
            service = np.random.choice(['http', 'smtp', 'ssh', 'ftp', 'other'], p=[0.62, 0.18, 0.10, 0.05, 0.05])
            # Normal flags are mostly SF (normal termination), with rare RST/REJ
            flag = np.random.choice(['SF', 'REJ', 'RSTO'], p=[0.93, 0.05, 0.02])
            logged_in = 1 if service in ['http', 'smtp', 'ssh', 'ftp'] and np.random.rand() > 0.12 else 0
        elif protocol == 'udp':
            service = np.random.choice(['dns', 'other'], p=[0.88, 0.12])
            flag = 'SF'
            logged_in = 0
        else: # icmp
            service = 'other'
            flag = 'SF'
            logged_in = 0
            
        duration = int(np.random.exponential(scale=15))
        src_bytes = int(np.clip(np.random.exponential(scale=650), 30, 25000))
        dst_bytes = int(np.clip(np.random.exponential(scale=2400), 0, 75000))
        # Normal users occasionally mistype credentials (1 failed login, rarely 2)
        failed_logins = int(np.random.choice([0, 1, 2], p=[0.96, 0.035, 0.005]))
        # Normal traffic can have bursts (e.g., page with 40 assets or streaming video)
        traffic_count = int(np.clip(np.random.exponential(scale=16) + 1, 1, 95))
        
        records.append({
            'duration': duration,
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'failed_logins': failed_logins,
            'traffic_count': traffic_count,
            'logged_in': logged_in,
            'label': 'Normal',
            'attack_type': 'Normal'
        })
        
    # 2. SYN Flood / DoS Attacks
    for _ in range(n_syn_flood):
        protocol = 'tcp'
        service = np.random.choice(['http', 'ssh', 'smtp', 'other'], p=[0.60, 0.15, 0.15, 0.10])
        flag = np.random.choice(['S0', 'RSTO', 'REJ'], p=[0.85, 0.10, 0.05])
        duration = int(np.random.choice([0, 0, 0, 1]))
        src_bytes = int(np.random.choice([0, 32, 40, 64, 128]))
        dst_bytes = int(np.random.choice([0, 0, 0, 48]))
        failed_logins = 0
        # High connection burst to the same host
        traffic_count = int(np.clip(np.random.normal(loc=310, scale=70), 90, 500))
        logged_in = 0
        
        records.append({
            'duration': duration,
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'failed_logins': failed_logins,
            'traffic_count': traffic_count,
            'logged_in': logged_in,
            'label': 'Intrusion/Attack',
            'attack_type': 'SYN Flood'
        })
        
    # 3. Brute Force Attacks (SSH / FTP / Telnet)
    for _ in range(n_brute_force):
        protocol = 'tcp'
        service = np.random.choice(['ssh', 'ftp', 'telnet'], p=[0.55, 0.30, 0.15])
        flag = np.random.choice(['SF', 'RSTO', 'REJ'], p=[0.55, 0.25, 0.20])
        duration = int(np.random.uniform(1, 40))
        src_bytes = int(np.random.uniform(120, 850))
        dst_bytes = int(np.random.uniform(40, 450))
        # Repeated failed authentication attempts
        failed_logins = int(np.random.choice([2, 3, 4, 5, 6], p=[0.20, 0.35, 0.25, 0.15, 0.05]))
        traffic_count = int(np.clip(np.random.poisson(lam=28), 4, 110))
        logged_in = 0
        
        records.append({
            'duration': duration,
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'failed_logins': failed_logins,
            'traffic_count': traffic_count,
            'logged_in': logged_in,
            'label': 'Intrusion/Attack',
            'attack_type': 'Brute Force'
        })
        
    # 4. Port Scan / Probing
    for _ in range(n_port_scan):
        protocol = np.random.choice(['tcp', 'udp', 'icmp'], p=[0.68, 0.22, 0.10])
        service = np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'other'])
        flag = np.random.choice(['REJ', 'S0', 'RSTR', 'SH', 'SF'], p=[0.50, 0.25, 0.12, 0.08, 0.05])
        duration = int(np.random.choice([0, 1, 2]))
        src_bytes = int(np.random.choice([0, 20, 44, 60, 120]))
        dst_bytes = int(np.random.choice([0, 0, 20]))
        failed_logins = 0
        traffic_count = int(np.clip(np.random.normal(loc=170, scale=55), 45, 420))
        logged_in = 0
        
        records.append({
            'duration': duration,
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'failed_logins': failed_logins,
            'traffic_count': traffic_count,
            'logged_in': logged_in,
            'label': 'Intrusion/Attack',
            'attack_type': 'Port Scan'
        })
        
    df = pd.DataFrame(records)
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating training dataset (6,000 records)...")
    train_df = generate_synthetic_ids_data(num_records=6000, random_seed=42)
    train_file = os.path.join(data_dir, 'network_traffic.csv')
    train_df.to_csv(train_file, index=False)
    print(f"Saved training data to: {train_file}")
    print(f"Class distribution:\n{train_df['label'].value_counts(normalize=True).round(3)}")
    print(f"Attack types breakdown:\n{train_df['attack_type'].value_counts()}\n")
    
    print("Generating sample batch test dataset (100 records)...")
    batch_df = generate_synthetic_ids_data(num_records=100, random_seed=999)
    batch_file = os.path.join(data_dir, 'sample_batch_test.csv')
    batch_df.to_csv(batch_file, index=False)
    print(f"Saved sample batch test data to: {batch_file}")

if __name__ == '__main__':
    main()
