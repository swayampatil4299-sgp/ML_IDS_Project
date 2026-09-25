"""
app.py
Streamlit Web Application for Machine Learning Intrusion Detection System (IDS).

Features:
1. Model Performance Dashboard & Confusion Matrix.
2. Real-Time Packet Inspector with Live Input Form and 1-Click Attack Presets.
3. Batch Traffic Simulation & Log Analyzer with CSV upload, metrics & charts.
4. Presentation and Educational Cybersecurity Guide.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SentinelShield IDS | ML Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN CYBERSECURITY UI ---
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #00d2ff;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Status Alert Badges */
    .alert-normal {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        color: #10b981;
        text-align: center;
    }
    .alert-attack {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        color: #ef4444;
        text-align: center;
    }
    
    /* Subtle glow for titles */
    .shield-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_ids_payload():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'ids_model_payload.joblib')
    if not os.path.exists(model_path):
        st.error(f"Model payload not found at `{model_path}`. Please run `train_model.py` first.")
        return None
    return joblib.load(model_path)

payload = load_ids_payload()

if payload is None:
    st.stop()

best_model_name = payload['best_model_name']
models_dict = payload['models']
categorical_options = payload['categorical_options']
feature_cols = payload['feature_cols']

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## 🛡️ **SentinelShield IDS**")
    st.caption("Machine Learning Network Guard")
    st.divider()

    st.subheader("⚙️ Model Configuration")
    model_choice = st.selectbox(
        "Active Inference Engine:",
        options=list(models_dict.keys()),
        index=0 if best_model_name == "Random Forest" else 1,
        help="Select which trained classification pipeline to use for real-time and batch evaluations."
    )
    
    selected_model_data = models_dict[model_choice]
    active_pipeline = selected_model_data['pipeline']
    active_metrics = selected_model_data['metrics']

    st.markdown(f"**Selected:** `{model_choice}`")
    if model_choice == best_model_name:
        st.success("⭐ Best Performer (Recommended)")
    else:
        st.info("ℹ️ Linear Baseline Model")

    st.markdown(f"""
    - **Accuracy:** `{active_metrics['accuracy']*100:.2f}%`
    - **Precision:** `{active_metrics['precision']*100:.2f}%`
    - **Recall:** `{active_metrics['recall']*100:.2f}%`
    - **F1-Score:** `{active_metrics['f1']*100:.2f}%`
    """)

    st.divider()
    st.markdown("### 📋 Navigation")
    app_mode = st.radio(
        "Go to Module:",
        [
            "⚡ Live Packet Inspector",
            "📁 Batch Traffic Simulator",
            "📊 Model Performance & Metrics",
            "📖 Presentation & IDS Guide"
        ]
    )

    st.divider()
    st.caption("Developed for Academic IDS Demonstration")
    st.caption("Dataset: Simulated Network Traffic (NSL-KDD Inspired)")


# ==============================================================================
# MODULE 1: LIVE PACKET INSPECTOR (MANUAL INPUT FORM)
# ==============================================================================
if app_mode == "⚡ Live Packet Inspector":
    st.markdown('<div class="shield-title">⚡ Real-Time Network Packet Inspector</div>', unsafe_allow_html=True)
    st.markdown("Simulate individual connection packets or select an attack preset to test real-time threat classification.")
    st.write("")

    # Presets Section
    st.subheader("🎯 Quick Attack & Traffic Presets")
    st.caption("Click any preset below to auto-populate the packet parameters for demonstration:")

    # Initialize preset values in session_state if not set
    if "preset_params" not in st.session_state:
        st.session_state.preset_params = {
            'duration': 12,
            'protocol_type': 'tcp',
            'service': 'http',
            'flag': 'SF',
            'src_bytes': 450,
            'dst_bytes': 1800,
            'failed_logins': 0,
            'traffic_count': 14,
            'logged_in': 1
        }

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🟢 Normal Web Browsing", use_container_width=True):
            st.session_state.preset_params = {
                'duration': 15,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'SF',
                'src_bytes': 520,
                'dst_bytes': 3200,
                'failed_logins': 0,
                'traffic_count': 12,
                'logged_in': 1
            }
            st.rerun()

    with p_col2:
        if st.button("🔴 SYN Flood (DoS)", use_container_width=True):
            st.session_state.preset_params = {
                'duration': 0,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'S0',
                'src_bytes': 40,
                'dst_bytes': 0,
                'failed_logins': 0,
                'traffic_count': 380,
                'logged_in': 0
            }
            st.rerun()

    with p_col3:
        if st.button("🔴 SSH Brute Force", use_container_width=True):
            st.session_state.preset_params = {
                'duration': 18,
                'protocol_type': 'tcp',
                'service': 'ssh',
                'flag': 'SF',
                'src_bytes': 340,
                'dst_bytes': 120,
                'failed_logins': 4,
                'traffic_count': 35,
                'logged_in': 0
            }
            st.rerun()

    with p_col4:
        if st.button("🔴 Port Scan Recon", use_container_width=True):
            st.session_state.preset_params = {
                'duration': 0,
                'protocol_type': 'tcp',
                'service': 'other',
                'flag': 'REJ',
                'src_bytes': 0,
                'dst_bytes': 0,
                'failed_logins': 0,
                'traffic_count': 220,
                'logged_in': 0
            }
            st.rerun()

    st.write("")

    # Live Input Form
    with st.form("packet_inspection_form"):
        st.subheader("📝 Packet Parameters Form")
        params = st.session_state.preset_params

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            protocol = st.selectbox(
                "Protocol Type",
                options=categorical_options['protocol_type'],
                index=categorical_options['protocol_type'].index(params['protocol_type'])
            )
            service = st.selectbox(
                "Network Service",
                options=categorical_options['service'],
                index=categorical_options['service'].index(params['service'])
            )
            flag = st.selectbox(
                "Connection Status Flag",
                options=categorical_options['flag'],
                index=categorical_options['flag'].index(params['flag']),
                help="SF = Normal finish, S0 = SYN without reply, REJ = Rejected, RSTO = Reset connection"
            )

        with col_b:
            duration = st.number_input(
                "Duration (seconds)",
                min_value=0,
                max_value=3600,
                value=int(params['duration'])
            )
            src_bytes = st.number_input(
                "Source Bytes (Sent)",
                min_value=0,
                max_value=100000,
                value=int(params['src_bytes'])
            )
            dst_bytes = st.number_input(
                "Destination Bytes (Received)",
                min_value=0,
                max_value=100000,
                value=int(params['dst_bytes'])
            )

        with col_c:
            traffic_count = st.slider(
                "Traffic Count (Connections in 2s)",
                min_value=1,
                max_value=500,
                value=int(params['traffic_count']),
                help="High counts indicate potential scanning or Denial-of-Service floods."
            )
            failed_logins = st.slider(
                "Failed Login Attempts",
                min_value=0,
                max_value=10,
                value=int(params['failed_logins']),
                help="Repeated failed logins indicate unauthorized credential stuffing or brute forcing."
            )
            logged_in_val = st.radio(
                "Authentication Status",
                options=["Authenticated (1)", "Unauthenticated (0)"],
                index=0 if params['logged_in'] == 1 else 1
            )
            logged_in = 1 if "Authenticated (1)" in logged_in_val else 0

        st.write("")
        submit_button = st.form_submit_button("🔍 Detect Activity", type="primary", use_container_width=True)

    if submit_button:
        # Construct input DataFrame
        input_data = pd.DataFrame([{
            'duration': duration,
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'failed_logins': failed_logins,
            'traffic_count': traffic_count,
            'logged_in': logged_in
        }])

        prediction = active_pipeline.predict(input_data)[0]
        probabilities = active_pipeline.predict_proba(input_data)[0]
        classes = list(active_pipeline.classes_)
        
        normal_idx = classes.index('Normal')
        attack_idx = classes.index('Intrusion/Attack')
        normal_prob = probabilities[normal_idx]
        attack_prob = probabilities[attack_idx]

        st.divider()
        st.subheader("🎯 Real-Time Detection Result")

        res_col1, res_col2 = st.columns([1.5, 1])

        with res_col1:
            if prediction == 'Normal':
                st.markdown(f"""
                <div class="alert-normal">
                    <h2 style="margin: 0; color: #10b981;">🟢 NORMAL NETWORK ACTIVITY</h2>
                    <p style="margin-top: 8px; font-size: 16px; color: #d1fae5;">
                        No threat detected. Connection matches typical benign network behavior.
                    </p>
                    <h3 style="margin: 12px 0 0 0; color: #10b981;">Confidence: {normal_prob*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-attack">
                    <h2 style="margin: 0; color: #ef4444;">🚨 INTRUSION / ATTACK DETECTED!</h2>
                    <p style="margin-top: 8px; font-size: 16px; color: #fee2e2;">
                        High threat level! Packet characteristics deviate strongly from normal baselines.
                    </p>
                    <h3 style="margin: 12px 0 0 0; color: #ef4444;">Threat Probability: {attack_prob*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            # Heuristic Risk Explanation
            st.write("")
            st.markdown("#### 🔎 Threat Vector Analysis:")
            risk_factors = []
            if traffic_count >= 100:
                risk_factors.append(f"⚠️ **High Connection Burst Rate**: `traffic_count = {traffic_count}` (Typical DoS / Port Scan threshold)")
            if flag in ['S0', 'REJ', 'RSTO']:
                risk_factors.append(f"⚠️ **Abnormal TCP Status Flag**: `flag = {flag}` (Incomplete handshake or forced connection rejection)")
            if failed_logins >= 2:
                risk_factors.append(f"⚠️ **Excessive Failed Authentications**: `failed_logins = {failed_logins}` (Brute Force credential attack pattern)")
            if duration == 0 and traffic_count > 60:
                risk_factors.append(f"⚠️ **Zero-Duration Rapid Packets**: Multiple 0-second connections to target host")
            if logged_in == 0 and service in ['ssh', 'ftp', 'telnet'] and failed_logins > 0:
                risk_factors.append(f"⚠️ **Unauthenticated Administrative Access**: Access attempt on {service.upper()} without valid login")

            if risk_factors:
                for rf in risk_factors:
                    st.warning(rf)
            else:
                st.success("✅ All network parameters reside within acceptable baseline distributions.")

        with res_col2:
            st.markdown("#### 📊 Confidence Breakdown")
            prob_df = pd.DataFrame({
                'Class': ['Normal', 'Intrusion'],
                'Probability': [normal_prob, attack_prob]
            })
            fig_prob = px.bar(
                prob_df,
                x='Probability',
                y='Class',
                orientation='h',
                color='Class',
                color_discrete_map={'Normal': '#10b981', 'Intrusion': '#ef4444'},
                range_x=[0, 1],
                text=prob_df['Probability'].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_prob.update_layout(
                showlegend=False,
                height=220,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            with st.expander("🔍 View Raw Transmitted Payload"):
                st.json(input_data.to_dict(orient='records')[0])


# ==============================================================================
# MODULE 2: BATCH TRAFFIC SIMULATION & LOG ANALYZER
# ==============================================================================
elif app_mode == "📁 Batch Traffic Simulator":
    st.markdown('<div class="shield-title">📁 Batch Traffic Log Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Upload network log CSV files or run the built-in batch test set to simulate high-throughput traffic inspection.")
    st.write("")

    batch_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_batch_test.csv')

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        uploaded_file = st.file_uploader(
            "Upload Network Traffic Logs (.csv):",
            type=['csv'],
            help="CSV must contain network features: duration, protocol_type, service, flag, src_bytes, dst_bytes, failed_logins, traffic_count, logged_in."
        )
    with col_b2:
        st.write("")
        st.write("")
        use_sample = st.button("📦 Load Built-in Demo Logs (100 Packets)", type="secondary", use_container_width=True)

    df_to_analyze = None

    if uploaded_file is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` containing {len(df_to_analyze)} records.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample:
        if os.path.exists(batch_file_path):
            df_to_analyze = pd.read_csv(batch_file_path)
            st.info(f"Loaded built-in test batch (`sample_batch_test.csv`) with {len(df_to_analyze)} records.")
        else:
            st.error("Sample batch test file not found. Please run `generate_data.py`.")

    if df_to_analyze is not None:
        # Check if required columns are present
        missing_cols = [c for c in feature_cols if c not in df_to_analyze.columns]
        if missing_cols:
            st.error(f"Missing required columns in dataset: {missing_cols}")
        else:
            with st.spinner("Processing network traffic with ML classification pipeline..."):
                X_batch = df_to_analyze[feature_cols]
                predictions = active_pipeline.predict(X_batch)
                probabilities = active_pipeline.predict_proba(X_batch)
                
                classes = list(active_pipeline.classes_)
                attack_idx = classes.index('Intrusion/Attack')
                threat_scores = probabilities[:, attack_idx]

                results_df = df_to_analyze.copy()
                results_df['Predicted_Status'] = predictions
                results_df['Threat_Score'] = (threat_scores * 100).round(2)

            total_records = len(results_df)
            normal_count = int((results_df['Predicted_Status'] == 'Normal').sum())
            intrusion_count = int((results_df['Predicted_Status'] == 'Intrusion/Attack').sum())
            intrusion_rate = (intrusion_count / total_records) * 100 if total_records > 0 else 0

            st.divider()
            st.subheader("📊 Batch Inspection Summary")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Packets</div>
                    <div class="metric-value">{total_records:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Normal Traffic</div>
                    <div class="metric-value" style="color: #10b981;">{normal_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Intrusions Intercepted</div>
                    <div class="metric-value" style="color: #ef4444;">{intrusion_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Threat Percentage</div>
                    <div class="metric-value" style="color: {'#ef4444' if intrusion_rate > 10 else '#10b981'};">{intrusion_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            # Visual Summaries: Donut Chart & Protocol/Attack Breakdown
            chart_col1, chart_col2 = st.columns([1, 1])

            with chart_col1:
                st.markdown("#### 🍩 Traffic Classification Distribution")
                pie_df = pd.DataFrame({
                    'Status': ['Normal', 'Intrusion/Attack'],
                    'Count': [normal_count, intrusion_count]
                })
                fig_pie = px.pie(
                    pie_df,
                    names='Status',
                    values='Count',
                    hole=0.55,
                    color='Status',
                    color_discrete_map={'Normal': '#10b981', 'Intrusion/Attack': '#ef4444'}
                )
                fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                st.markdown("#### 🚨 Intrusions by Status Flag")
                attack_df = results_df[results_df['Predicted_Status'] == 'Intrusion/Attack']
                if not attack_df.empty:
                    flag_counts = attack_df['flag'].value_counts().reset_index()
                    flag_counts.columns = ['Flag', 'Attacks']
                    fig_flags = px.bar(
                        flag_counts,
                        x='Flag',
                        y='Attacks',
                        color='Flag',
                        color_discrete_sequence=px.colors.sequential.Reds_r,
                        text='Attacks'
                    )
                    fig_flags.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300, showlegend=False)
                    st.plotly_chart(fig_flags, use_container_width=True)
                else:
                    st.info("No intrusions detected in this batch.")

            # Filterable Table
            st.divider()
            st.subheader("📋 Detailed Classified Packet Logs")
            
            filter_choice = st.radio(
                "Filter Log View:",
                ["All Packets", "Intrusions Only (Threats)", "Normal Only (Benign)"],
                horizontal=True
            )

            if filter_choice == "Intrusions Only (Threats)":
                display_df = results_df[results_df['Predicted_Status'] == 'Intrusion/Attack']
            elif filter_choice == "Normal Only (Benign)":
                display_df = results_df[results_df['Predicted_Status'] == 'Normal']
            else:
                display_df = results_df

            styler = display_df.style
            if hasattr(styler, 'map'):
                styler = styler.map(
                    lambda v: 'background-color: rgba(239, 68, 68, 0.2); color: #f87171;' if v == 'Intrusion/Attack' 
                    else ('background-color: rgba(16, 185, 129, 0.2); color: #34d399;' if v == 'Normal' else ''),
                    subset=['Predicted_Status']
                )
            else:
                styler = styler.applymap(
                    lambda v: 'background-color: rgba(239, 68, 68, 0.2); color: #f87171;' if v == 'Intrusion/Attack' 
                    else ('background-color: rgba(16, 185, 129, 0.2); color: #34d399;' if v == 'Normal' else ''),
                    subset=['Predicted_Status']
                )

            st.dataframe(
                styler,
                use_container_width=True,
                height=350
            )

            # Download CSV Button
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Classified Results CSV",
                data=csv_export,
                file_name="classified_ids_traffic_report.csv",
                mime="text/csv"
            )


# ==============================================================================
# MODULE 3: MODEL PERFORMANCE & METRICS DASHBOARD
# ==============================================================================
elif app_mode == "📊 Model Performance & Metrics":
    st.markdown('<div class="shield-title">📊 Model Evaluation & Comparison Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive performance metrics, confusion matrices, and feature importances for both trained classifiers.")
    st.write("")

    # Side-by-Side Model Comparison Cards
    st.subheader("⚖️ Algorithm Comparison on Test Set (Stratified 20%)")
    
    comp_col1, comp_col2 = st.columns(2)

    rf_metrics = models_dict['Random Forest']['metrics']
    lr_metrics = models_dict['Logistic Regression']['metrics']

    with comp_col1:
        st.markdown("### 🌲 Random Forest Classifier")
        if best_model_name == "Random Forest":
            if hasattr(st, 'badge'):
                st.badge("Best Performer", icon="⭐", color="green")
            else:
                st.success("⭐ Best Performer")
        st.markdown(f"""
        - **Accuracy:** `{rf_metrics['accuracy']*100:.2f}%`
        - **Precision (Attack):** `{rf_metrics['precision']*100:.2f}%`
        - **Recall (Attack):** `{rf_metrics['recall']*100:.2f}%`
        - **F1-Score:** `{rf_metrics['f1']*100:.2f}%`
        """)

    with comp_col2:
        st.markdown("### 📈 Logistic Regression")
        if best_model_name == "Logistic Regression":
            if hasattr(st, 'badge'):
                st.badge("Best Performer", icon="⭐", color="green")
            else:
                st.success("⭐ Best Performer")
        st.markdown(f"""
        - **Accuracy:** `{lr_metrics['accuracy']*100:.2f}%`
        - **Precision (Attack):** `{lr_metrics['precision']*100:.2f}%`
        - **Recall (Attack):** `{lr_metrics['recall']*100:.2f}%`
        - **F1-Score:** `{lr_metrics['f1']*100:.2f}%`
        """)

    st.divider()

    # Confusion Matrix Visualizer
    st.subheader("🧩 Confusion Matrix Analysis")
    st.caption("A confusion matrix reveals how many benign packets and attacks were correctly or falsely classified.")

    inspect_model = st.radio(
        "Select Model to Inspect Confusion Matrix:",
        options=["Random Forest", "Logistic Regression"],
        horizontal=True
    )

    cm_data = np.array(models_dict[inspect_model]['cm'])
    # cm_data: [[TN, FP], [FN, TP]]
    tn, fp = cm_data[0]
    fn, tp = cm_data[1]

    cm_col1, cm_col2 = st.columns([1.2, 1])

    with cm_col1:
        # Plotly Heatmap
        labels_x = ['Pred: Normal', 'Pred: Intrusion']
        labels_y = ['Actual: Normal', 'Actual: Intrusion']
        
        annotations_text = [
            [f"True Negative (TN)<br><b>{tn}</b>", f"False Positive (FP)<br><b>{fp}</b>"],
            [f"False Negative (FN)<br><b>{fn}</b>", f"True Positive (TP)<br><b>{tp}</b>"]
        ]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=labels_x,
            y=labels_y,
            colorscale='Blues',
            showscale=False,
            text=annotations_text,
            texttemplate="%{text}",
            hoverinfo='text'
        ))
        fig_cm.update_layout(
            height=320,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14), autorange="reversed")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_col2:
        st.markdown("#### 💡 Metric Interpretation:")
        st.markdown(f"""
        - **True Negatives (TN = {tn}):** Benign packets correctly verified as safe.
        - **True Positives (TP = {tp}):** Malicious attacks successfully detected and intercepted!
        - **False Positives (FP = {fp}):** False alarms (benign packets misclassified as attacks).
        - **False Negatives (FN = {fn}):** Critical misses (attacks mistakenly classified as normal).
        
        > **Cybersecurity Note:** In intrusion detection, minimizing **False Negatives (FN)** is vital because an unintercepted attack can compromise the entire infrastructure.
        """)

    # Feature Importance Section
    st.divider()
    st.subheader("🔑 Top Discriminative Features (Feature Importance)")
    st.caption("Highlights which packet attributes have the highest predictive power in spotting intrusions.")

    feat_imp_dict = models_dict['Random Forest']['feature_importance']
    if feat_imp_dict:
        feat_df = pd.DataFrame(list(feat_imp_dict.items()), columns=['Feature', 'Importance'])
        feat_df = feat_df.sort_values(by='Importance', ascending=True)

        fig_imp = px.bar(
            feat_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='teal',
            text=feat_df['Importance'].apply(lambda x: f"{x:.3f}")
        )
        fig_imp.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# MODULE 4: PRESENTATION & EDUCATIONAL GUIDE
# ==============================================================================
elif app_mode == "📖 Presentation & IDS Guide":
    st.markdown('<div class="shield-title">📖 Cybersecurity Presentation Guide</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive reference notes, conceptual diagrams, and presentation talking points for your project group.")
    st.write("")

    exp1 = st.expander("🛡️ 1. What is an Intrusion Detection System (IDS)?", expanded=True)
    with exp1:
        st.markdown("""
        An **Intrusion Detection System (IDS)** monitors network traffic for malicious activity or policy violations:
        - **Signature-Based IDS (e.g. Snort):** Compares network packet signatures against a known database of attack signatures. Cannot detect zero-day exploits.
        - **Anomaly-Based IDS:** Establishes a baseline profile of normal network traffic and flags statistical deviations.
        - **Machine Learning-Based IDS (Our System):** Learns non-linear representations across multi-dimensional packet telemetry to classify unseen zero-day attacks with high accuracy.
        """)

    exp2 = st.expander("📊 2. Evaluation Metrics in Cybersecurity Context", expanded=True)
    with exp2:
        st.markdown("""
        | Metric | Mathematical Formula | Cybersecurity Relevance |
        | :--- | :--- | :--- |
        | **Accuracy** | $(TP + TN) / Total$ | Overall system correctness across both benign and malicious traffic. |
        | **Precision** | $TP / (TP + FP)$ | How many flagged alerts were actually real attacks (low precision = alert fatigue). |
        | **Recall** | $TP / (TP + FN)$ | How many real attacks were successfully caught (low recall = missed breaches). |
        | **F1-Score** | $2 \\cdot \\frac{P \\cdot R}{P + R}$ | Harmonic balance between Precision and Recall. |
        """)

    exp3 = st.expander("🗣️ 3. Recommended 3-Minute Presentation Walkthrough", expanded=True)
    with exp3:
        st.markdown("""
        1. **Slide / Introduction (30s):**
           - "Hello everyone, today we present our Machine Learning Intrusion Detection System (IDS), built to safeguard networks against cyberattacks like SYN Floods, Port Scans, and Brute Force attempts."
        2. **Machine Learning Pipeline (60s):**
           - "We simulated realistic packet telemetry including packet byte counts, protocol types, TCP connection flags, failed logins, and traffic bursts."
           - "We preprocessed the data using `StandardScaler` for numeric features and `OneHotEncoder` for categorical flags, training both a **Random Forest Classifier** and **Logistic Regression**."
           - "Random Forest achieved superior F1-score due to its ability to capture non-linear interactions (e.g. high traffic count + S0 flag)."
        3. **Live Demonstration (60s):**
           - Switch to the **⚡ Live Packet Inspector** tab.
           - Click the **SYN Flood** preset and demonstrate instant detection with >99% confidence.
           - Click the **Normal Web Browsing** preset and show benign classification.
           - Switch to **📁 Batch Traffic Simulator** and show 1-click batch classification of 100 packets with the donut chart.
        4. **Conclusion (30s):**
           - "Our system demonstrates that machine learning pipelines can be effectively deployed into user-friendly real-time defensive tools."
        """)
