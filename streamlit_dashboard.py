"""
Streamlit Dashboard for Hybrid Identity Monitoring & Deepfake-Resistant Verification
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import time

# Configure page
st.set_page_config(
    page_title="Identity Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE_URL = "http://localhost:8000"

def main():
    st.title("🛡️ Hybrid Identity Monitoring & Deepfake-Resistant Verification")
    st.markdown("**Bank of Baroda Hackathon Solution**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Video KYC", "Deepfake Detection", "Monitoring", "Alerts", "Deployment Status"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Video KYC":
        show_video_kyc()
    elif page == "Deepfake Detection":
        show_deepfake_detection()
    elif page == "Monitoring":
        show_monitoring()
    elif page == "Alerts":
        show_alerts()
    elif page == "Deployment Status":
        show_deployment_status()

def show_dashboard():
    st.header("📊 System Dashboard")
    
    # System Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Users", "1,234", "12%")
    
    with col2:
        st.metric("Verifications Today", "89", "5%")
    
    with col3:
        st.metric("Deepfake Detections", "3", "-2%")
    
    with col4:
        st.metric("System Health", "99.9%", "0.1%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Verification Trends")
        # Sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
        verification_data = pd.DataFrame({
            'Date': dates,
            'Video KYC': [45, 52, 48, 61, 55, 49, 57],
            'Document KYC': [23, 28, 25, 31, 29, 26, 30],
            'Biometric': [12, 15, 13, 18, 16, 14, 17]
        })
        
        fig = px.line(verification_data, x='Date', y=['Video KYC', 'Document KYC', 'Biometric'],
                     title="Daily Verification Counts")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Security Metrics")
        # Sample data
        security_data = pd.DataFrame({
            'Metric': ['Deepfake Resistance', 'Identity Accuracy', 'System Uptime', 'Response Time'],
            'Score': [98.5, 96.2, 99.9, 94.1]
        })
        
        fig = px.bar(security_data, x='Metric', y='Score', 
                     title="Security Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.subheader("Recent Activity")
    activity_data = pd.DataFrame({
        'Time': ['10:30 AM', '10:25 AM', '10:20 AM', '10:15 AM', '10:10 AM'],
        'User': ['user_001', 'user_002', 'user_003', 'user_004', 'user_005'],
        'Action': ['Video KYC Completed', 'Deepfake Detected', 'Identity Verified', 'Alert Resolved', 'Monitoring Started'],
        'Status': ['✅ Success', '⚠️ Warning', '✅ Success', '✅ Success', '✅ Success']
    })
    
    st.dataframe(activity_data, use_container_width=True)

def show_video_kyc():
    st.header("🎥 Video KYC Verification")
    
    st.markdown("### Upload Video for KYC Verification")
    
    # File upload
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        # User ID input
        user_id = st.text_input("User ID", value="user_001")
        
        # Document upload (optional)
        uploaded_document = st.file_uploader("Upload ID Document (Optional)", type=['jpg', 'jpeg', 'png'])
        
        if st.button("Process Video KYC"):
            with st.spinner("Processing video KYC..."):
                # Simulate processing
                time.sleep(3)
                
                # Display results
                st.success("Video KYC Processing Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Face Quality Score", "94.2%")
                
                with col2:
                    st.metric("Liveness Score", "87.5%")
                
                with col3:
                    st.metric("Overall Confidence", "91.8%")
                
                # Detailed results
                st.subheader("Detailed Analysis")
                
                analysis_data = pd.DataFrame({
                    'Check': ['Face Detection', 'Liveness Verification', 'Deepfake Detection', 'Document Consistency'],
                    'Score': [94.2, 87.5, 12.3, 89.1],
                    'Status': ['✅ Pass', '✅ Pass', '✅ Pass', '✅ Pass']
                })
                
                st.dataframe(analysis_data, use_container_width=True)

def show_deepfake_detection():
    st.header("🔍 Deepfake Detection")
    
    st.markdown("### Upload Video for Deepfake Analysis")
    
    # File upload
    uploaded_video = st.file_uploader("Choose a video file for analysis", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("Analyze for Deepfake"):
            with st.spinner("Analyzing video for deepfake..."):
                # Simulate analysis
                time.sleep(4)
                
                # Display results
                st.subheader("Deepfake Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    deepfake_score = 23.4  # Sample score
                    if deepfake_score < 30:
                        st.success(f"✅ **REAL VIDEO** - Deepfake Score: {deepfake_score}%")
                    elif deepfake_score < 70:
                        st.warning(f"⚠️ **SUSPICIOUS** - Deepfake Score: {deepfake_score}%")
                    else:
                        st.error(f"❌ **DEEPFAKE DETECTED** - Deepfake Score: {deepfake_score}%")
                
                with col2:
                    st.metric("Confidence Level", f"{100 - deepfake_score}%")
                
                # Detailed analysis
                st.subheader("Detailed Analysis")
                
                analysis_metrics = pd.DataFrame({
                    'Metric': ['Facial Consistency', 'Eye Blinking', 'Head Movement', 'Artifact Detection'],
                    'Score': [87.3, 92.1, 78.9, 15.2],
                    'Status': ['✅ Normal', '✅ Normal', '✅ Normal', '✅ Clean']
                })
                
                st.dataframe(analysis_metrics, use_container_width=True)
                
                # Visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = deepfake_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Deepfake Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)

def show_monitoring():
    st.header("👁️ Identity Monitoring")
    
    # User selection
    user_id = st.selectbox("Select User", ["user_001", "user_002", "user_003", "user_004", "user_005"])
    
    if st.button("Get Monitoring Data"):
        with st.spinner("Loading monitoring data..."):
            time.sleep(2)
            
            # Display monitoring dashboard
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Anomaly Score", "12.3%", "2.1%")
            
            with col2:
                st.metric("Risk Level", "Low", "0%")
            
            with col3:
                st.metric("Monitoring Status", "Active", "0%")
            
            # Activity timeline
            st.subheader("Activity Timeline")
            
            timeline_data = pd.DataFrame({
                'Time': pd.date_range(start='2024-01-01 09:00', periods=10, freq='30min'),
                'Activity': ['Login', 'Video KYC', 'Transaction', 'Logout', 'Login', 'Document Upload', 'Verification', 'Transaction', 'Logout', 'Login'],
                'Anomaly Score': [5.2, 8.1, 12.3, 3.4, 6.7, 15.2, 9.8, 11.5, 4.1, 7.3],
                'Status': ['Normal', 'Normal', 'Warning', 'Normal', 'Normal', 'Alert', 'Normal', 'Warning', 'Normal', 'Normal']
            })
            
            fig = px.scatter(timeline_data, x='Time', y='Anomaly Score', 
                           color='Status', size='Anomaly Score',
                           title="User Activity Anomaly Timeline")
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            st.subheader("Risk Assessment")
            
            risk_factors = pd.DataFrame({
                'Factor': ['Login Location', 'Device Type', 'Time Pattern', 'Transaction Amount', 'Biometric Confidence'],
                'Risk Score': [15, 25, 5, 35, 20],
                'Status': ['Low', 'Medium', 'Low', 'High', 'Medium']
            })
            
            fig = px.bar(risk_factors, x='Factor', y='Risk Score', color='Status',
                        title="Risk Factor Analysis")
            st.plotly_chart(fig, use_container_width=True)

def show_alerts():
    st.header("🚨 System Alerts")
    
    # Alert filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
    
    with col2:
        user_filter = st.selectbox("User", ["All", "user_001", "user_002", "user_003"])
    
    with col3:
        status_filter = st.selectbox("Status", ["All", "Open", "Resolved"])
    
    # Sample alerts data
    alerts_data = pd.DataFrame({
        'Time': ['10:30 AM', '10:25 AM', '10:20 AM', '10:15 AM', '10:10 AM'],
        'Type': ['Deepfake Detected', 'High Risk User', 'Anomaly Detected', 'System Error', 'Low Confidence'],
        'Severity': ['Critical', 'High', 'Medium', 'High', 'Low'],
        'User': ['user_001', 'user_002', 'user_003', 'system', 'user_004'],
        'Message': ['Deepfake detected in video KYC', 'High risk behavior detected', 'Unusual login pattern', 'Database connection failed', 'Low biometric confidence'],
        'Status': ['Open', 'Resolved', 'Open', 'Resolved', 'Open']
    })
    
    # Filter data
    if severity_filter != "All":
        alerts_data = alerts_data[alerts_data['Severity'] == severity_filter]
    
    if user_filter != "All":
        alerts_data = alerts_data[alerts_data['User'] == user_filter]
    
    if status_filter != "All":
        status_map = {"Open": "Open", "Resolved": "Resolved"}
        alerts_data = alerts_data[alerts_data['Status'] == status_map[status_filter]]
    
    # Display alerts
    st.dataframe(alerts_data, use_container_width=True)
    
    # Alert statistics
    st.subheader("Alert Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alerts_data))
    
    with col2:
        critical_count = len(alerts_data[alerts_data['Severity'] == 'Critical'])
        st.metric("Critical", critical_count)
    
    with col3:
        open_count = len(alerts_data[alerts_data['Status'] == 'Open'])
        st.metric("Open", open_count)
    
    with col4:
        resolved_count = len(alerts_data[alerts_data['Status'] == 'Resolved'])
        st.metric("Resolved", resolved_count)

def show_deployment_status():
    st.header("🏗️ Deployment Status")
    
    # Deployment overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Deployment Mode", "Hybrid", "0%")
    
    with col2:
        st.metric("On-Premises Status", "Active", "0%")
    
    with col3:
        st.metric("Cloud Status", "Active", "0%")
    
    # Endpoint status
    st.subheader("Endpoint Status")
    
    endpoints_data = pd.DataFrame({
        'Endpoint': ['On-Premises API', 'Cloud API', 'Database', 'Redis Cache', 'Monitoring Service'],
        'Status': ['Active', 'Active', 'Active', 'Active', 'Active'],
        'Response Time': ['45ms', '120ms', '12ms', '2ms', '78ms'],
        'Uptime': ['99.9%', '99.8%', '100%', '100%', '99.7%']
    })
    
    st.dataframe(endpoints_data, use_container_width=True)
    
    # System metrics
    st.subheader("System Metrics")
    
    metrics_data = pd.DataFrame({
        'Metric': ['CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O', 'Database Connections'],
        'Current': [45, 67, 23, 156, 12],
        'Peak': [78, 89, 45, 234, 25],
        'Average': [52, 71, 28, 178, 15]
    })
    
    fig = px.bar(metrics_data, x='Metric', y=['Current', 'Peak', 'Average'],
                title="System Resource Usage", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Sync status
    st.subheader("Data Synchronization")
    
    sync_data = pd.DataFrame({
        'Time': pd.date_range(start='2024-01-01 00:00', periods=24, freq='H'),
        'Records Synced': [45, 67, 23, 89, 156, 234, 178, 123, 89, 67, 45, 78, 156, 234, 189, 145, 98, 67, 45, 78, 123, 156, 189, 234],
        'Sync Status': ['Success'] * 24
    })
    
    fig = px.line(sync_data, x='Time', y='Records Synced',
                 title="Data Synchronization Over Time")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
