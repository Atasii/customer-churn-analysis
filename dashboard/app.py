# dashboard/app.py (CORRECTED VERSION)
# Run with: streamlit run dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    import os
    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, 'dashboard_data.csv')
    df = pd.read_csv(file_path)
    return df

# Load data
df = load_data()

# ============================================
# FIX: Reconstruct original columns from one-hot encoded columns
# ============================================

# Reconstruct Contract column
contract_cols = [col for col in df.columns if col.startswith('Contract_')]
if contract_cols:
    df['Contract'] = df[contract_cols].idxmax(axis=1).str.replace('Contract_', '')

# Reconstruct PaymentMethod column
payment_cols = [col for col in df.columns if col.startswith('PaymentMethod_')]
if payment_cols:
    df['PaymentMethod'] = df[payment_cols].idxmax(axis=1).str.replace('PaymentMethod_', '')

# Reconstruct InternetService column
internet_cols = [col for col in df.columns if col.startswith('InternetService_')]
if internet_cols:
    df['InternetService'] = df[internet_cols].idxmax(axis=1).str.replace('InternetService_', '')

# Reconstruct Gender column
gender_cols = [col for col in df.columns if col.startswith('gender_')]
if gender_cols:
    df['gender'] = df[gender_cols].idxmax(axis=1).str.replace('gender_', '')

# Print column names for debugging (remove after testing)
print("Available columns in data:", df.columns.tolist())

# ============================================
# Calculate metrics
# ============================================

# Calculate key metrics
total_customers = len(df)
churned_customers = df['Churn'].sum() if 'Churn' in df.columns else int(len(df) * 0.27)
churn_rate = (churned_customers / total_customers) * 100

# Calculate risk scores if not present
if 'churn_probability' not in df.columns:
    df['churn_probability'] = df['MonthlyCharges'] / 150 + (1 - df['tenure']/100)
    df['churn_probability'] = df['churn_probability'].clip(0, 1)

# Risk categories
df['risk_category'] = pd.cut(df['churn_probability'], 
                              bins=[0, 0.3, 0.6, 1.0], 
                              labels=['Low Risk', 'Medium Risk', 'High Risk'])

# ============================================
# Sidebar Filters
# ============================================

st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

# Contract type filter
if 'Contract' in df.columns:
    contract_options = ['All'] + sorted(df['Contract'].unique().tolist())
    selected_contract = st.sidebar.selectbox("Contract Type", contract_options)
else:
    selected_contract = 'All'

# Payment method filter
if 'PaymentMethod' in df.columns:
    payment_options = ['All'] + sorted(df['PaymentMethod'].unique().tolist())
    selected_payment = st.sidebar.selectbox("Payment Method", payment_options)
else:
    selected_payment = 'All'

# Risk level filter
risk_options = ['All', 'High Risk', 'Medium Risk', 'Low Risk']
selected_risk = st.sidebar.selectbox("Risk Level", risk_options)

# Apply filters
filtered_df = df.copy()
if selected_contract != 'All' and 'Contract' in df.columns:
    filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
if selected_payment != 'All' and 'PaymentMethod' in df.columns:
    filtered_df = filtered_df[filtered_df['PaymentMethod'] == selected_payment]
if selected_risk != 'All':
    filtered_df = filtered_df[filtered_df['risk_category'] == selected_risk]

# ============================================
# Main Dashboard
# ============================================

# Main header
st.markdown('<h1 class="main-header">📊 Customer Churn & Retention Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Customers",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - total_customers}" if len(filtered_df) != total_customers else None
    )

with col2:
    filtered_churn = filtered_df['Churn'].sum() if 'Churn' in filtered_df.columns else int(len(filtered_df) * 0.27)
    filtered_churn_rate = (filtered_churn / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    st.metric(
        label="Churn Rate",
        value=f"{filtered_churn_rate:.1f}%",
        delta=f"{filtered_churn_rate - churn_rate:.1f}%" if len(filtered_df) != total_customers else None,
        delta_color="inverse"
    )

with col3:
    high_risk_in_filtered = len(filtered_df[filtered_df['risk_category'] == 'High Risk'])
    st.metric(
        label="High Risk Customers",
        value=f"{high_risk_in_filtered:,}",
        delta=f"{(high_risk_in_filtered/len(filtered_df)*100):.0f}% of filtered" if len(filtered_df) > 0 else None
    )

with col4:
    avg_monthly = filtered_df['MonthlyCharges'].mean()
    st.metric(
        label="Avg Monthly Charge",
        value=f"${avg_monthly:.2f}",
        delta=f"${avg_monthly - df['MonthlyCharges'].mean():.2f}" if len(filtered_df) != total_customers else None
    )

st.markdown("---")

# ============================================
# Row 2: Churn Drivers
# ============================================

st.markdown('<h2 class="sub-header">🎯 Key Churn Drivers</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Contract type analysis (if Contract column exists)
    if 'Contract' in df.columns and 'Churn' in df.columns:
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
        fig1 = go.Figure(data=[
            go.Bar(name='Stayed', x=contract_churn.index, y=contract_churn[0]),
            go.Bar(name='Churned', x=contract_churn.index, y=contract_churn[1])
        ])
        fig1.update_layout(
            title="Churn Rate by Contract Type",
            barmode='stack',
            xaxis_title="Contract Type",
            yaxis_title="Percentage (%)",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Contract type data not available in current format")

with col2:
    # Tenure analysis
    if 'tenure' in df.columns and 'Churn' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1 year', '1-2 years', '2-4 years', '4-6 years'])
        tenure_churn = pd.crosstab(df['tenure_group'], df['Churn'], normalize='index') * 100
        fig2 = go.Figure(data=[
            go.Bar(name='Stayed', x=tenure_churn.index, y=tenure_churn[0]),
            go.Bar(name='Churned', x=tenure_churn.index, y=tenure_churn[1])
        ])
        fig2.update_layout(
            title="Churn Rate by Customer Tenure",
            barmode='stack',
            xaxis_title="Tenure",
            yaxis_title="Percentage (%)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# Row 3: Risk Analysis
# ============================================

st.markdown('<h2 class="sub-header">⚠️ Risk Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Risk distribution pie chart
    risk_counts = df['risk_category'].value_counts()
    fig3 = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Customer Risk Distribution",
        color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c']
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Monthly charges vs risk
    fig4 = px.box(
        df,
        x='risk_category',
        y='MonthlyCharges',
        color='risk_category',
        title="Monthly Charges by Risk Level",
        color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c']
    )
    st.plotly_chart(fig4, use_container_width=True)

# ============================================
# Row 4: Additional Insights
# ============================================

st.markdown('<h2 class="sub-header">📈 Additional Insights</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Payment method analysis
    if 'PaymentMethod' in df.columns and 'Churn' in df.columns:
        payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=payment_churn.index, y=payment_churn[1], name='Churned', marker_color='#e74c3c'))
        fig5.add_trace(go.Bar(x=payment_churn.index, y=payment_churn[0], name='Stayed', marker_color='#2ecc71'))
        fig5.update_layout(
            title="Churn Rate by Payment Method",
            barmode='stack',
            xaxis_title="Payment Method",
            yaxis_title="Percentage (%)",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Payment method data not available in current format")

with col2:
    # Senior citizen analysis
    if 'SeniorCitizen' in df.columns and 'Churn' in df.columns:
        senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize='index') * 100
        senior_churn.index = ['Non-Senior', 'Senior']
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(x=senior_churn.index, y=senior_churn[1], name='Churned', marker_color='#e74c3c'))
        fig6.add_trace(go.Bar(x=senior_churn.index, y=senior_churn[0], name='Stayed', marker_color='#2ecc71'))
        fig6.update_layout(
            title="Churn Rate by Senior Citizen Status",
            barmode='stack',
            xaxis_title="Customer Type",
            yaxis_title="Percentage (%)",
            height=400
        )
        st.plotly_chart(fig6, use_container_width=True)

# ============================================
# Row 5: High Risk Customer Table
# ============================================

st.markdown('<h2 class="sub-header">🚨 High Risk Customers (Top 20)</h2>', unsafe_allow_html=True)

high_risk_customers = df[df['risk_category'] == 'High Risk'].nlargest(20, 'churn_probability')
display_cols = ['tenure', 'MonthlyCharges', 'churn_probability']
if 'Contract' in df.columns:
    display_cols.append('Contract')
if 'PaymentMethod' in df.columns:
    display_cols.append('PaymentMethod')

available_cols = [col for col in display_cols if col in high_risk_customers.columns]
high_risk_display = high_risk_customers[available_cols].copy()
high_risk_display['churn_probability'] = high_risk_display['churn_probability'].apply(lambda x: f"{x*100:.1f}%")

st.dataframe(high_risk_display, use_container_width=True)

# ============================================
# Row 6: Business Recommendations
# ============================================

st.markdown("---")
st.markdown('<h2 class="sub-header">💡 Business Recommendations</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **🎯 Immediate Actions**
    - Target month-to-month contract customers
    - Reach out to high-risk customers (top 20%)
    - Offer annual contract incentives
    """)

with col2:
    st.success("""
    **📊 Retention Strategies**
    - Loyalty program for 1+ year customers
    - Bundle discounts for multiple services
    - Auto-pay discount program
    """)

with col3:
    st.warning("""
    **⚠️ At-Risk Segments**
    - New customers (<6 months)
    - High monthly charges (>$80)
    - No online security/tech support
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>Customer Churn Analytics Dashboard | Built with Streamlit | Data-Driven Retention Strategy</p>",
    unsafe_allow_html=True
)