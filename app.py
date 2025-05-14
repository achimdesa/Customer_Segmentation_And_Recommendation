
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# MUST be the first Streamlit command
st.set_page_config(
    layout="wide", 
    page_title="Bank Customer Insights",
    page_icon="🏦"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('german_credit_data.csv')
        
        # Add clustering if missing
        if 'Cluster' not in df.columns:
            #st.warning("Performing clustering now...")
            features = ['Age', 'Job', 'Credit amount', 'Duration']
            X = df[features]
            
            # Scale and cluster
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=4, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Only proceed if data loaded successfully
if not df.empty:
    # Rest of your dashboard code...
    TEAM_RECOMMENDATIONS = {
        "Marketing": {
        0: "📢 Digital ads on LinkedIn targeting young entrepreneurs",
        1: "🚗 YouTube ads for auto loans with local dealerships",
        2: "🛋️ Instagram stories showing home furnishing loans",
        3: "📧 Personalized email campaigns for high-net-worth individuals"
    },
    "Risk Management": {
        0: "📝 Require business plan review for loans >€5,000",
        1: "🔐 Standard secured auto loan package (LTV 80%)",
        2: "💳 Salary deduction option for repayment",
        3: "🏠 Property appraisal required for all loans"
    },
    "Product Development": {
        0: "🚀 Startup loan package with mentorship program",
        1: "🛡️ Auto loan with insurance bundle",
        2: "🛒 Retail partner financing programs",
        3: "💎 Wealth management advisory services"
    },
    "Customer Service": {
        0: "👔 Assign dedicated business banking reps",
        1: "📞 24/7 auto loan application hotline",
        2: "🛠️ In-store financing support",
        3: "🎩 Concierge service for premium clients"
    }
        # ... (keep your existing recommendations dictionary)
    }
    
    # Display the dashboard
    st.title("Bank Customer Segmentation Dashboard")
    st.write(f"Loaded {len(df)} records with {len(df['Cluster'].unique())} clusters")
    
    # ... rest of your dashboard implementation
    # Dashboard layout
    ## st.title("Bank Customer Segmentation Dashboard")

    # Sidebar - Team selection
    team = st.sidebar.selectbox(
        "SELECT BUSINESS TEAM",
        list(TEAM_RECOMMENDATIONS.keys()),
        index=0
    )

    # Main header
    st.header(f"{team} Team Recommendations")

    # Cluster performance summary
    st.subheader("Cluster Performance Overview")
    cluster_summary = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Credit amount': 'mean',
        'Duration': 'mean',
        'Purpose': lambda x: x.mode()[0]
    }).reset_index()

    # Format summary
    cluster_summary['Age'] = cluster_summary['Age'].round(1)
    cluster_summary['Credit amount'] = "€" + cluster_summary['Credit amount'].round(0).astype(int).astype(str)
    cluster_summary['Duration'] = cluster_summary['Duration'].round(1).astype(str) + " months"

    # Display summary table with team recommendations
    for cluster in sorted(df['Cluster'].unique()):
        with st.expander(f"CLUSTER {cluster} DETAILS", expanded=True):
            cols = st.columns([1, 3])
            
            with cols[0]:
                st.markdown("**Key Metrics**")
                st.dataframe(
                    cluster_summary[cluster_summary['Cluster'] == cluster].drop(columns=['Cluster']),
                    hide_index=True,
                    use_container_width=True
                )
            
            with cols[1]:
                st.markdown("**Team Recommendation**")
                st.success(TEAM_RECOMMENDATIONS[team][cluster])
                
                # Feature distribution
                feature = st.selectbox(
                    "Select feature to analyze:",
                    ['Age', 'Credit amount', 'Duration', 'Purpose'],
                    key=f"feature_{cluster}"
                )
                
                if feature in ['Age', 'Credit amount', 'Duration']:
                    fig = px.histogram(
                        df[df['Cluster'] == cluster],
                        x=feature,
                        title=f"Cluster {cluster} {feature} Distribution"
                    )
                else:
                    fig = px.pie(
                        df[df['Cluster'] == cluster],
                        names=feature,
                        title=f"Cluster {cluster} Loan Purposes"
                    )
                st.plotly_chart(fig, use_container_width=True)

    # Comparative analysis
    st.header("Cross-Cluster Comparison")
    compare_features = st.multiselect(
        "Select metrics to compare:",
        ['Age', 'Credit amount', 'Duration'],
        default=['Credit amount']
    )

    if compare_features:
        fig = px.bar(
            df.groupby('Cluster')[compare_features].mean().reset_index(),
            x='Cluster',
            y=compare_features,
            barmode='group',
            title="Cluster Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)


else:
    st.error("Failed to load data. Please check your CSV file.")




# Data export
st.sidebar.download_button(
    "📥 Export Cluster Data",
    df.to_csv(index=False),
    "customer_segments.csv",
    "text/csv"
)