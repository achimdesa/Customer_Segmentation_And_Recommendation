import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page config must be first Streamlit command
st.set_page_config(layout="wide", page_title="Bank Customer Insights", page_icon="🏦")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('data/german_credit_data.csv')
    
    # Handle missing values
    df['Saving accounts'] = df['Saving accounts'].fillna('little')
    df['Checking account'] = df['Checking account'].fillna('little')

    # Encode categorical features
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

df, label_encoders = load_data()

# Feature selection and scaling
numeric_features = ['Age', 'Job', 'Credit amount', 'Duration']
X = df[numeric_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate cluster distribution
cluster_counts = df['Cluster'].value_counts()
cluster_percents = (cluster_counts / len(df)) * 100

# Enhanced recommendation engine
def generate_personalized_recommendations(cluster_data, label_encoders):
    recommendations = []
    
    # Decode all categorical features
    decoded = {
        'Sex': label_encoders['Sex'].inverse_transform([cluster_data['Sex'].mode()[0]])[0],
        'Housing': label_encoders['Housing'].inverse_transform([cluster_data['Housing'].mode()[0]])[0],
        'Saving': label_encoders['Saving accounts'].inverse_transform([cluster_data['Saving accounts'].mode()[0]])[0],
        'Checking': label_encoders['Checking account'].inverse_transform([cluster_data['Checking account'].mode()[0]])[0],
        'Purpose': label_encoders['Purpose'].inverse_transform([cluster_data['Purpose'].mode()[0]])[0]
    }
    
    # Calculate numerical metrics
    metrics = {
        'Avg Age': cluster_data['Age'].mean(),
        'Avg Job Level': cluster_data['Job'].mean(),
        'Avg Credit': cluster_data['Credit amount'].mean(),
        'Avg Duration': cluster_data['Duration'].mean()
    }
    
    # 1. Demographic-based recommendations
    if metrics['Avg Age'] < 30:
        recommendations.append(f"🎯 **Young Adults Program**: Digital-first banking solutions with mobile app incentives")
    elif metrics['Avg Age'] >= 35:
        recommendations.append(f"🎯 **Mature Clients**: Retirement planning and wealth preservation services")
    
    # 2. Financial capacity recommendations
    if decoded['Saving'] in ['rich', 'quite rich'] and decoded['Checking'] in ['rich', 'quite rich']:
        recommendations.append("💰 **High Net Worth**: Private banking with personalized investment advisory")
    elif decoded['Saving'] == 'little' and metrics['Avg Credit'] > 3000:
        recommendations.append("⚠️ **Risk Alert**: Consider secured loan options or guarantor requirements")
    
    # 3. Housing situation offers
    if decoded['Housing'] == 'rent':
        recommendations.append("🏠 **Renters Package**: Renter's insurance bundle with low-deposit accounts")
    elif decoded['Housing'] == 'own':
        recommendations.append("🏡 **Homeowners**: Home equity line of credit opportunities")
    elif decoded['Housing'] == 'free':
        recommendations.append("🏡 **No own home**: Offer mortgage loans to have their own house")
    


    # 4. Purpose-specific products
    purpose_offers = {
        'car': "🚗 **Auto Solutions**: Partner with local car dealerships for financing options, Low-interest car loans with free insurance for 1 year",
        'business': "📈 **Business Boost**: Startup financing with mentorship program",
        'education': "🎓 **Education Fund**: Student loans with grace period until graduation",
        'furniture/equipment': "🛋️ **Home Makeover**: 0% EMI for 12 months on furniture purchases",
        'radio/TV': "📺 **Electronics**: Instant point-of-sale financing at partner stores",
        'domestic appliances': "🧺 **Appliance Plan**: Extended warranty and service packages",
        'repairs': "🔧 **Home Maintenance**: Quick-approval repair loans"
    }
    recommendations.append(purpose_offers.get(decoded['Purpose'], "📌 General purpose loan products"))
    
    # 5. Job-level services
    if metrics['Avg Job Level'] >= 2.5:
        recommendations.append("👔 **Professional Banking**: Premium services with dedicated relationship manager")
    elif metrics['Avg Job Level'] <= 1:
        recommendations.append("🛠️ **Essential Banking**: No-frills accounts with financial literacy workshops")
    
    # 6. Credit behavior suggestions
    if metrics['Avg Duration'] > 24:
        recommendations.append("⏳ **Long-Term Planning**: Fixed-rate loan options for stability")
    elif metrics['Avg Duration'] < 12:
        recommendations.append("⚡ **Short-Term Needs**: Revolving credit lines for flexibility")
    
    

    return recommendations

# Streamlit UI
st.title("Bank Customer Segmentation Dashboard")

# Cluster distribution visualization
st.subheader("📊 Customer Segments Overview")
col1, col2, col3, col4 = st.columns(4)
for i, cluster in enumerate(sorted(df['Cluster'].unique())):
    with eval(f"col{i+1}"):
        st.metric(
            label=f"Cluster {cluster}",
            value=f"{cluster_percents[cluster]:.1f}%",
            help=f"{cluster_counts[cluster]} customers"
        )

# Cluster pie chart
fig_pie = px.pie(
    names=[f"Cluster {x}" for x in cluster_percents.index],
    values=cluster_percents.values,
    title="Customer Segment Distribution"
)
st.plotly_chart(fig_pie, use_container_width=True)

# Cluster selection
selected_cluster = st.sidebar.selectbox(
    "Select Customer Segment for Detailed Analysis", 
    sorted(df['Cluster'].unique()),
    format_func=lambda x: f"Cluster {x}"
)

cluster_data = df[df['Cluster'] == selected_cluster]



# Profile overview
st.header("📊 Segment Profile")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Average Age", f"{cluster_data['Age'].mean():.1f} years")
with col2:
    st.metric("Average Credit", f"€{cluster_data['Credit amount'].mean():,.0f}")
with col3:
    st.metric("Primary Purpose", 
              label_encoders['Purpose'].inverse_transform(
                  [cluster_data['Purpose'].mode()[0]])[0])
with col4:
    st.metric("Housing", 
              label_encoders['Housing'].inverse_transform(
                  [cluster_data['Housing'].mode()[0]])[0])

# Recommendations
st.header("💡 Personalized Recommendations")
recommendations = generate_personalized_recommendations(cluster_data, label_encoders)

for i, rec in enumerate(recommendations, 1):
    #st.success(f"**Recommendation {i}:** {rec}")
    st.success(f"{rec}")

# Detailed analysis
st.header("🔍 Feature Analysis")
tab1, tab2 = st.tabs(["📈 Numerical Features", "📊 Categorical Features"])

with tab1:
    num_feature = st.selectbox("Select numerical feature", numeric_features)
    
    # Check if the feature is discrete (integer values with limited unique values)
    is_discrete = cluster_data[num_feature].nunique() <= 10 and cluster_data[num_feature].dtype in ['int64', 'int32']
    
    if is_discrete:
        # Bar chart for discrete numerical features
        fig = px.bar(
            cluster_data[num_feature].value_counts().sort_index(),
            title=f"{num_feature} Distribution (Counts)",
            labels={'index': num_feature, 'value': 'Count'},
            text_auto=True
        )
        fig.update_layout(xaxis_type='category')  # Treat as categorical
    else:
        # Histogram for continuous numerical features
        fig = px.histogram(
            cluster_data,
            x=num_feature,
            title=f"{num_feature} Distribution",
            nbins=20,
            marginal="box"  # Adds box plot on top
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.subheader("📝 Summary Statistics")
    st.dataframe(cluster_data[num_feature].describe().to_frame().T.style.format("{:.2f}"))

with tab2:
    cat_feature = st.selectbox("Select category", 
                             ['Saving accounts', 'Checking account', 'Purpose'])
    decoded = label_encoders[cat_feature].inverse_transform(cluster_data[cat_feature])
    fig = px.pie(names=decoded, title=f"{cat_feature} Composition")
    st.plotly_chart(fig, use_container_width=True)
    
    # Add count table for categorical features
    st.subheader("🔢 Value Counts")
    value_counts = pd.Series(decoded).value_counts().reset_index()
    value_counts.columns = ['Value', 'Count']
    st.dataframe(value_counts)

# Raw data explorer
with st.expander("🔎 Explore Segment Data"):
    st.dataframe(cluster_data)


#new added
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
