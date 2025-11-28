import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import shapiro, jarque_bera
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import io
import base64
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Macroeconomics Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F4FD 0%, #B8D4E8 100%);
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #1F2937 !important;
    }
    .insight-box * {
        color: #1F2937 !important;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #1F2937 !important;
    }
    .warning-box * {
        color: #1F2937 !important;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #1F2937 !important;
    }
    .success-box * {
        color: #1F2937 !important;
    }
    .equation-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #00d4ff;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("finance_economics_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data
def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def calculate_vif(X, feature_names):
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    vif_data = []
    X_df = pd.DataFrame(X, columns=feature_names)
    for i, col in enumerate(feature_names):
        if X_df[col].std() == 0:
            vif_data.append({'Variable': col, 'VIF': np.inf})
        else:
            y_temp = X_df[col]
            X_temp = X_df.drop(columns=[col])
            if X_temp.shape[1] > 0:
                model = LinearRegression()
                model.fit(X_temp, y_temp)
                r2 = model.score(X_temp, y_temp)
                vif = 1 / (1 - r2) if r2 < 1 else np.inf
                vif_data.append({'Variable': col, 'VIF': vif})
            else:
                vif_data.append({'Variable': col, 'VIF': 1.0})
    return pd.DataFrame(vif_data)

def calculate_regression_stats(X, y, feature_names):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    y_pred = sklearn_model.predict(X)
    
    sklearn_model_scaled = LinearRegression()
    sklearn_model_scaled.fit(X_scaled, y)
    standardized_coefs = sklearn_model_scaled.coef_
    
    vif_df = calculate_vif(X, feature_names)
    
    stats_dict = {
        'coefficients': sklearn_model.coef_,
        'standardized_coefs': standardized_coefs,
        'intercept': sklearn_model.intercept_,
        'r2': r2_score(y, y_pred),
        'adj_r2': model.rsquared_adj,
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'f_stat': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'p_values': model.pvalues[1:],
        't_values': model.tvalues[1:],
        'std_errors': model.bse[1:],
        'conf_int': model.conf_int()[1:],
        'feature_names': feature_names,
        'y_pred': y_pred,
        'residuals': y - y_pred,
        'model': sklearn_model,
        'sm_model': model,
        'vif': vif_df
    }
    return stats_dict

def format_equation(intercept, coefficients, feature_names, dependent_var):
    equation = f"{dependent_var} = {intercept:.4f}"
    for coef, name in zip(coefficients, feature_names):
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.4f}×({name})"
    equation += " + ε"
    return equation

def interpret_correlation(corr_value):
    abs_corr = abs(corr_value)
    direction = "positive" if corr_value > 0 else "negative"
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    elif abs_corr >= 0.2:
        strength = "weak"
    else:
        strength = "very weak/negligible"
    return f"{strength} {direction}"

def calculate_advanced_diagnostics(X, y, residuals, y_pred):
    """Calculate advanced diagnostic tests for regression models"""
    diagnostics = {}
    
    try:
        shapiro_stat, shapiro_p = shapiro(residuals[:min(5000, len(residuals))])
        diagnostics['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
    except:
        diagnostics['shapiro'] = {'statistic': np.nan, 'p_value': np.nan}
    
    try:
        jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
        diagnostics['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_p, 'skew': skew, 'kurtosis': kurtosis}
    except:
        diagnostics['jarque_bera'] = {'statistic': np.nan, 'p_value': np.nan, 'skew': np.nan, 'kurtosis': np.nan}
    
    try:
        X_const = sm.add_constant(X)
        bp_stat, bp_p, f_stat, f_p = het_breuschpagan(residuals, X_const)
        diagnostics['breusch_pagan'] = {'statistic': bp_stat, 'p_value': bp_p, 'f_stat': f_stat, 'f_p': f_p}
    except:
        diagnostics['breusch_pagan'] = {'statistic': np.nan, 'p_value': np.nan, 'f_stat': np.nan, 'f_p': np.nan}
    
    try:
        dw_stat = durbin_watson(residuals)
        diagnostics['durbin_watson'] = {'statistic': dw_stat}
    except:
        diagnostics['durbin_watson'] = {'statistic': np.nan}
    
    return diagnostics

def create_qq_plot(residuals, title="Q-Q Plot of Residuals"):
    """Create Q-Q plot for normality assessment"""
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sorted_residuals,
        mode='markers',
        marker=dict(color='#1E88E5', size=6, opacity=0.6),
        name='Residuals'
    ))
    
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Normal Line'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=400,
        showlegend=True
    )
    return fig

def generate_report_data(stats, model_name, diagnostics=None):
    """Generate report data for download"""
    report = f"""
# {model_name} Regression Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance Metrics
- R² Score: {stats['r2']:.4f}
- Adjusted R²: {stats['adj_r2']:.4f}
- RMSE: {stats['rmse']:.4f}
- MAE: {stats['mae']:.4f}
- F-Statistic: {stats['f_stat']:.4f}
- F-Statistic p-value: {stats['f_pvalue']:.6f}

## Coefficient Analysis
| Variable | Coefficient | Std Error | t-value | p-value | Significant |
|----------|-------------|-----------|---------|---------|-------------|
"""
    for i, name in enumerate(stats['feature_names']):
        sig = "Yes" if stats['p_values'][i] < 0.05 else "No"
        report += f"| {name} | {stats['coefficients'][i]:.4f} | {stats['std_errors'][i]:.4f} | {stats['t_values'][i]:.4f} | {stats['p_values'][i]:.4f} | {sig} |\n"
    
    if diagnostics:
        report += f"""
## Diagnostic Tests

### Normality Tests
- Shapiro-Wilk: Statistic = {diagnostics['shapiro']['statistic']:.4f}, p-value = {diagnostics['shapiro']['p_value']:.4f}
- Jarque-Bera: Statistic = {diagnostics['jarque_bera']['statistic']:.4f}, p-value = {diagnostics['jarque_bera']['p_value']:.4f}

### Heteroscedasticity Test
- Breusch-Pagan: Statistic = {diagnostics['breusch_pagan']['statistic']:.4f}, p-value = {diagnostics['breusch_pagan']['p_value']:.4f}

### Autocorrelation Test
- Durbin-Watson: Statistic = {diagnostics['durbin_watson']['statistic']:.4f}

## Interpretation
- Normality: {'Residuals appear normally distributed (p > 0.05)' if diagnostics['shapiro']['p_value'] > 0.05 else 'Residuals may not be normally distributed (p < 0.05)'}
- Heteroscedasticity: {'No significant heteroscedasticity detected (p > 0.05)' if diagnostics['breusch_pagan']['p_value'] > 0.05 else 'Potential heteroscedasticity present (p < 0.05)'}
- Autocorrelation: {'No significant autocorrelation (DW ≈ 2)' if 1.5 < diagnostics['durbin_watson']['statistic'] < 2.5 else 'Potential autocorrelation detected'}
"""
    
    return report

def main():
    st.markdown('<div class="main-header">📊 Macroeconomics Analytics Dashboard</div>', unsafe_allow_html=True)
    
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    numeric_cols = get_numeric_columns(df)
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/economics.png", width=80)
        st.title("Navigation & Filters")
        
        page = st.radio(
            "Select Analysis Section",
            ["📈 Overview", "🔗 Correlation Analysis", "💹 Stock Index Regression", 
             "📊 GDP Growth Analysis", "🔮 Prediction Tool", "📉 Advanced Visualizations"],
            index=0
        )
        
        st.divider()
        st.subheader("🔍 Data Filters")
        
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
        
        stock_indices = df['Stock Index'].unique().tolist()
        selected_indices = st.multiselect(
            "Select Stock Indices",
            options=stock_indices,
            default=stock_indices
        )
        
        st.divider()
        st.markdown("### 📋 Dataset Info")
        st.info(f"📅 Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        st.info(f"📊 Total Records: {len(df):,}")
        st.info(f"📈 Variables: {len(df.columns)}")
    
    if len(date_range) == 2:
        mask = (df['Date'] >= pd.Timestamp(date_range[0])) & (df['Date'] <= pd.Timestamp(date_range[1]))
        df_filtered = df[mask]
    else:
        df_filtered = df.copy()
    
    if selected_indices:
        df_filtered = df_filtered[df_filtered['Stock Index'].isin(selected_indices)]
    
    if len(df_filtered) == 0:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return
    
    if page == "📈 Overview":
        show_overview(df_filtered, numeric_cols)
    elif page == "🔗 Correlation Analysis":
        show_correlation_analysis(df_filtered, numeric_cols)
    elif page == "💹 Stock Index Regression":
        show_stock_regression(df_filtered)
    elif page == "📊 GDP Growth Analysis":
        show_gdp_analysis(df_filtered)
    elif page == "🔮 Prediction Tool":
        show_prediction_tool(df_filtered)
    elif page == "📉 Advanced Visualizations":
        show_advanced_visualizations(df_filtered, numeric_cols)

def show_overview(df, numeric_cols):
    st.header("📈 Data Overview & Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_close = df['Close Price'].mean()
        st.metric("Avg Close Price", f"${avg_close:,.2f}", 
                  delta=f"{df['Close Price'].std():.2f} std")
    with col2:
        avg_gdp = df['GDP Growth (%)'].mean()
        st.metric("Avg GDP Growth", f"{avg_gdp:.2f}%",
                  delta=f"{df['GDP Growth (%)'].std():.2f} std")
    with col3:
        avg_inflation = df['Inflation Rate (%)'].mean()
        st.metric("Avg Inflation Rate", f"{avg_inflation:.2f}%",
                  delta=f"{df['Inflation Rate (%)'].std():.2f} std")
    with col4:
        avg_unemployment = df['Unemployment Rate (%)'].mean()
        st.metric("Avg Unemployment", f"{avg_unemployment:.2f}%",
                  delta=f"{df['Unemployment Rate (%)'].std():.2f} std")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Key Economic Indicators Over Time")
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Close Price Trend', 'GDP Growth Trend', 
            'Inflation Rate Trend', 'Interest Rate Trend'
        ))
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close Price'], 
                                  mode='lines', name='Close Price',
                                  line=dict(color='#1E88E5')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['GDP Growth (%)'], 
                                  mode='lines', name='GDP Growth',
                                  line=dict(color='#43A047')), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Inflation Rate (%)'], 
                                  mode='lines', name='Inflation',
                                  line=dict(color='#FB8C00')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Interest Rate (%)'], 
                                  mode='lines', name='Interest Rate',
                                  line=dict(color='#E53935')), row=2, col=2)
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>📌 Key Insights:</strong>
        <ul>
            <li><strong>Close Price:</strong> Shows market valuation trends over time</li>
            <li><strong>GDP Growth:</strong> Indicates economic expansion or contraction periods</li>
            <li><strong>Inflation Rate:</strong> Reflects purchasing power changes</li>
            <li><strong>Interest Rate:</strong> Central bank monetary policy indicator</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📈 Stock Index Distribution")
        index_counts = df['Stock Index'].value_counts()
        fig = px.pie(values=index_counts.values, names=index_counts.index,
                     title='Distribution of Stock Indices',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Summary Statistics")
        summary_cols = ['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)', 
                        'Interest Rate (%)', 'Crude Oil Price (USD per Barrel)', 
                        'Gold Price (USD per Ounce)']
        st.dataframe(df[summary_cols].describe().round(2), use_container_width=True)
    
    st.divider()
    st.subheader("📉 Commodity Prices Comparison")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Crude Oil Price (USD per Barrel)'],
                   name="Crude Oil Price", line=dict(color='#795548')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Gold Price (USD per Ounce)'],
                   name="Gold Price", line=dict(color='#FFD700')),
        secondary_y=True
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Oil Price (USD/Barrel)", secondary_y=False)
    fig.update_yaxes(title_text="Gold Price (USD/Ounce)", secondary_y=True)
    fig.update_layout(height=400, title="Oil vs Gold Price Trends")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>📌 Commodity Price Insights:</strong>
    <ul>
        <li><strong>Oil prices</strong> often move inversely to economic growth during supply shocks</li>
        <li><strong>Gold</strong> typically serves as a safe-haven asset during economic uncertainty</li>
        <li>Both commodities are key inflation drivers in macroeconomic analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_correlation_analysis(df, numeric_cols):
    st.header("🔗 Correlation Analysis")
    
    st.subheader("Select Variables for Correlation Analysis")
    key_variables = ['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)', 
                     'Unemployment Rate (%)', 'Interest Rate (%)', 
                     'Consumer Confidence Index', 'Crude Oil Price (USD per Barrel)',
                     'Gold Price (USD per Ounce)', 'Forex USD/EUR', 'Forex USD/JPY',
                     'Trading Volume', 'Government Debt (Billion USD)']
    
    available_vars = [v for v in key_variables if v in df.columns]
    selected_vars = st.multiselect(
        "Choose variables for correlation matrix",
        options=available_vars,
        default=available_vars[:8]
    )
    
    if len(selected_vars) < 2:
        st.warning("Please select at least 2 variables for correlation analysis.")
        return
    
    corr_matrix = df[selected_vars].corr()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Correlation Heatmap")
        # Create correlation matrix with annotations
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=selected_vars,
            y=selected_vars,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 11, "color": "black"},
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(
            height=600,
            title="Correlation Heatmap",
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("🔝 Top Correlations")
        corr_pairs = []
        for i in range(len(selected_vars)):
            for j in range(i+1, len(selected_vars)):
                corr_pairs.append({
                    'Variable 1': selected_vars[i],
                    'Variable 2': selected_vars[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
        
        st.dataframe(
            corr_df[['Variable 1', 'Variable 2', 'Correlation']].head(10).style.background_gradient(
                subset=['Correlation'], cmap='RdBu_r', vmin=-1, vmax=1
            ).format({'Correlation': '{:.3f}'}),
            use_container_width=True
        )
    
    st.divider()
    st.subheader("📈 Significant Relationships Interpretation")
    
    strong_correlations = corr_df[corr_df['Abs Correlation'] >= 0.3].head(6)
    
    cols = st.columns(2)
    for idx, row in strong_correlations.iterrows():
        col_idx = idx % 2
        with cols[col_idx]:
            interpretation = interpret_correlation(row['Correlation'])
            sign = "🟢" if row['Correlation'] > 0 else "🔴"
            st.markdown(f"""
            <div class="insight-box">
            <strong>{sign} {row['Variable 1']} ↔ {row['Variable 2']}</strong><br>
            Correlation: <strong>{row['Correlation']:.3f}</strong> ({interpretation})<br>
            <em>As {row['Variable 1'].lower()} changes, {row['Variable 2'].lower()} tends to move in {'the same' if row['Correlation'] > 0 else 'the opposite'} direction.</em>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📊 Scatter Plot Matrix")
    
    scatter_vars = st.multiselect(
        "Select variables for scatter plot (max 5 for clarity)",
        options=selected_vars,
        default=selected_vars[:4],
        max_selections=5
    )
    
    if len(scatter_vars) >= 2:
        fig = px.scatter_matrix(
            df[scatter_vars],
            dimensions=scatter_vars,
            color_discrete_sequence=['#1E88E5'],
            opacity=0.6
        )
        fig.update_layout(height=700)
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>📌 How to Read Scatter Matrix:</strong>
        <ul>
            <li>Each cell shows the relationship between two variables</li>
            <li>Linear patterns indicate strong correlations</li>
            <li>Clusters may indicate different market regimes or conditions</li>
            <li>Outliers represent unusual economic events</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_stock_regression(df):
    st.header("💹 Stock Index Regression Analysis")
    
    st.markdown("""
    <div class="equation-box">
    <strong>Regression Model:</strong><br>
    Y (Close Price) = β₀ + β₁(Inflation) + β₂(Oil Price) + β₃(Interest Rate) + β₄(Gold Price) + β₅(FX Rate) + ε
    </div>
    """, unsafe_allow_html=True)
    
    fx_col = None
    if 'Forex USD/EUR' in df.columns:
        fx_col = 'Forex USD/EUR'
        fx_display = 'FX Rate (USD/EUR)'
    elif 'Forex USD/JPY' in df.columns:
        fx_col = 'Forex USD/JPY'
        fx_display = 'FX Rate (USD/JPY)'
    else:
        st.error("No FX Rate column (Forex USD/EUR or Forex USD/JPY) found in dataset.")
        return
    
    base_required = ['Close Price', 'Inflation Rate (%)', 'Crude Oil Price (USD per Barrel)',
                     'Interest Rate (%)', 'Gold Price (USD per Ounce)']
    
    missing_cols = [col for col in base_required if col not in df.columns]
    if missing_cols:
        st.error(f"Required columns not found: {', '.join(missing_cols)}")
        return
    
    feature_names = ['Inflation Rate (%)', 'Crude Oil Price (USD per Barrel)',
                     'Interest Rate (%)', 'Gold Price (USD per Ounce)', fx_col]
    display_names = ['Inflation', 'Oil Price', 'Interest Rate', 'Gold Price', fx_display]
    
    all_cols = ['Close Price'] + feature_names
    df_clean = df[all_cols].dropna()
    
    if len(df_clean) < 10:
        st.warning(f"Insufficient data for regression analysis. Only {len(df_clean)} valid observations found.")
        return
    
    st.info(f"📊 Analysis based on {len(df_clean):,} observations after removing missing values.")
    
    X = df_clean[feature_names].values
    y = df_clean['Close Price'].values
    
    stats = calculate_regression_stats(X, y, display_names)
    
    st.subheader("📐 Derived Regression Equation")
    equation = format_equation(stats['intercept'], stats['coefficients'], display_names, 'Close Price')
    st.markdown(f'<div class="equation-box">{equation}</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{stats['r2']:.4f}", 
                  help="Proportion of variance explained by the model")
    with col2:
        st.metric("Adjusted R²", f"{stats['adj_r2']:.4f}",
                  help="R² adjusted for number of predictors")
    with col3:
        st.metric("RMSE", f"{stats['rmse']:.2f}",
                  help="Root Mean Square Error")
    with col4:
        f_sig = "✅ Significant" if stats['f_pvalue'] < 0.05 else "❌ Not Significant"
        st.metric("F-Statistic", f"{stats['f_stat']:.2f}", delta=f_sig)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Coefficient Analysis")
        coef_df = pd.DataFrame({
            'Variable': display_names,
            'Coefficient': stats['coefficients'],
            'Std Error': stats['std_errors'],
            't-value': stats['t_values'],
            'p-value': stats['p_values'],
            'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in stats['p_values']]
        })
        
        st.dataframe(
            coef_df.style.background_gradient(subset=['Coefficient'], cmap='RdYlGn', vmin=-max(abs(coef_df['Coefficient'])), vmax=max(abs(coef_df['Coefficient']))),
            use_container_width=True
        )
        
        fig = go.Figure()
        colors = ['#4CAF50' if p < 0.05 else '#FF5722' for p in stats['p_values']]
        fig.add_trace(go.Bar(
            x=display_names,
            y=stats['coefficients'],
            marker_color=colors,
            text=[f"{c:.3f}" for c in stats['coefficients']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Coefficient Values (Green = Significant, Red = Not Significant)",
            xaxis_title="Predictor Variable",
            yaxis_title="Coefficient Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Actual vs Predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, y=stats['y_pred'],
            mode='markers',
            marker=dict(color='#1E88E5', opacity=0.6),
            name='Predictions'
        ))
        min_val, max_val = min(y.min(), stats['y_pred'].min()), max(y.max(), stats['y_pred'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="Actual vs Predicted Close Price",
            xaxis_title="Actual Close Price",
            yaxis_title="Predicted Close Price",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📉 Residual Distribution")
        fig = px.histogram(stats['residuals'], nbins=30,
                           labels={'value': 'Residual', 'count': 'Frequency'},
                           color_discrete_sequence=['#673AB7'])
        fig.update_layout(title="Distribution of Residuals", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("📊 Individual Variable Relationships")
    
    tabs = st.tabs(display_names)
    for idx, (tab, feat_name, disp_name) in enumerate(zip(tabs, feature_names, display_names)):
        with tab:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.scatter(df_clean, x=feat_name, y='Close Price',
                                 trendline='ols',
                                 labels={feat_name: disp_name, 'Close Price': 'Close Price'},
                                 color_discrete_sequence=['#1E88E5'])
                fig.update_layout(title=f"Close Price vs {disp_name}", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                corr = df_clean[feat_name].corr(df_clean['Close Price'])
                interpretation = interpret_correlation(corr)
                
                coef = stats['coefficients'][idx]
                p_val = stats['p_values'][idx]
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>📌 Interpretation: {disp_name}</strong><br><br>
                <strong>Correlation:</strong> {corr:.3f} ({interpretation})<br><br>
                <strong>Coefficient:</strong> {coef:.4f}<br>
                <em>→ A 1-unit increase in {disp_name.lower()} is associated with a {abs(coef):.2f} {'increase' if coef > 0 else 'decrease'} in Close Price</em><br><br>
                <strong>P-value:</strong> {p_val:.4f}<br>
                <em>→ {'Statistically significant at 5% level ✅' if p_val < 0.05 else 'Not statistically significant ❌'}</em>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📝 Model Summary & Interpretation")
    
    st.markdown(f"""
    <div class="success-box">
    <strong>🎯 Overall Model Performance:</strong>
    <ul>
        <li><strong>R² = {stats['r2']:.4f}:</strong> The model explains {stats['r2']*100:.1f}% of the variance in Close Price</li>
        <li><strong>F-statistic = {stats['f_stat']:.2f} (p = {stats['f_pvalue']:.4e}):</strong> {'The overall model is statistically significant' if stats['f_pvalue'] < 0.05 else 'The overall model is not statistically significant'}</li>
        <li><strong>RMSE = {stats['rmse']:.2f}:</strong> Average prediction error is ${stats['rmse']:.2f}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    sig_vars = [name for name, p in zip(display_names, stats['p_values']) if p < 0.05]
    nonsig_vars = [name for name, p in zip(display_names, stats['p_values']) if p >= 0.05]
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>📊 Variable Significance:</strong>
    <ul>
        <li><strong>Significant predictors (p < 0.05):</strong> {', '.join(sig_vars) if sig_vars else 'None'}</li>
        <li><strong>Non-significant predictors (p ≥ 0.05):</strong> {', '.join(nonsig_vars) if nonsig_vars else 'None'}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("🔍 Multicollinearity Analysis (VIF)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        vif_df = stats['vif'].copy()
        vif_df['Status'] = vif_df['VIF'].apply(lambda x: '✅ OK' if x < 5 else ('⚠️ Moderate' if x < 10 else '🔴 High'))
        st.dataframe(vif_df.style.format({'VIF': '{:.2f}'}).background_gradient(
            subset=['VIF'], cmap='YlOrRd', vmin=1, vmax=10
        ), use_container_width=True)
    
    with col2:
        high_vif = vif_df[vif_df['VIF'] >= 5]
        if len(high_vif) > 0:
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ Multicollinearity Warning:</strong><br>
            Variables with VIF ≥ 5 may have multicollinearity issues: <strong>{', '.join(high_vif['Variable'].tolist())}</strong><br>
            <em>High VIF values indicate that these predictors are highly correlated with other predictors, 
            which can inflate standard errors and make coefficient estimates unstable.</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            <strong>✅ No Multicollinearity Issues:</strong><br>
            All VIF values are below 5, indicating low multicollinearity among predictors.
            Coefficient estimates are reliable.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>📌 VIF Interpretation Guide:</strong>
        <ul>
            <li><strong>VIF = 1:</strong> No correlation with other variables</li>
            <li><strong>VIF 1-5:</strong> Moderate correlation (acceptable)</li>
            <li><strong>VIF 5-10:</strong> High correlation (caution advised)</li>
            <li><strong>VIF > 10:</strong> Severe multicollinearity (problematic)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("🧪 Advanced Diagnostics")
    
    diagnostics = calculate_advanced_diagnostics(X, y, stats['residuals'], stats['y_pred'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Q-Q Plot (Normality Check)")
        qq_fig = create_qq_plot(stats['residuals'], "Q-Q Plot of Stock Regression Residuals")
        st.plotly_chart(qq_fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Residuals vs Fitted Values")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats['y_pred'], y=stats['residuals'],
            mode='markers',
            marker=dict(color='#9C27B0', opacity=0.6, size=6),
            name='Residuals'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residuals vs Fitted (Heteroscedasticity Check)",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Diagnostic Test Results")
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    
    with diag_col1:
        shapiro_pass = diagnostics['shapiro']['p_value'] > 0.05 if not np.isnan(diagnostics['shapiro']['p_value']) else False
        st.metric(
            "Shapiro-Wilk Test",
            f"p = {diagnostics['shapiro']['p_value']:.4f}" if not np.isnan(diagnostics['shapiro']['p_value']) else "N/A",
            delta="Normal ✅" if shapiro_pass else "Non-normal ⚠️"
        )
    
    with diag_col2:
        bp_pass = diagnostics['breusch_pagan']['p_value'] > 0.05 if not np.isnan(diagnostics['breusch_pagan']['p_value']) else False
        st.metric(
            "Breusch-Pagan Test",
            f"p = {diagnostics['breusch_pagan']['p_value']:.4f}" if not np.isnan(diagnostics['breusch_pagan']['p_value']) else "N/A",
            delta="Homoscedastic ✅" if bp_pass else "Heteroscedastic ⚠️"
        )
    
    with diag_col3:
        dw_stat = diagnostics['durbin_watson']['statistic']
        dw_pass = 1.5 < dw_stat < 2.5 if not np.isnan(dw_stat) else False
        st.metric(
            "Durbin-Watson",
            f"{dw_stat:.4f}" if not np.isnan(dw_stat) else "N/A",
            delta="No autocorrelation ✅" if dw_pass else "Autocorrelation ⚠️"
        )
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>📌 Diagnostic Interpretation:</strong>
    <ul>
        <li><strong>Normality (Shapiro-Wilk):</strong> {'Residuals appear normally distributed - regression assumptions satisfied' if shapiro_pass else 'Residuals may not be normally distributed - consider robust standard errors'}</li>
        <li><strong>Heteroscedasticity (Breusch-Pagan):</strong> {'Constant variance assumption holds' if bp_pass else 'Variance of residuals may not be constant - consider weighted least squares'}</li>
        <li><strong>Autocorrelation (Durbin-Watson):</strong> {'No significant autocorrelation detected (DW ≈ 2)' if dw_pass else 'Potential autocorrelation - consider time-series models'}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📥 Download Report")
    report = generate_report_data(stats, "Stock Index", diagnostics)
    st.download_button(
        label="📄 Download Full Regression Report",
        data=report,
        file_name="stock_index_regression_report.md",
        mime="text/markdown"
    )

def show_gdp_analysis(df):
    st.header("📊 GDP Growth Rate Analysis")
    
    st.markdown("""
    <div class="equation-box">
    <strong>GDP Growth Regression Model:</strong><br>
    GDP Growth = β₀ + β₁(Inflation) + β₂(Unemployment) + β₃(Interest Rate) + β₄(Consumer Confidence) + β₅(Oil Price) + β₆(FX Rate) + β₇(Gov Debt) + ε
    </div>
    """, unsafe_allow_html=True)
    
    fx_col = None
    fx_display = None
    if 'Forex USD/EUR' in df.columns:
        fx_col = 'Forex USD/EUR'
        fx_display = 'FX Rate (USD/EUR)'
    elif 'Forex USD/JPY' in df.columns:
        fx_col = 'Forex USD/JPY'
        fx_display = 'FX Rate (USD/JPY)'
    
    base_required = ['GDP Growth (%)', 'Inflation Rate (%)', 'Unemployment Rate (%)',
                     'Interest Rate (%)', 'Consumer Confidence Index', 
                     'Crude Oil Price (USD per Barrel)']
    
    optional_cols = []
    optional_display = []
    if fx_col and fx_display and fx_col in df.columns:
        optional_cols.append(fx_col)
        optional_display.append(fx_display)
    if 'Government Debt (Billion USD)' in df.columns:
        optional_cols.append('Government Debt (Billion USD)')
        optional_display.append('Gov Debt')
    
    missing_cols = [col for col in base_required if col not in df.columns]
    if missing_cols:
        st.error(f"Required columns not found: {', '.join(missing_cols)}")
        return
    
    all_cols = base_required + optional_cols
    df_clean = df[all_cols].dropna()
    
    if len(df_clean) < 10:
        st.warning(f"Insufficient data for regression analysis. Only {len(df_clean)} valid observations found.")
        return
    
    st.info(f"📊 Analysis based on {len(df_clean):,} observations after removing missing values.")
    
    feature_names = ['Inflation Rate (%)', 'Unemployment Rate (%)', 'Interest Rate (%)',
                     'Consumer Confidence Index', 'Crude Oil Price (USD per Barrel)'] + optional_cols
    display_names = ['Inflation', 'Unemployment', 'Interest Rate', 'Consumer Confidence', 'Oil Price'] + optional_display
    
    X = df_clean[feature_names].values
    y = df_clean['GDP Growth (%)'].values
    
    stats = calculate_regression_stats(X, y, display_names)
    
    st.subheader("📐 Derived GDP Growth Equation")
    equation = format_equation(stats['intercept'], stats['coefficients'], display_names, 'GDP Growth')
    st.markdown(f'<div class="equation-box">{equation}</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{stats['r2']:.4f}")
    with col2:
        st.metric("Adjusted R²", f"{stats['adj_r2']:.4f}")
    with col3:
        st.metric("RMSE", f"{stats['rmse']:.2f}%")
    with col4:
        f_sig = "✅ Significant" if stats['f_pvalue'] < 0.05 else "❌ Not Significant"
        st.metric("F-Statistic", f"{stats['f_stat']:.2f}", delta=f_sig)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 GDP Coefficient Analysis")
        coef_df = pd.DataFrame({
            'Variable': display_names,
            'Coefficient': stats['coefficients'],
            'Std Error': stats['std_errors'],
            't-value': stats['t_values'],
            'p-value': stats['p_values'],
            'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in stats['p_values']]
        })
        
        st.dataframe(coef_df, use_container_width=True)
        
        fig = go.Figure()
        colors = ['#4CAF50' if p < 0.05 else '#FF5722' for p in stats['p_values']]
        fig.add_trace(go.Bar(
            x=display_names,
            y=stats['coefficients'],
            marker_color=colors,
            error_y=dict(type='data', array=stats['std_errors']),
            text=[f"{c:.4f}" for c in stats['coefficients']],
            textposition='outside'
        ))
        fig.update_layout(
            title="GDP Growth Coefficients with Standard Errors",
            xaxis_title="Predictor Variable",
            yaxis_title="Coefficient Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 GDP: Actual vs Predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, y=stats['y_pred'],
            mode='markers',
            marker=dict(color='#43A047', opacity=0.6, size=8),
            name='Predictions'
        ))
        min_val, max_val = min(y.min(), stats['y_pred'].min()), max(y.max(), stats['y_pred'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="GDP Growth: Actual vs Predicted",
            xaxis_title="Actual GDP Growth (%)",
            yaxis_title="Predicted GDP Growth (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📉 GDP Residual Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats['y_pred'],
            y=stats['residuals'],
            mode='markers',
            marker=dict(color='#9C27B0', opacity=0.6),
            name='Residuals'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residuals vs Fitted Values",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("📊 GDP vs Individual Predictors")
    
    tabs = st.tabs(display_names)
    for idx, (tab, feat_name, disp_name) in enumerate(zip(tabs, feature_names, display_names)):
        with tab:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.scatter(df_clean, x=feat_name, y='GDP Growth (%)',
                                 trendline='ols',
                                 color_discrete_sequence=['#43A047'])
                fig.update_layout(title=f"GDP Growth vs {disp_name}", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                corr = df_clean[feat_name].corr(df_clean['GDP Growth (%)'])
                interpretation = interpret_correlation(corr)
                coef = stats['coefficients'][idx]
                p_val = stats['p_values'][idx]
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>📌 {disp_name} → GDP Growth</strong><br><br>
                <strong>Correlation:</strong> {corr:.3f} ({interpretation})<br><br>
                <strong>Coefficient:</strong> {coef:.4f}<br>
                <em>→ A 1-unit increase in {disp_name.lower()} is associated with a {abs(coef):.2f}% {'increase' if coef > 0 else 'decrease'} in GDP growth</em><br><br>
                <strong>P-value:</strong> {p_val:.4f}<br>
                <em>→ {'Statistically significant ✅' if p_val < 0.05 else 'Not statistically significant ❌'}</em>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 GDP Growth Distribution")
        fig = px.histogram(df, x='GDP Growth (%)', nbins=30, 
                           color_discrete_sequence=['#43A047'],
                           marginal='box')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 GDP Growth by Stock Index")
        fig = px.box(df, x='Stock Index', y='GDP Growth (%)',
                     color='Stock Index',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="success-box">
    <strong>🎯 GDP Growth Model Summary:</strong>
    <ul>
        <li><strong>Model Fit:</strong> R² = {stats['r2']:.4f} - The model explains {stats['r2']*100:.1f}% of GDP growth variance</li>
        <li><strong>Statistical Significance:</strong> F-stat = {stats['f_stat']:.2f}, p = {stats['f_pvalue']:.4e}</li>
        <li><strong>Prediction Accuracy:</strong> RMSE = {stats['rmse']:.2f}% - average prediction error</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("🔍 Multicollinearity Analysis (VIF)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        vif_df = stats['vif'].copy()
        vif_df['Status'] = vif_df['VIF'].apply(lambda x: '✅ OK' if x < 5 else ('⚠️ Moderate' if x < 10 else '🔴 High'))
        st.dataframe(vif_df.style.format({'VIF': '{:.2f}'}).background_gradient(
            subset=['VIF'], cmap='YlOrRd', vmin=1, vmax=10
        ), use_container_width=True)
    
    with col2:
        high_vif = vif_df[vif_df['VIF'] >= 5]
        if len(high_vif) > 0:
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ Multicollinearity Warning:</strong><br>
            Variables with VIF ≥ 5 may have multicollinearity issues: <strong>{', '.join(high_vif['Variable'].tolist())}</strong><br>
            <em>This may affect the reliability of individual coefficient estimates, though the overall model predictions remain valid.</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            <strong>✅ No Multicollinearity Issues:</strong><br>
            All VIF values are below 5, indicating reliable coefficient estimates.
            </div>
            """, unsafe_allow_html=True)
    
    sig_vars = [name for name, p in zip(display_names, stats['p_values']) if p < 0.05]
    nonsig_vars = [name for name, p in zip(display_names, stats['p_values']) if p >= 0.05]
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>📊 GDP Model Variable Significance:</strong>
    <ul>
        <li><strong>Significant predictors (p < 0.05):</strong> {', '.join(sig_vars) if sig_vars else 'None'}</li>
        <li><strong>Non-significant predictors (p ≥ 0.05):</strong> {', '.join(nonsig_vars) if nonsig_vars else 'None'}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_advanced_visualizations(df, numeric_cols):
    st.header("📉 Advanced Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Time Series Analysis", "Distribution Analysis", "Comparative Analysis", 
         "Economic Indicators Dashboard", "Volatility Analysis"]
    )
    
    if viz_type == "Time Series Analysis":
        st.subheader("📈 Time Series Trends")
        
        available_vars = ['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)', 
                          'Interest Rate (%)', 'Crude Oil Price (USD per Barrel)',
                          'Gold Price (USD per Ounce)', 'Consumer Confidence Index',
                          'Unemployment Rate (%)']
        available_vars = [v for v in available_vars if v in df.columns]
        
        selected_series = st.multiselect(
            "Select variables for time series",
            options=available_vars,
            default=available_vars[:3]
        )
        
        if selected_series:
            add_ma = st.checkbox("Add Moving Average", value=True)
            ma_window = st.slider("Moving Average Window", 5, 50, 20) if add_ma else None
            
            for var in selected_series:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df[var],
                                         mode='lines', name=var,
                                         line=dict(color='#1E88E5')))
                
                if add_ma:
                    ma = df[var].rolling(window=ma_window).mean()
                    fig.add_trace(go.Scatter(x=df['Date'], y=ma,
                                             mode='lines', name=f'{ma_window}-period MA',
                                             line=dict(color='#FF5722', dash='dash')))
                
                fig.update_layout(title=f"{var} Over Time", height=350,
                                  xaxis_title="Date", yaxis_title=var)
                st.plotly_chart(fig, use_container_width=True)
                
                trend = "upward" if df[var].iloc[-1] > df[var].iloc[0] else "downward"
                volatility = df[var].std() / df[var].mean() * 100
                st.markdown(f"""
                <div class="insight-box">
                <strong>📌 {var} Insights:</strong>
                <ul>
                    <li>Overall trend: <strong>{trend}</strong></li>
                    <li>Coefficient of Variation: <strong>{volatility:.1f}%</strong> (higher = more volatile)</li>
                    <li>Range: <strong>{df[var].min():.2f}</strong> to <strong>{df[var].max():.2f}</strong></li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Distribution Analysis":
        st.subheader("📊 Distribution Analysis")
        
        dist_var = st.selectbox(
            "Select variable for distribution analysis",
            options=[c for c in numeric_cols if c not in ['Open Price', 'Daily High', 'Daily Low']]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=dist_var, nbins=40, 
                               marginal='box',
                               color_discrete_sequence=['#1E88E5'])
            fig.update_layout(title=f"Distribution of {dist_var}", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.violin(df, y=dist_var, box=True, points='outliers',
                            color_discrete_sequence=['#9C27B0'])
            fig.update_layout(title=f"Violin Plot: {dist_var}", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        skewness = df[dist_var].skew()
        kurtosis = df[dist_var].kurtosis()
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>📌 Statistical Properties of {dist_var}:</strong>
        <ul>
            <li><strong>Mean:</strong> {df[dist_var].mean():.2f}</li>
            <li><strong>Median:</strong> {df[dist_var].median():.2f}</li>
            <li><strong>Std Dev:</strong> {df[dist_var].std():.2f}</li>
            <li><strong>Skewness:</strong> {skewness:.3f} ({'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Approximately symmetric'})</li>
            <li><strong>Kurtosis:</strong> {kurtosis:.3f} ({'Heavy tails/outliers' if kurtosis > 1 else 'Light tails' if kurtosis < -1 else 'Normal-like tails'})</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif viz_type == "Comparative Analysis":
        st.subheader("📊 Comparative Analysis Across Stock Indices")
        
        compare_var = st.selectbox(
            "Select variable to compare",
            options=['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)', 
                     'Trading Volume', 'Consumer Confidence Index']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='Stock Index', y=compare_var, color='Stock Index',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(title=f"{compare_var} by Stock Index", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            agg_df = df.groupby('Stock Index')[compare_var].agg(['mean', 'std', 'min', 'max']).reset_index()
            fig = go.Figure(data=[
                go.Bar(name='Mean', x=agg_df['Stock Index'], y=agg_df['mean'], marker_color='#1E88E5'),
            ])
            fig.add_trace(go.Scatter(
                x=agg_df['Stock Index'], y=agg_df['max'],
                mode='markers', name='Max', marker=dict(size=12, color='#4CAF50', symbol='triangle-up')
            ))
            fig.add_trace(go.Scatter(
                x=agg_df['Stock Index'], y=agg_df['min'],
                mode='markers', name='Min', marker=dict(size=12, color='#F44336', symbol='triangle-down')
            ))
            fig.update_layout(title=f"{compare_var} Statistics by Index", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Summary Statistics by Stock Index")
        summary = df.groupby('Stock Index')[compare_var].describe().round(2)
        st.dataframe(summary, use_container_width=True)
    
    elif viz_type == "Economic Indicators Dashboard":
        st.subheader("🏦 Economic Indicators Dashboard")
        
        indicators = {
            'GDP Growth (%)': {'color': '#43A047', 'icon': '📈'},
            'Inflation Rate (%)': {'color': '#FF9800', 'icon': '💹'},
            'Unemployment Rate (%)': {'color': '#E53935', 'icon': '👥'},
            'Interest Rate (%)': {'color': '#1E88E5', 'icon': '🏦'},
            'Consumer Confidence Index': {'color': '#9C27B0', 'icon': '😊'}
        }
        
        cols = st.columns(len(indicators))
        for col, (ind, props) in zip(cols, indicators.items()):
            with col:
                latest = df[ind].iloc[-1] if len(df) > 0 else 0
                avg = df[ind].mean()
                delta = latest - avg
                st.metric(
                    f"{props['icon']} {ind.split('(')[0].strip()}",
                    f"{latest:.2f}",
                    delta=f"{delta:+.2f} vs avg"
                )
        
        fig = make_subplots(rows=2, cols=3, subplot_titles=list(indicators.keys())[:5] + [''])
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        for (row, col), (ind, props) in zip(positions, indicators.items()):
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df[ind], mode='lines',
                           line=dict(color=props['color']), name=ind),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>📌 Economic Indicators Explanation:</strong>
        <ul>
            <li><strong>GDP Growth:</strong> Measures economic output expansion - positive indicates growth</li>
            <li><strong>Inflation:</strong> Price level changes - moderate inflation (2-3%) is healthy</li>
            <li><strong>Unemployment:</strong> Labor market health - lower is generally better</li>
            <li><strong>Interest Rate:</strong> Cost of borrowing - affects investment and spending</li>
            <li><strong>Consumer Confidence:</strong> Household optimism - drives consumer spending</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif viz_type == "Volatility Analysis":
        st.subheader("📉 Volatility Analysis")
        
        vol_vars = ['Close Price', 'GDP Growth (%)', 'Crude Oil Price (USD per Barrel)', 
                    'Gold Price (USD per Ounce)']
        vol_vars = [v for v in vol_vars if v in df.columns]
        
        selected_vol = st.multiselect("Select variables for volatility analysis",
                                       options=vol_vars, default=vol_vars[:2])
        
        if selected_vol:
            window = st.slider("Rolling Window Size", 5, 50, 20)
            
            for var in selected_vol:
                rolling_std = df[var].rolling(window=window).std()
                rolling_mean = df[var].rolling(window=window).mean()
                cv = (rolling_std / rolling_mean) * 100
                
                fig = make_subplots(rows=2, cols=1, 
                                    subplot_titles=(f'{var} with Rolling Mean', 
                                                    f'Rolling Volatility (Std Dev)'))
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df[var], 
                                         mode='lines', name=var, 
                                         line=dict(color='#1E88E5', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=rolling_mean, 
                                         mode='lines', name=f'{window}-period MA',
                                         line=dict(color='#FF5722', width=2)), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df['Date'], y=rolling_std, 
                                         mode='lines', name='Volatility',
                                         fill='tozeroy',
                                         line=dict(color='#9C27B0')), row=2, col=1)
                
                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                high_vol_periods = df[rolling_std > rolling_std.quantile(0.9)]['Date']
                st.markdown(f"""
                <div class="warning-box">
                <strong>⚠️ High Volatility Periods for {var}:</strong><br>
                Average Volatility: <strong>{rolling_std.mean():.2f}</strong><br>
                Max Volatility: <strong>{rolling_std.max():.2f}</strong><br>
                Number of high volatility periods (>90th percentile): <strong>{len(high_vol_periods)}</strong>
                </div>
                """, unsafe_allow_html=True)

def show_prediction_tool(df):
    st.header("🔮 Prediction Tool")
    
    st.markdown("""
    <div class="insight-box">
    <strong>📌 How to Use:</strong>
    <ul>
        <li>Select a target variable you want to predict</li>
        <li>Choose predictor variables from the available economic indicators</li>
        <li>Enter values for the predictors to get predictions</li>
        <li>The model will use linear regression to make predictions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Select target variable
    target_options = ['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)', 
                     'Unemployment Rate (%)', 'Interest Rate (%)', 
                     'Consumer Spending (Billion USD)']
    available_targets = [t for t in target_options if t in df.columns]
    
    target_var = st.selectbox(
        "Select Target Variable to Predict",
        options=available_targets,
        index=0
    )
    
    # Select predictor variables
    predictor_options = [col for col in df.select_dtypes(include=[np.number]).columns 
                        if col != target_var and col not in ['Open Price', 'Daily High', 'Daily Low', 'Trading Volume']]
    
    selected_predictors = st.multiselect(
        "Select Predictor Variables",
        options=predictor_options,
        default=predictor_options[:5] if len(predictor_options) >= 5 else predictor_options
    )
    
    if len(selected_predictors) == 0:
        st.warning("Please select at least one predictor variable.")
        return
    
    # Prepare data for model training
    all_cols = [target_var] + selected_predictors
    df_clean = df[all_cols].dropna()
    
    if len(df_clean) < 10:
        st.error(f"Insufficient data for prediction. Only {len(df_clean)} valid observations found.")
        return
    
    st.info(f"📊 Training model on {len(df_clean):,} observations")
    
    # Train model
    X = df_clean[selected_predictors].values
    y = df_clean[target_var].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("Training Samples", len(df_clean))
    
    st.divider()
    st.subheader("📝 Enter Values for Prediction")
    
    # Get input values from user
    input_values = {}
    cols = st.columns(min(3, len(selected_predictors)))
    
    for idx, predictor in enumerate(selected_predictors):
        col_idx = idx % len(cols)
        with cols[col_idx]:
            # Get statistics for the predictor
            mean_val = df_clean[predictor].mean()
            min_val = df_clean[predictor].min()
            max_val = df_clean[predictor].max()
            std_val = df_clean[predictor].std()
            
            input_values[predictor] = st.number_input(
                f"{predictor}",
                value=float(mean_val),
                min_value=float(min_val - 2*std_val),
                max_value=float(max_val + 2*std_val),
                step=float(std_val / 10),
                help=f"Range: {min_val:.2f} to {max_val:.2f}, Mean: {mean_val:.2f}"
            )
    
    # Make prediction
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        # Prepare input array
        input_array = np.array([[input_values[p] for p in selected_predictors]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Calculate confidence interval (simplified - using RMSE as proxy)
        confidence_interval = 1.96 * rmse  # 95% confidence interval
        
        st.divider()
        st.subheader("📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="success-box">
            <h3>Predicted {target_var}</h3>
            <h2 style="color: #1E88E5; font-size: 2.5rem;">{prediction:.2f}</h2>
            <p><strong>95% Confidence Interval:</strong><br>
            {prediction - confidence_interval:.2f} to {prediction + confidence_interval:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📋 Input Values Used")
            input_df = pd.DataFrame({
                'Predictor': selected_predictors,
                'Value': [input_values[p] for p in selected_predictors]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        # Show model equation
        st.divider()
        st.subheader("📐 Model Equation")
        equation = f"{target_var} = {model.intercept_:.4f}"
        for i, pred in enumerate(selected_predictors):
            sign = "+" if model.coef_[i] >= 0 else ""
            equation += f" {sign} {abs(model.coef_[i]):.4f} × {pred}"
        
        st.markdown(f'<div class="equation-box">{equation}</div>', unsafe_allow_html=True)
        
        # Show contribution of each predictor
        st.subheader("📊 Contribution of Each Predictor")
        contributions = []
        for i, pred in enumerate(selected_predictors):
            contrib = model.coef_[i] * input_values[pred]
            contributions.append({
                'Predictor': pred,
                'Coefficient': model.coef_[i],
                'Input Value': input_values[pred],
                'Contribution': contrib
            })
        
        contrib_df = pd.DataFrame(contributions)
        contrib_df['Contribution %'] = (contrib_df['Contribution'] / (prediction - model.intercept_) * 100).round(2)
        
        fig = go.Figure()
        colors = ['#4CAF50' if c >= 0 else '#F44336' for c in contrib_df['Contribution']]
        fig.add_trace(go.Bar(
            x=contrib_df['Predictor'],
            y=contrib_df['Contribution'],
            marker_color=colors,
            text=[f"{c:.2f}" for c in contrib_df['Contribution']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Contribution of Each Predictor to Prediction",
            xaxis_title="Predictor Variable",
            yaxis_title="Contribution",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
        
        st.dataframe(contrib_df.style.format({
            'Coefficient': '{:.4f}',
            'Input Value': '{:.2f}',
            'Contribution': '{:.2f}',
            'Contribution %': '{:.2f}%'
        }), use_container_width=True)
    
    st.divider()
    st.subheader("📈 Model Performance Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, y=y_pred,
            mode='markers',
            marker=dict(color='#1E88E5', opacity=0.6),
            name='Predictions'
        ))
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="Actual vs Predicted (Training Data)",
            xaxis_title=f"Actual {target_var}",
            yaxis_title=f"Predicted {target_var}",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        residuals = y - y_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            marker=dict(color='#9C27B0', opacity=0.6),
            name='Residuals'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residuals Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()
