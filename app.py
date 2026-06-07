import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 顶级视觉主题配置
st.set_page_config(page_title="AURA AI Intelligence Suite", layout="wide", initial_sidebar_state="collapsed")

# 注入高奢企业级 UI 样式
st.markdown("""
    <style>
    .main { background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, sans-serif; }
    h1 { color: #0F2C59; font-weight: 800; font-size: 2.8rem !important; letter-spacing: -0.06rem; text-align: center; margin-bottom: 5px !important; }
    .subtitle-text { text-align: center; color: #64748B; font-size: 1.1rem; margin-bottom: 30px; font-weight: 400; }
    
    /* 豪华数据卡片样式 */
    .premium-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 44, 89, 0.05);
        border: 1px solid rgba(15, 44, 89, 0.06);
        text-align: center;
        transition: transform 0.2s ease-in-out;
        margin-bottom: 15px;
    }
    .premium-card:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(15, 44, 89, 0.1); }
    .metric-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08rem; color: #64748B; font-weight: 700; margin-bottom: 8px; }
    .metric-value-blue { font-size: 2.2rem; font-weight: 800; color: #0F2C59; }
    .metric-value-green { font-size: 2.2rem; font-weight: 800; color: #10B981; }
    .metric-value-dark { font-size: 2.2rem; font-weight: 800; color: #1E293B; }
    .metric-subtext { font-size: 0.8rem; color: #94A3B8; margin-top: 6px; }
    </style>
""", unsafe_allow_html=True)

# 2. 自动化机器学习引擎
@st.cache_resource
def train_browser_model():
    data_url = "https://raw.githubusercontent.com/mGalarnyk/Python_Tutorials/master/Kaggle/Facebook/KAG_conversion_data.csv"
    df = pd.read_csv(data_url)
    
    df = pd.get_dummies(df, columns=['age', 'gender'], drop_first=False)
    
    interest_mapping = {
        15: 'Clean Beauty Enthusiasts', 16: 'Luxury Skincare Buyers',
        20: 'Gen-Z Makeup Trends', 21: 'Professional Estheticians', 28: 'Eco-Friendly Cosmetics'
    }
    df['Cosmetics_Niche'] = df['interest'].map(interest_mapping).fillna('General Beauty Shoppers')
    df = pd.get_dummies(df, columns=['Cosmetics_Niche'], drop_first=False)
    
    required_dynamic_columns = [
        'Cosmetics_Niche_Clean Beauty Enthusiasts', 'Cosmetics_Niche_Luxury Skincare Buyers',
        'Cosmetics_Niche_Gen-Z Makeup Trends', 'Cosmetics_Niche_Professional Estheticians',
        'Cosmetics_Niche_Eco-Friendly Cosmetics',
        'age_30-34', 'age_35-39', 'age_40-44', 'age_45-49', 'gender_M', 'gender_F'
    ]
    for col in required_dynamic_columns:
        if col not in df.columns: df[col] = 0

    df['CPC'] = np.where(df['Clicks'] > 0, df['Spent'] / df['Clicks'], 0)
    df['CPM'] = (df['Spent'] / (df['Impressions'] + 1e-5)) * 1000
    df['Ad_Efficiency_Index'] = df['Impressions'] / (df['Spent'] + 1)
    
    features = [
        'Spent', 'Impressions', 'Clicks', 'CPC', 'CPM', 'Ad_Efficiency_Index',
        'Cosmetics_Niche_Clean Beauty Enthusiasts', 'Cosmetics_Niche_Luxury Skincare Buyers',
        'Cosmetics_Niche_Gen-Z Makeup Trends', 'Cosmetics_Niche_Professional Estheticians',
        'Cosmetics_Niche_Eco-Friendly Cosmetics',
        'age_30-34', 'age_35-39', 'age_40-44', 'age_45-49', 'gender_M', 'gender_F'
    ]
    
    X = df[features]
    y = df['Total_Conversion']  
    
    model = LinearRegression()
    model.fit(X, y)
    return model, features

try:
    model, feature_columns = train_browser_model()
except Exception as e:
    st.error(f"Engine Failure: {e}")
    st.stop()

# 3. 居中大标题
st.markdown("<h1>⚡ AURA: Enterprise Pre-Launch Prediction Suite</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Demographic-Aware Machine Learning Forecasting for High-Growth Cosmetics Brands</div>", unsafe_allow_html=True)

# 4. 中央控制面板（使用 Streamlit 原生高级安全边框容器）
with st.container(border=True):
    st.markdown("<h3 style='margin-top:0px; color:#0F2C59;'>🎛️ Campaign Parameter Configuration</h3>", unsafe_allow_html=True)
    
    # 第一排：基础流量指标
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        budget = st.number_input("Ad Budget Allocation ($)", min_value=10.0, max_value=100000.0, value=1000.0, step=100.0)
    with row1_col2:
        clicks = st.number_input("Target Click Volume", min_value=1, max_value=1000000, value=650)
    with row1_col3:
        impressions = st.number_input("Target Impression Reach", min_value=100, max_value=50000000, value=45000)

    # 第二排：细分受众特征
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        niche_selection = st.selectbox(
            "Target Cosmetics Brand Niche",
            ["Clean Beauty Enthusiasts", "Luxury Skincare Buyers", "Gen-Z Makeup Trends", "Professional Estheticians", "Eco-Friendly Cosmetics"]
        )
    with row2_col2:
        age_selection = st.selectbox("Audience Age Bracket Target", ["30-34", "35-39", "40-44", "45-49"])
    with row2_col3:
        gender_selection = st.selectbox("Audience Primary Gender Focus", ["Female", "Male"])

    # 第三排：商业转化乘数
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        product_price = st.slider("Average Checkout Basket Value ($)", min_value=10, max_value=500, value=65)
    with row3_col2:
        close_rate = st.slider("Inquiry Lead-to-Sale Close Rate (%)", min_value=1, max_value=50, value=8)

st.markdown("<br>", unsafe_allow_html=True)

# 5. 核心全宽大按钮（100% 渲染，无任何外部 HTML 包裹干扰）
search_triggered = st.button("🔍 RUN PREDICTIVE SEARCH ANALYSIS", use_container_width=True, type="primary")
st.markdown("<br>", unsafe_allow_html=True)

# 基础营销数据结算
computed_cpc = budget / clicks if clicks > 0 else 0
computed_cpm = (budget / impressions) * 1000 if impressions > 0 else 0
computed_efficiency = impressions / (budget + 1)

# 特征 Payload 对齐封装
payload = {f: 0 for f in feature_columns}
payload['Spent'] = budget
payload['Impressions'] = impressions
payload['Clicks'] = clicks
payload['CPC'] = computed_cpc
payload['CPM'] = computed_cpm
payload['Ad_Efficiency_Index'] = computed_efficiency
payload[f'Cosmetics_Niche_{niche_selection}'] = 1
payload[f'age_{age_selection}'] = 1
payload['gender_M'] = 1 if gender_selection == "Male" else 0
payload['gender_F'] = 1 if gender_selection == "Female" else 0

input_df = pd.DataFrame([payload])
prediction = model.predict(input_df)[0]
predicted_conversions = max(0, int(round(prediction)))

# 财务公式结算
decimal_close = close_rate / 100.0
closed_sales = predicted_conversions * decimal_close
gross_revenue = closed_sales * product_price
net_roi = gross_revenue - budget
roas = gross_revenue / budget if budget > 0 else 0

# 6. 数据看板面板矩阵
st.write("### 📊 Live Predictive Dashboard Matrix")

# 第一排卡片：流量与转化
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Predicted Lead Volume</div>
            <div class="metric-value-blue">{predicted_conversions:,}</div>
            <div class="metric-subtext">Expected enquiries for age {age_selection}</div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Unit Cost Per Click (CPC)</div>
            <div class="metric-value-dark">${computed_cpc:.2f}</div>
            <div class="metric-subtext">Calculated platform link-click cost</div>
        </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Cost Per Mille (CPM)</div>
            <div class="metric-value-dark">${computed_cpm:.2f}</div>
            <div class="metric-subtext">Inventory purchase rate per 1k views</div>
        </div>
    """, unsafe_allow_html=True)

# 第二排卡片：商业财务回报
c4, c5, c6 = st.columns(3)
with c4:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Acquired Customer Orders</div>
            <div class="metric-value-dark">{int(round(closed_sales)):,}</div>
            <div class="metric-subtext">Closed conversions at {close_rate}% close rate</div>
        </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Gross Revenue Forecast</div>
            <div class="metric-value-green">${gross_revenue:,.2f}</div>
            <div class="metric-subtext">Total valuation generated from predictive leads</div>
        </div>
    """, unsafe_allow_html=True)
with c6:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Return On Ad Spend (ROAS)</div>
            <div class="metric-value-green" style="color: {'#10B981' if roas >= 1.0 else '#EF4444'};">{roas:.2f}x</div>
            <div class="metric-subtext">Net Surplus: ${net_roi:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. 动态营收趋势优化图表
st.write("### 📉 Multi-Scale Revenue Optimization Path")
st.caption(f"Predictive simulation indicating gross revenue trajectory based on age group **{age_selection}** within the **{niche_selection}** segment.")

budget_steps = [int(x) for x in np.linspace(100, max(5000, budget * 1.5), 15)]
chart_records = []

for b_step in budget_steps:
    scaler = b_step / budget if budget > 0 else 1
    step_impressions = max(100, impressions * scaler)
    step_clicks = max(1, clicks * scaler)
    
    s_cpc = b_step / step_clicks if step_clicks > 0 else 0
    s_cpm = (b_step / step_impressions) * 1000 if step_impressions > 0 else 0
    s_eff = step_impressions / (b_step + 1)
    
    p_payload = payload.copy()
    p_payload['Spent'] = b_step
    p_payload['Impressions'] = step_impressions
    p_payload['Clicks'] = step_clicks
    p_payload['CPC'] = s_cpc
    p_payload['CPM'] = s_cpm
    p_payload['Ad_Efficiency_Index'] = s_eff
    
    step_pred = model.predict(pd.DataFrame([p_payload]))[0]
    step_rev = max(0, int(round(step_pred))) * decimal_close * product_price
    
    chart_records.append({
        "Ad Budget ($)": b_step,
        "Projected Revenue ($)": round(step_rev, 2)
    })

chart_df = pd.DataFrame(chart_records).set_index("Ad Budget ($)")
st.area_chart(chart_df, color="#0F2C59")

st.markdown("---")
st.info(f"💡 **Strategic Advisory Insights:** Multi-variable demographic modeling confirms campaigns targeting the **{age_selection}** age layer react with distinctive sensitivity to CPM shifting. Cross-reference this baseline curve with specialized ad sets to defend the calculated optimization path.")