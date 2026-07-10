import streamlit as st
import joblib
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="种子发芽预测系统",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 动态 CSS 样式（支持暗色模式） ====================
dark_mode = st.session_state.get("dark_mode", False)

if dark_mode:
    bg_grad = "linear-gradient(145deg, #1a2a1a 0%, #0f1f0f 100%)"
    card_bg = "rgba(30, 50, 30, 0.80)"
    text_color = "#d0e0d0"
    sub_color = "#a0b8a0"
    input_bg = "rgba(50, 70, 50, 0.70)"
    border_color = "rgba(100, 140, 100, 0.30)"
    box_shadow = "0 30px 60px rgba(0,0,0,0.6)"
else:
    bg_grad = "linear-gradient(145deg, #f0f5ee 0%, #e2ebe0 100%)"
    card_bg = "rgba(255, 255, 255, 0.70)"
    text_color = "#1d3b1d"
    sub_color = "#4d6b4d"
    input_bg = "rgba(255, 255, 255, 0.70)"
    border_color = "rgba(255, 255, 255, 0.50)"
    box_shadow = "0 30px 60px rgba(30, 50, 30, 0.15)"

css = f"""
<style>
    .stApp {{
        background: {bg_grad};
    }}
    .main > div {{
        background: {card_bg};
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 48px;
        padding: 32px 40px;
        border: 1px solid {border_color};
        box-shadow: {box_shadow};
        margin: 20px auto;
        max-width: 1200px;
    }}
    h1, h2, h3, .name, .result-rate {{
        color: {text_color} !important;
    }}
    .sub, .footer, .stNumberInput label {{
        color: {sub_color} !important;
    }}
    .stNumberInput > div > div > input {{
        background: {input_bg} !important;
        border-color: {border_color} !important;
    }}
    .species-card {{
        background: {card_bg};
        border-color: {border_color};
    }}
    /* 可根据需要补充更多样式 */
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==================== 品种配置 ====================
SPECIES_CONFIG = {
    "白沙蒿": {
        "emoji": "🌾",
        "sub": "耐旱固沙 · 保持水土",
        "count": 15,
        "model_path": "BSH_xgboost_optimized_model.joblib"
    },
    "花棒": {
        "emoji": "🌸",
        "sub": "防风固沙 · 生态修复",
        "count": 29,
        "model_path": "HB_xgboost_optimized_model.joblib"
    },
}

# 初始化 session_state
if "selected_species" not in st.session_state:
    st.session_state.selected_species = None

# ==================== 加载模型 ====================
@st.cache_resource
def load_model(model_path):
    if model_path is None or not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# ==================== 预测函数 ====================
def predict(features, species_name):
    config = SPECIES_CONFIG[species_name]
    model = load_model(config["model_path"])
    if model is None:
        return None
    import numpy as np
    X = np.array(features).reshape(1, -1)
    result = model.predict(X)[0]
    return result

# ==================== 圆环图 ====================
def create_donut_chart(percentage, size=120, stroke_width=12):
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.14159 * radius
    offset = circumference - (percentage / 100) * circumference
    color = "#4caf50" if percentage >= 70 else "#ff9800" if percentage >= 50 else "#f44336"
    svg = f"""
    <div style="display: inline-block; position: relative; width: {size}px; height: {size}px;">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle cx="{size/2}" cy="{size/2}" r="{radius}"
                    fill="none" stroke="#e0e8e0" stroke-width="{stroke_width}"/>
            <circle cx="{size/2}" cy="{size/2}" r="{radius}"
                    fill="none" stroke="{color}" stroke-width="{stroke_width}"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                    stroke-linecap="round"
                    transform="rotate(-90 {size/2} {size/2})"
                    style="transition: stroke-dashoffset 0.6s ease;"/>
        </svg>
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                    display: flex; align-items: center; justify-content: center;
                    font-size: {size*0.25}px; font-weight: 700; color: #1d3b1d;">
            {percentage:.1f}%
        </div>
    </div>
    """
    return svg

# ==================== 首页 ====================
def show_home():
    st.markdown('<h1>种子<span>发芽预测</span>系统</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#4d6b4d; margin-bottom:32px;">选择种子品种，输入特征数据，智能预测发芽率</p>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, (name, config) in enumerate(SPECIES_CONFIG.items()):
        with cols[idx % 3]:
            model_available = config["model_path"] is not None and os.path.exists(config["model_path"])
            tag_text = f"{config['count']} 个特征" + (" ✅" if model_available else " ⚠️ 模型缺失")
            st.markdown(f"""
                <div class="species-card">
                    <span class="emoji">{config['emoji']}</span>
                    <div class="name">{name}</div>
                    <div class="sub">{config['sub']}</div>
                    <div class="tag">{tag_text}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"开始预测 →", key=f"btn_{name}", use_container_width=True, disabled=not model_available):
                st.session_state.selected_species = name
                st.rerun()
    
    st.markdown('<div class="footer">🌿 基于机器学习 · 科学种植辅助工具</div>', unsafe_allow_html=True)

# ==================== 预测页（唯一版本，含暗色切换） ====================
def show_predict():
    species = st.session_state.selected_species
    config = SPECIES_CONFIG[species]
    
    # 返回按钮
    if st.button("← 返回品种", key="back_home"):
        st.session_state.selected_species = None
        st.rerun()
    
    # 暗色模式切换（放在标题旁边）
    col_title, col_toggle = st.columns([4, 1])
    with col_title:
        st.markdown(f"<h2>{config['emoji']} {species}</h2>", unsafe_allow_html=True)
    with col_toggle:
        dark = st.toggle("🌙 暗色", value=st.session_state.get("dark_mode", False), key="dark_toggle")
        if dark != st.session_state.get("dark_mode", False):
            st.session_state.dark_mode = dark
            st.rerun()
    
    st.markdown(f'<p style="color:#6f8f6f; margin-bottom:8px;">该品种需要 <strong>{config["count"]}</strong> 个特征值</p>', unsafe_allow_html=True)
    
    # 特征输入
    feature_values = []
    cols = st.columns(4)
    for i in range(config["count"]):
        with cols[i % 4]:
            val = st.number_input(
                f"特征 {i+1}",
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"feat_{species}_{i}"
            )
            feature_values.append(val)
    
    # 预测按钮
    if st.button("🔮 开始预测"):
        with st.spinner("分析中..."):
            rate = predict(feature_values, species)
            if rate is not None:
                donut_html = create_donut_chart(rate, 150, 14)
                if rate >= 70:
                    bg = "rgba(76, 175, 80, 0.15)"
                    border = "#4caf50"
                    advice = "🌟 适宜发芽"
                elif rate >= 50:
                    bg = "rgba(255, 152, 0, 0.15)"
                    border = "#ff9800"
                    advice = "🌤️ 条件一般"
                else:
                    bg = "rgba(244, 67, 54, 0.15)"
                    border = "#f44336"
                    advice = "⚠️ 发芽困难"
                
                st.markdown(f"""
                <div style="background: {bg}; border: 2px solid {border}; border-radius: 28px; 
                            padding: 24px 30px; margin-top: 20px;
                            display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
                    <div style="flex: 0 0 auto;">
                        {donut_html}
                    </div>
                    <div style="flex: 1;">
                        <h3 style="color: {border}; margin: 0;">{advice}</h3>
                        <p style="font-size: 18px; color: #3d5a3d;">预测发芽率 <strong>{rate:.2f}%</strong></p>
                        <p style="color: #6f8f6f;">当前种子{'适宜' if rate>=70 else '发芽条件一般' if rate>=50 else '发芽困难'}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="footer">🌿 预测结果仅供参考 · 实际发芽受多种因素影响</div>', unsafe_allow_html=True)

# ==================== 路由 ====================
if st.session_state.selected_species is None:
    show_home()
else:
    show_predict()