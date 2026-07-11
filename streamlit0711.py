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

# ==================== 漂亮的 CSS 样式 ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(145deg, #f0f5ee 0%, #e2ebe0 100%);
    }
    .main > div {
        background: rgba(255, 255, 255, 0.70);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 48px;
        padding: 32px 40px;
        border: 1px solid rgba(255, 255, 255, 0.50);
        box-shadow: 0 30px 60px rgba(30, 50, 30, 0.15);
        margin: 20px auto;
        max-width: 1200px;
    }
    h1 {
        text-align: center;
        font-weight: 700 !important;
        color: #1d3b1d !important;
    }
    h1 span {
        background: linear-gradient(135deg, #2d8a4e, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .species-card {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 28px;
        padding: 24px 16px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.60);
        box-shadow: 0 8px 20px rgba(40, 60, 40, 0.06);
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
    }
    .species-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 44px rgba(40, 80, 40, 0.12);
        border-color: rgba(76, 175, 80, 0.25);
    }
    .species-card .emoji {
        font-size: 48px;
        display: block;
        margin-bottom: 6px;
    }
    .species-card .name {
        font-size: 20px;
        font-weight: 600;
        color: #1d3b1d;
    }
    .species-card .sub {
        font-size: 14px;
        color: #5f7a5f;
        margin-bottom: 12px;
    }
    .species-card .tag {
        display: inline-block;
        background: rgba(76, 175, 80, 0.12);
        padding: 3px 14px;
        border-radius: 30px;
        font-size: 12px;
        color: #4d7a4d;
    }
    .stButton button {
        background: linear-gradient(135deg, #2d8a4e, #3ca55c) !important;
        color: #fff !important;
        font-weight: 600 !important;
        border-radius: 40px !important;
        padding: 12px 48px !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(45, 138, 78, 0.30) !important;
        transition: all 0.3s ease !important;
        font-size: 18px !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 32px rgba(45, 138, 78, 0.40) !important;
    }
    .back-btn {
        display: inline-block;
        background: rgba(255, 255, 255, 0.60);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(200, 215, 200, 0.50);
        padding: 8px 22px;
        border-radius: 40px;
        color: #2d4a2d;
        cursor: pointer;
        transition: all 0.25s ease;
        text-decoration: none;
        margin-bottom: 16px;
    }
    .back-btn:hover {
        background: rgba(255, 255, 255, 0.85);
        transform: translateX(-4px);
    }
    .stNumberInput > div > div > input {
        border-radius: 16px !important;
        border: 1.5px solid rgba(180, 200, 180, 0.40) !important;
        background: rgba(255, 255, 255, 0.70) !important;
        padding: 10px 16px !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.12) !important;
    }
    .result-rate {
        font-size: 52px;
        font-weight: 700;
        background: linear-gradient(135deg, #1d7a3d, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer {
        text-align: center;
        color: #8aaa8a;
        font-size: 13px;
        border-top: 1px solid rgba(180, 200, 180, 0.20);
        padding-top: 20px;
        margin-top: 24px;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 品种配置 ====================
SPECIES_CONFIG = {
    "白沙蒿": {
        "emoji": "🌾",
        "sub": "耐旱固沙 · 保持水土",
        "count": 15,
        "model_path": "BSH_xgboost_optimized_model.joblib",
        # ---------- 新增：15 个特征的具体名称 ----------
        "feature_names": [
            "2,2-二甲基丁酸乙烯酯",
            "2,4-二甲基十二烷",
            "正十五烷",
            "5-甲基正十四烷",
            "碳酸二十烷基异丙烯基酯",
            "Z-9-十八碳烯醛",
            "亚油酸乙酯",
            "油酸缩水甘油酯",
            "Z-7-十六碳烯醛",
            "1-顺式异油酰甘油",
            "4-十四烷基甲氧基乙酸酯",
            "2,3-环氧鲨烯",
            "胆固醇",
            "菜油甾醇",
            "24-亚丙基胆固醇"
        ]
        # ---------------------------------------------
    },
    "花棒": {
        "emoji": "🌸",
        "sub": "防风固沙 · 生态修复",
        "count": 29,
        "model_path": "HB_xgboost_optimized_model.joblib"
        # 没有 feature_names，会使用默认“特征 1…特征 29”
    },
}

# 初始化 session_state
if "selected_species" not in st.session_state:
    st.session_state.selected_species = None


# ==================== 加载模型（带缓存） ====================
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


# ==================== 预测页 ====================
def show_predict():
    species = st.session_state.selected_species
    config = SPECIES_CONFIG[species]
    
    if st.button("← 返回品种", key="back_home"):
        st.session_state.selected_species = None
        st.rerun()
    
    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:24px; flex-wrap:wrap;">
            <span style="font-size:28px; font-weight:700; color:#1d3b1d;">{config['emoji']} {species}</span>
            <span style="background:linear-gradient(135deg,#2d8a4e,#4caf50); color:#fff; padding:4px 20px; border-radius:40px; font-size:16px;">预测模型</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<p style="color:#6f8f6f; margin-bottom:8px;">该品种需要 <strong>{config["count"]}</strong> 个特征值</p>', unsafe_allow_html=True)
    
    # ---------- 获取特征名称列表（若未定义则使用默认） ----------
    feature_names = config.get("feature_names", [])
    # 如果 feature_names 长度不足，用默认补齐
    if len(feature_names) < config["count"]:
        # 补全默认名称
        default_names = [f"特征 {i+1}" for i in range(config["count"])]
        # 将已有的 feature_names 放在前面，后面补默认
        feature_names = feature_names + default_names[len(feature_names):]
    # -------------------------------------------------------
    
    feature_values = []
    cols = st.columns(4)
    for i in range(config["count"]):
        with cols[i % 4]:
            # 使用特征名称
            label = feature_names[i] if i < len(feature_names) else f"特征 {i+1}"
            val = st.number_input(
                label,
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"feat_{species}_{i}"
            )
            feature_values.append(val)
    
    if st.button("🔮 开始预测", type="primary", use_container_width=False):
        with st.spinner("正在分析数据，请稍候…"):
            rate = predict(feature_values, species)
            if rate is None:
                st.error("模型文件缺失，无法进行预测。请检查模型路径。")
            else:
                col_rate, col_status = st.columns([1, 1])
                with col_rate:
                    st.markdown(f'<div class="result-rate">{rate:.2f}%</div>', unsafe_allow_html=True)
                    st.markdown('<span style="color:#4d6b4d;">预测发芽率</span>', unsafe_allow_html=True)
                with col_status:
                    if rate >= 70:
                        st.success("🌟 适宜发芽")
                    elif rate >= 50:
                        st.warning("🌤️ 条件一般")
                    else:
                        st.error("⚠️ 发芽困难")
                
                st.divider()
                col_advice, col_conf = st.columns(2)
                with col_advice:
                    st.metric("📋 建议", "当前种子适宜发芽" if rate >= 70 else "发芽率较低")
    
    st.markdown('<div class="footer">🌿 预测结果仅供参考 · 实际发芽受多种因素影响</div>', unsafe_allow_html=True)


# ==================== 路由 ====================
if st.session_state.selected_species is None:
    show_home()
else:
    show_predict()