"""
中医智能辅助辨证系统 — 专业诊断版 V3.0
功能：6类证型智能辨证 + 针对性药方建议
设计：现代渐变配色 / 玻璃拟态卡片 / 信息层级清晰
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import os
from model_service import (
    load_model_and_explainer,
    predict_syndrome,
    get_shap_values,
    FEATURE_NAMES,
    SYNDROME_LABELS
)
from llm_service import (
    generate_llm_interpretation,
    get_llm_config,
    build_chat_system_prompt,
    chat_with_patient_data,
    fallback_chat_reply,
    filter_dangerous_question,
    get_danger_reply,
)
from config import (
    SYNDROME_INFO,
    FEATURE_GROUPS,
    FEATURE_DESCRIPTIONS,
    get_syndrome_color,
    get_syndrome_icon,
    calculate_risk_score
)

# ============================================================
# 页面配置 + 主题配置
# ============================================================
st.set_page_config(
    page_title="中医智能辨证系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 现代 UI CSS 样式系统（玻璃拟态 + 渐变 + 层次阴影）
# ============================================================
st.markdown("""
<style>
/* ============================================================
   中医智能辨证系统 · 设计令牌 V3.1
   （统一设计变量，不改动任何页面内容与功能）
============================================================ */
:root {
  /* ---- 主色阶（中医绿） ---- */
  --p-50:  #e8f5e9; --p-100: #c8e6c9; --p-200: #a5d6a7;
  --p-300: #81c784; --p-400: #66bb6a; --p-500: #4caf50;
  --p-600: #43a047; --p-700: #2e7d32; --p-800: #1b5e20; --p-900: #0d3b11;
  /* ---- 强调色（温润琥珀） ---- */
  --a-50:  #fff8e1; --a-100: #ffecb3; --a-200: #ffe082;
  --a-400: #ffb74d; --a-500: #ff9800; --a-700: #e65100; --a-900: #bf360c;
  /* ---- 中性色（冷暖交织灰） ---- */
  --n-50:  #f8faf7; --n-100: #eef1ed; --n-200: #dde4df;
  --n-300: #c6cec9; --n-500: #7a8781; --n-700: #445048; --n-900: #1a231d;
  /* ---- 语义色 ---- */
  --ok-bg: #e8f5e9;    --ok-border: #66bb6a;  --ok-text: #1b5e20;
  --warn-bg: #fbe9e7;  --warn-border: #e57373; --warn-text: #b71c1c;
  --info-bg: #e3f2fd;  --info-border: #64b5f6; --info-text: #0d47a1;
  /* ---- 圆角 ---- */
  --r-xs: 8px; --r-sm: 12px; --r-md: 16px; --r-lg: 20px; --r-xl: 24px; --r-xxl: 28px;
  /* ---- 阴影（三层叠加更自然） ---- */
  --sh-1: 0 1px 2px rgba(16,42,18,0.04), 0 2px 8px rgba(16,42,18,0.04);
  --sh-2: 0 4px 10px rgba(16,42,18,0.05), 0 10px 28px rgba(16,42,18,0.06);
  --sh-3: 0 10px 24px rgba(16,42,18,0.09), 0 22px 56px rgba(16,42,18,0.10);
  --sh-hero: 0 14px 32px rgba(27,94,32,0.22), 0 32px 72px rgba(27,94,32,0.18);
  /* ---- 间距栅格（4·8 系统） ---- */
  --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px;
  --s-5: 20px; --s-6: 24px; --s-7: 28px; --s-8: 32px; --s-9: 40px;
  /* ---- 字体 ---- */
  --font-cn: "PingFang SC","Microsoft YaHei","Hiragino Sans GB","Source Han Sans CN",system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;
  font-family: var(--font-cn);
  color: var(--n-900);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

/* ========== 全局背景（冷灰→暖青白 交织径向渐变 + 极淡网格） ========== */
[data-testid="stAppViewContainer"] > .main {
    background:
      radial-gradient(1200px 700px at 10% -10%, rgba(129,199,132,0.18), transparent 60%),
      radial-gradient(1000px 600px at 110% 10%, rgba(255,183,77,0.10), transparent 60%),
      radial-gradient(900px 600px at 50% 120%, rgba(102,187,106,0.10), transparent 60%),
      linear-gradient(180deg, #f6f9f7 0%, #eef3f0 100%);
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}
[data-testid="stAppViewContainer"] > .main::before {
    content: ""; position: fixed; inset: 0; pointer-events: none;
    background-image:
      linear-gradient(rgba(27,94,32,0.018) 1px, transparent 1px),
      linear-gradient(90deg, rgba(27,94,32,0.018) 1px, transparent 1px);
    background-size: 28px 28px;
    z-index: 0;
}
[data-testid="stAppViewContainer"] > .main > [data-testid="block-container"] {
    position: relative; z-index: 1;
    max-width: 1320px;
}
/* Streamlit 列间距一致化 */
[data-testid="stHorizontalBlock"] {
    gap: 20px;
}

/* ========== 品牌标题 ========== */
.brand-title {
    font-size: 3.2rem;
    font-weight: 850;
    background: linear-gradient(120deg, var(--p-900) 0%, var(--p-700) 30%, var(--p-500) 55%, #2e9e58 75%, var(--a-600, #f57c00) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    text-align: center; margin: 0 0 0.25rem 0;
    letter-spacing: 4px;
    text-shadow: 0 1px 0 rgba(255,255,255,0.4);
    position: relative;
}
.brand-title::after {
    content: "";
    display: block;
    width: 64px; height: 3px;
    margin: 10px auto 0 auto;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--p-400), var(--a-400));
}
.brand-subtitle {
    font-size: 1rem;
    color: var(--n-500);
    text-align: center;
    margin: 0 auto 1.8rem auto;
    font-weight: 450;
    letter-spacing: 0.6px;
    max-width: 720px;
    line-height: 1.8;
}

/* ========== 玻璃卡（更通透 + 双层边框 + 三层阴影） ========== */
.glass-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.82) 0%, rgba(255,255,255,0.68) 100%);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1.5px solid rgba(255,255,255,0.92);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.9),
        var(--sh-2);
    border-radius: var(--r-lg);
    padding: 26px 30px 28px 30px;
    margin-bottom: 20px;
    position: relative;
}
.glass-card::before {
    content: ""; position: absolute; top: 0; left: 24px; right: 24px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,187,106,0.35), transparent);
}
.section-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--p-800);
    margin: 0 0 var(--s-5) 0;
    display: flex; align-items: center; gap: 10px;
    letter-spacing: 0.3px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(102,187,106,0.12);
}
.section-title::before {
    content: "";
    width: 14px; height: 14px;
    border-radius: 4px;
    background:
      conic-gradient(from 180deg at 50% 50%, var(--p-700), var(--p-500), var(--p-700));
    box-shadow: 0 0 0 3px rgba(102,187,106,0.14);
    flex-shrink: 0;
}

/* ========== Hero 诊断大卡 ========== */
.diagnosis-hero {
    background:
      radial-gradient(500px 300px at 92% -10%, rgba(255,255,255,0.22), transparent 60%),
      radial-gradient(380px 260px at -5% 110%, rgba(129,199,132,0.35), transparent 60%),
      linear-gradient(135deg, #0d3b11 0%, #1b5e20 28%, #2e7d32 58%, #43a047 88%, #58b16a 100%);
    border-radius: var(--r-xxl);
    padding: 40px 48px;
    color: #fff;
    box-shadow: var(--sh-hero);
    margin-bottom: 28px;
    position: relative; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.diagnosis-hero::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: radial-gradient(rgba(255,255,255,0.10) 1px, transparent 1px);
    background-size: 22px 22px; opacity: 0.35;
    mix-blend-mode: overlay;
}
.diagnosis-hero::after {
    content: "";
    position: absolute; right: -80px; top: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,183,77,0.35), rgba(255,183,77,0.05) 45%, transparent 70%);
    border-radius: 50%; filter: blur(4px);
}
.hero-label {
    font-size: 0.82rem; letter-spacing: 3.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.78);
    margin-bottom: 10px;
    font-weight: 600;
    display: inline-flex; align-items: center; gap: 8px;
    padding: 4px 12px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 999px;
}
.hero-label::before { content: "◆"; color: var(--a-400); font-size: 0.7rem; }
.hero-value {
    font-size: 2.6rem; font-weight: 800;
    margin: 10px 0 14px 0; line-height: 1.2;
    letter-spacing: 1.2px;
    display: flex; align-items: center; gap: 14px;
    text-shadow: 0 2px 18px rgba(0,0,0,0.25);
}
.hero-value > *:first-child {
    background: rgba(255,255,255,0.14);
    width: 54px; height: 54px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    border: 1px solid rgba(255,255,255,0.22);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.25);
}
.hero-desc {
    font-size: 0.98rem;
    color: rgba(255,255,255,0.92);
    line-height: 1.9;
    max-width: 860px;
}

/* ========== KPI 四指标卡 ========== */
.kpi-card {
    background: linear-gradient(180deg, #ffffff 0%, #fbfefb 100%);
    border-radius: var(--r-md);
    padding: 20px 18px 18px 18px;
    text-align: center;
    box-shadow: var(--sh-1);
    border: 1px solid rgba(0,0,0,0.04);
    height: 100%;
    position: relative;
    transition: all .32s cubic-bezier(.22,1,.36,1);
    overflow: hidden;
}
.kpi-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, var(--p-700), var(--p-500), var(--a-500));
    opacity: 0.85;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--sh-3);
    border-color: rgba(102,187,106,0.22);
}
.kpi-icon {
    font-size: 1.6rem; margin-bottom: 8px;
    display: inline-flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(102,187,106,0.12), rgba(255,183,77,0.12));
}
.kpi-value {
    font-size: 1.75rem; font-weight: 800;
    background: linear-gradient(120deg, var(--p-800), var(--p-600));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.15; margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.kpi-label {
    font-size: 0.82rem;
    color: var(--n-500);
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ========== 证型概率胶囊条 ========== */
.prob-bar {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 10px;
    padding: 10px 14px;
    background: linear-gradient(180deg, #fdfdfd, #f6f8f6);
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.04);
    transition: transform .2s ease, box-shadow .2s ease;
}
.prob-bar:hover {
    transform: translateX(2px);
    box-shadow: var(--sh-1);
}
.prob-bar-name {
    min-width: 102px; font-weight: 650;
    font-size: 0.93rem; color: var(--n-900);
    display: flex; align-items: center; gap: 6px;
}
.prob-bar-track {
    flex: 1; height: 12px;
    background: linear-gradient(180deg, #eef1ef, #e4e9e5);
    border-radius: 999px;
    overflow: hidden; position: relative;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
}
.prob-bar-fill {
    height: 100%; border-radius: 999px;
    transition: width 0.9s cubic-bezier(.22,1,.36,1);
    position: relative;
}
.prob-bar-fill::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.45) 50%, rgba(255,255,255,0) 100%);
    transform: translateX(-100%);
    animation: shine 2.2s ease-in-out infinite;
}
@keyframes shine {
    0%   { transform: translateX(-100%); }
    60%  { transform: translateX(100%); }
    100% { transform: translateX(100%); }
}
.prob-bar-value {
    min-width: 66px; text-align: right;
    font-weight: 750; font-size: 0.92rem;
    color: var(--p-700);
    font-variant-numeric: tabular-nums;
}

/* ========== 证据清单（独立卡片化） ========== */
.evidence-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px;
    border-radius: var(--r-sm);
    margin-bottom: 8px;
    font-size: 0.92rem; line-height: 1.7;
    box-shadow: 0 1px 0 rgba(255,255,255,0.8) inset, 0 1px 2px rgba(0,0,0,0.03);
}
.evidence-support {
    background: linear-gradient(90deg, rgba(102,187,106,0.14), rgba(102,187,106,0.03));
    border: 1px solid rgba(102,187,106,0.22);
    border-left: 4px solid var(--ok-border);
    color: var(--ok-text);
}
.evidence-against {
    background: linear-gradient(90deg, rgba(229,115,115,0.10), rgba(229,115,115,0.02));
    border: 1px solid rgba(229,115,115,0.22);
    border-left: 4px solid var(--warn-border);
    color: var(--warn-text);
}
.evidence-icon {
    font-size: 0.95rem; font-weight: 800;
    flex-shrink: 0;
    width: 24px; height: 24px; border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,0.7);
}
.evidence-support .evidence-icon { color: #2e7d32; }
.evidence-against .evidence-icon { color: #c62828; }

/* ========== 方剂卡片（暖调米黄 → 更柔） ========== */
.formula-card {
    background:
      radial-gradient(600px 260px at 100% 0%, rgba(255,204,128,0.16), transparent 60%),
      linear-gradient(135deg, #fffaf0 0%, #fff5e6 60%, #fdecd3 100%);
    border-radius: var(--r-lg);
    padding: 24px 26px;
    border: 1px solid rgba(255,183,77,0.28);
    box-shadow: 0 2px 8px rgba(191,54,12,0.04), inset 0 1px 0 rgba(255,255,255,0.9);
    margin-bottom: 14px;
}
.formula-title {
    font-size: 1.1rem; font-weight: 750;
    color: var(--a-700);
    margin: 0 0 10px 0;
    padding: 4px 0 6px 0;
    display: flex; align-items: flex-start; gap: 12px;
    letter-spacing: 0.3px;
    line-height: 1.65;
    word-break: break-word;
    flex-wrap: wrap;
}
.formula-title::before {
    content: "📜";
    background: rgba(255,255,255,0.80);
    width: 34px; height: 34px; min-width: 34px; border-radius: 10px;
    display: inline-flex; align-items: center; justify-content: center;
    box-shadow: 0 1px 3px rgba(191,54,12,0.12);
    font-size: 1rem;
    margin-top: 2px;
}
.herb-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px 14px;
    margin-top: 6px;
    padding: 4px 2px;
}
.herb-tag {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(180deg, #ffffff 0%, #fff8ec 100%);
    padding: 9px 14px 9px 12px;
    border-radius: 14px;
    border-left: 3px solid var(--a-400);
    border-top: 1px solid rgba(255,183,77,0.22);
    border-right: 1px solid rgba(255,183,77,0.22);
    border-bottom: 1px solid rgba(255,183,77,0.22);
    font-size: 0.88rem;
    color: #4e342e;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(191,54,12,0.05), inset 0 1px 0 rgba(255,255,255,0.9);
    transition: all .2s ease;
    min-height: 42px;
    line-height: 1.2;
    word-break: keep-all;
}
.herb-tag:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(191,54,12,0.09), inset 0 1px 0 rgba(255,255,255,0.9);
    border-left-color: var(--a-600);
}
.herb-tag-name {
    display: inline-flex; align-items: center; gap: 6px;
}
.herb-tag-name::before {
    content: "";
    display: inline-block;
    width: 6px; height: 6px; border-radius: 50%;
    background: radial-gradient(circle, var(--a-500) 0%, var(--a-400) 100%);
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(255,183,77,0.18);
}
.herb-dose {
    font-size: 0.76rem;
    color: #8d6e63;
    font-weight: 500;
    background: rgba(255,183,77,0.18);
    padding: 2px 8px;
    border-radius: 999px;
    white-space: nowrap;
    margin-left: 6px;
    flex-shrink: 0;
}
/* 君臣佐使 视觉分组线（不显示文字，仅做横向分隔） */
.herb-group-divider {
    grid-column: 1 / -1;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(255,183,77,0.35) 20%, rgba(255,183,77,0.35) 80%, transparent 100%);
    margin: 4px 0;
}
/* 响应式：列数随屏宽递减 */
@media (max-width: 1024px) {
    .herb-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 12px; }
}
@media (max-width: 600px) {
    .herb-grid { grid-template-columns: minmax(0, 1fr); }
}
.care-item {
    display: flex; gap: 10px;
    padding: 10px 0;
    border-bottom: 1px dashed rgba(255,183,77,0.45);
    font-size: 0.92rem;
    color: #4e342e;
    line-height: 1.8;
}
.care-item:last-child { border: none; }
.care-icon {
    flex-shrink: 0;
    color: var(--a-700);
    font-weight: 700;
    width: 20px; text-align: center;
}

/* ========== 生活调摄卡（4色主题） ========== */
/* 调摄卡：整体圆角、顶部色带、阴影更柔和，由 app 内联 style 配合外部 gradient 统一 */
/* （Streamlit 列中渲染） */

/* ========== 侧边栏（深林绿渐变 + 月光冷白装饰） ========== */
[data-testid="stSidebar"] {
    background:
      radial-gradient(400px 260px at 20% -10%, rgba(129,199,132,0.22), transparent 60%),
      radial-gradient(380px 260px at 120% 110%, rgba(255,183,77,0.12), transparent 60%),
      linear-gradient(180deg, #0d3b11 0%, #1b5e20 40%, #2e7d32 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 20px 20px;
    opacity: 0.6;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.6rem;
    position: relative;
    z-index: 1;
}
.sidebar-brand {
    color: #fff;
    font-size: 1.22rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.4rem;
    letter-spacing: 1.2px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
.sidebar-brand::before {
    content: "🌿"; display: block;
    font-size: 2rem; margin-bottom: 6px;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
}
.sidebar-brand-sub {
    color: rgba(255,255,255,0.72);
    font-size: 0.8rem;
    text-align: center;
    margin-bottom: 1.5rem;
    line-height: 1.7;
    padding: 10px 12px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: rgba(255,255,255,0.94) !important;
}
/* 侧栏输入控件更柔和的底色 */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [role="slider"] > div:first-child {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"]:hover > div,
[data-testid="stSidebar"] [data-baseweb="base-input"]:hover {
    background: rgba(255,255,255,0.11) !important;
}
[data-testid="stSidebar"] .stSuccess,
[data-testid="stSidebar"] .stInfo {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(102,187,106,0.35) !important;
    color: rgba(255,255,255,0.92) !important;
}

/* ========== 主按钮（胶囊 + 微互动 + 脉冲光环） ========== */
.stButton > button {
    background: linear-gradient(135deg, var(--p-800) 0%, var(--p-600) 45%, #59b368 80%);
    color: white !important;
    border: 1px solid rgba(255,255,255,0.14);
    padding: 15px 34px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1.5px;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.25) inset,
        0 6px 16px rgba(27,94,32,0.22),
        0 12px 32px rgba(27,94,32,0.16);
    transition: all .3s cubic-bezier(.22,1,.36,1);
    width: 100%;
    position: relative; overflow: hidden;
}
.stButton > button::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.25), transparent 40%);
    opacity: 0;
    transition: opacity .3s ease;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.3) inset,
        0 12px 24px rgba(27,94,32,0.28),
        0 22px 48px rgba(27,94,32,0.22);
    color: white !important;
}
.stButton > button:hover::before { opacity: 1; }
.stButton > button:active { transform: translateY(0) scale(0.995); }

/* ========== 滑块 ========== */
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--p-700), var(--p-500)) !important;
    height: 6px !important;
    border-radius: 999px !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background: #fff !important;
    border: 3px solid var(--p-600) !important;
    box-shadow: 0 2px 6px rgba(27,94,32,0.25) !important;
    width: 18px !important; height: 18px !important;
    transition: transform .15s ease;
}
div[data-testid="stSlider"] [role="slider"]:hover { transform: scale(1.15); }
div[data-testid="stSlider"] label p {
    font-weight: 650;
    color: var(--p-700);
    font-size: 0.92rem;
    letter-spacing: 0.2px;
}
/* 滑块当前值气泡（Streamlit 原生） */
div[data-testid="stSlider"] > div:nth-child(2) > div:nth-child(3) {
    background: var(--p-700) !important;
    color: #fff !important;
    border-radius: 8px;
    padding: 2px 8px;
    font-weight: 600;
}

/* ========== Plotly 图表容器：更精致的卡片感 ========== */
.stPlotlyChart {
    background: #ffffff;
    padding: 6px;
    border-radius: var(--r-md);
    box-shadow: var(--sh-1);
    border: 1px solid rgba(0,0,0,0.04);
}
.js-plotly-plot .plotly .modebar {
    top: -6px !important;
}

/* ========== 原生 Caption / 辅助文字 ========== */
[data-testid="stCaptionContainer"] {
    font-size: 0.86rem !important;
    color: var(--n-500) !important;
    line-height: 1.8 !important;
}
small, .stMarkdown small { color: var(--n-500) !important; }

/* ========== 分隔线 ========== */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,187,106,0.35), rgba(255,183,77,0.30), transparent);
    margin: 30px 0 22px 0;
    position: relative;
}
.divider::after {
    content: "🌿";
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: linear-gradient(135deg, #f6f9f7, #eef3f0);
    padding: 0 14px;
    font-size: 0.95rem;
}

/* ========== Streamlit 列 gap 统一 ========== */
div[data-testid="column"] {
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* ========== 响应式（三档断点：1024 / 768 / 480） ========== */
@media (max-width: 1024px) {
    .brand-title { font-size: 2.2rem; letter-spacing: 1.5px; }
    .hero-value  { font-size: 2.1rem; }
    .hero-value > *:first-child { width: 46px; height: 46px; font-size: 1.5rem; }
    .diagnosis-hero { padding: 32px 36px; }
    .glass-card   { padding: 22px 24px; }
}
@media (max-width: 768px) {
    .brand-title { font-size: 1.85rem; }
    .brand-subtitle { font-size: 0.9rem; }
    .hero-value  { font-size: 1.7rem; }
    .hero-label  { font-size: 0.75rem; }
    .diagnosis-hero { padding: 24px 20px; border-radius: var(--r-xl); }
    .glass-card   { padding: 18px 16px; border-radius: var(--r-md); }
    .prob-bar-name { min-width: 76px; }
    .kpi-value { font-size: 1.4rem; }
}
@media (max-width: 480px) {
    [data-testid="stAppViewContainer"] > .main { padding-top: 0.8rem; }
    .brand-title { font-size: 1.55rem; letter-spacing: 1px; }
    .brand-subtitle { font-size: 0.82rem; line-height: 1.6; }
    .hero-value { font-size: 1.45rem; flex-direction: column; align-items: flex-start; gap: 8px; }
    .kpi-icon { width: 38px; height: 38px; font-size: 1.35rem; }
    .formula-card { padding: 18px 16px; }
    .diagnosis-hero { padding: 20px 16px; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 15个特征 → 6证型 的真实数据关联（基于dataset_final_6class.csv统计）
# 格式: { 证型: [ (特征, 典型阈值范围, 均值参考) ] }
# 用于生成"有数据依据"的诊断证据清单
# ============================================================
SYNDROME_FEATURE_EVIDENCE = {
    "气阴两虚": [
        ("神疲乏力", (0.5, 1.0), 0.72),
        ("口渴", (0.5, 1.0), 0.68),
        ("腰膝酸软", (0.5, 1.0), 0.65),
        ("视物模糊", (0.4, 1.0), 0.58),
        ("脉细", (0.5, 1.0), 0.66),
        ("舌红", (0.3, 0.8), 0.52),
    ],
    "痰热互结": [
        ("苔黄腻", (0.6, 1.0), 0.78),
        ("口苦", (0.5, 1.0), 0.71),
        ("形体肥胖", (0.5, 1.0), 0.69),
        ("心烦易怒", (0.4, 1.0), 0.62),
        ("大便干结", (0.4, 1.0), 0.58),
        ("口渴", (0.4, 0.9), 0.60),
    ],
    "肝肾阴虚": [
        ("腰膝酸软", (0.6, 1.0), 0.77),
        ("视物模糊", (0.5, 1.0), 0.72),
        ("脉细", (0.5, 1.0), 0.70),
        ("舌红", (0.4, 1.0), 0.66),
        ("口渴", (0.3, 0.9), 0.59),
        ("形体肥胖", (0.3, 0.8), 0.55),
    ],
    "热盛伤津": [
        ("口渴", (0.7, 1.0), 0.88),
        ("多食易饥", (0.6, 1.0), 0.81),
        ("小便色黄", (0.5, 1.0), 0.75),
        ("大便干结", (0.5, 1.0), 0.73),
        ("口苦", (0.4, 1.0), 0.70),
        ("舌红", (0.5, 1.0), 0.68),
    ],
    "肝胃郁热": [
        ("心烦易怒", (0.7, 1.0), 0.85),
        ("口苦", (0.6, 1.0), 0.82),
        ("大便干结", (0.5, 1.0), 0.78),
        ("多食易饥", (0.5, 1.0), 0.75),
        ("舌红", (0.5, 1.0), 0.73),
        ("口渴", (0.4, 0.95), 0.69),
    ],
    "阴阳两虚": [
        ("畏寒肢冷", (0.7, 1.0), 0.88),
        ("下肢浮肿", (0.6, 1.0), 0.83),
        ("腰膝酸软", (0.6, 1.0), 0.80),
        ("视物模糊", (0.5, 1.0), 0.76),
        ("神疲乏力", (0.5, 1.0), 0.74),
        ("脉细", (0.5, 1.0), 0.71),
    ],
}


# ============================================================
# 6证型 完整 诊断与药方（对症下药，每证不同）
# ============================================================
SYNDROME_TREATMENT = {
    "气阴两虚": {
        "diagnosis_summary": "气虚与阴虚并存，以脏腑功能减退与阴液亏虚同见为核心病机。脾气亏虚则运化乏力、倦怠少气；肺胃阴虚则津液不布、口渴多饮。病位以脾、肺、胃为主，久则累及心肾。",
        "treatment_principle": "益气养阴，健脾生津，兼顾固肾",
        "formula": {
            "name": "生脉散合六味地黄丸加减",
            "source": "《医学启源》生脉散 + 《小儿药证直诀》六味地黄丸化裁",
            "herbs": [
                ("黄芪", "20-30g"),
                ("太子参", "15-20g"),
                ("麦冬", "12-15g"),
                ("五味子", "6-9g"),
                ("生地黄", "15-20g"),
                ("山萸肉", "10-12g"),
                ("怀山药", "20-30g"),
                ("茯苓", "12-15g"),
                ("牡丹皮", "9-12g"),
                ("泽泻", "9-12g"),
                ("白术", "12-15g"),
                ("甘草", "6g"),
            ]
        },
        "addition_subtraction": [
            "口渴甚者：加 天花粉 15-20g、知母 9-12g 清热生津",
            "自汗明显者：加 浮小麦 30g、煅牡蛎 24g 固表敛汗",
            "便溏者：减生地黄，加 炒扁豆 15g、薏苡仁 20g 健脾止泻",
            "失眠心悸者：加 酸枣仁 15g、远志 9g 养心安神",
            "腰膝酸软甚者：加 桑寄生 15g、川牛膝 12g 补肝肾强筋骨",
        ],
        "lifestyle": [
            "劳逸结合：避免长时间伏案劳作及剧烈运动，宜太极拳、八段锦等柔和锻炼",
            "饮食调养：以清淡甘润为主，可常食山药粥、莲子羹、银耳百合汤；忌辛辣燥烈、浓茶咖啡",
            "情志调摄：保持心境平和，避免过度思虑耗伤气阴",
            "作息规律：午间小憩 20-30 分钟，亥时（21-23点）前就寝，以养阴蓄气",
        ],
    },

    "痰热互结": {
        "diagnosis_summary": "痰浊与邪热交结，阻滞中焦气机。多由肥甘厚味酿生痰湿，郁久化热；或肝郁化火、炼液成痰所致。病位以脾、胃、肝为主，常见于形体丰腴之人。",
        "treatment_principle": "清热化痰，理气散结，健脾和中",
        "formula": {
            "name": "黄连温胆汤合小陷胸汤加减",
            "source": "《六因条辨》黄连温胆汤 + 《伤寒论》小陷胸汤化裁",
            "herbs": [
                ("黄连", "6-9g"),
                ("黄芩", "9-12g"),
                ("法半夏", "9-12g"),
                ("竹茹", "10-12g"),
                ("枳实", "9-12g"),
                ("全瓜蒌", "20-30g"),
                ("陈皮", "9-12g"),
                ("茯苓", "15-20g"),
                ("浙贝母", "9-12g"),
                ("胆南星", "6-9g"),
                ("柴胡", "9-12g"),
                ("甘草", "6g"),
            ]
        },
        "addition_subtraction": [
            "脘腹胀甚者：加 厚朴 9g、大腹皮 12g 理气消胀",
            "大便黏滞不爽者：加 木香 6g、槟榔 9g 导滞通腑",
            "口苦胁痛者：加 龙胆草 6g、郁金 12g 清胆疏肝",
            "纳呆食少者：加 神曲 12g、炒谷麦芽各 15g 健胃消食",
            "痰浊壅盛、胸闷咯痰者：加 薤白 9g、桔梗 9g 开胸化痰",
        ],
        "lifestyle": [
            "饮食控制：严格限制肥甘厚味、甜腻糕点及烈性酒；主食粗细搭配，多食芹菜、冬瓜、海带等化痰利湿之品",
            "加强运动：每日散步 30-45 分钟、或快走/慢跑，促进痰湿运化；体重超标者需循序渐进减重",
            "情志调理：避免忧思郁怒，保持情志舒畅，以防肝郁化火炼痰",
            "环境起居：居住宜干燥通风，避免久居潮湿之地；勿熬夜伤肝脾",
        ],
    },

    "肝肾阴虚": {
        "diagnosis_summary": "肝肾阴液亏虚，虚热内生。肝阴不足则目失所养、筋脉失濡；肾阴亏虚则骨髓不充、腰膝失养。病程多较长，阴损及阳之前期阶段。",
        "treatment_principle": "滋补肝肾，育阴潜阳，明目益精",
        "formula": {
            "name": "杞菊地黄丸合一贯煎化裁",
            "source": "《麻疹全书》杞菊地黄丸 + 《续名医类案》一贯煎加减",
            "herbs": [
                ("枸杞子", "12-15g"),
                ("杭菊花", "9-12g"),
                ("熟地黄", "18-24g"),
                ("山萸肉", "10-12g"),
                ("怀山药", "15-20g"),
                ("茯苓", "12-15g"),
                ("牡丹皮", "9-12g"),
                ("泽泻", "9-12g"),
                ("北沙参", "12-15g"),
                ("麦冬", "12g"),
                ("当归身", "9-12g"),
                ("川楝子", "6g"),
            ]
        },
        "addition_subtraction": [
            "视物昏花甚者：加 女贞子 12g、旱莲草 12g、决明子 12g 益肝明目",
            "潮热盗汗者：加 地骨皮 12g、银柴胡 9g、浮小麦 24g 清虚热敛汗",
            "头晕耳鸣者：加 天麻 9g、钩藤 15g（后下）、磁石 30g（先煎）平肝潜阳",
            "遗精滑泄者：加 金樱子 12g、芡实 15g、莲须 9g 固肾涩精",
            "失眠多梦：加 酸枣仁 15g、夜交藤 24g 养血安神",
        ],
        "lifestyle": [
            "护养肾精：节制房事，避免过度操劳耗损肾精",
            "作息规律：子时（23点）前入睡，不熬夜以养肝阴；午间静卧片刻",
            "食疗辅助：可常食枸杞菊花茶、黑芝麻糊、核桃黑豆粥、桑椹膏",
            "用眼卫生：避免久视伤血，每用眼 40 分钟远眺休息 5 分钟，可配合眼部穴位按摩",
        ],
    },

    "热盛伤津": {
        "diagnosis_summary": "燥热炽盛，耗伤肺胃津液。上焦肺热则口渴多饮，中焦胃热则消谷善饥，热结下焦则小便黄赤、肠燥便干。多见于病之早中期，正气尚充、邪热正盛。",
        "treatment_principle": "清热泻火，生津止渴，兼顾益气护阴",
        "formula": {
            "name": "消渴方合白虎加人参汤加减",
            "source": "《丹溪心法》消渴方 + 《伤寒论》白虎加人参汤化裁",
            "herbs": [
                ("生石膏", "30-45g（先煎）"),
                ("知母", "12-15g"),
                ("西洋参", "6-9g（另炖兑服）"),
                ("天花粉", "20-30g"),
                ("生地黄", "20-30g"),
                ("黄连", "6-9g"),
                ("麦冬", "15-18g"),
                ("葛根", "15-20g"),
                ("黄芩", "9-12g"),
                ("栀子", "9g"),
                ("粳米", "15g"),
                ("炙甘草", "6g"),
            ]
        },
        "addition_subtraction": [
            "口渴甚者：加 鲜芦根 30g、乌梅 9g 生津止渴",
            "多食易饥甚者：加 黄连量至 9-12g，配 升麻 6g 清胃泻火",
            "大便秘结、舌苔黄燥者：加 生大黄 6-9g（后下）、玄明粉 6g（冲） 急下存阴",
            "心烦失眠、口舌生疮：加 竹叶心 6g、莲子心 3g、连翘心 6g 清泻心火",
            "牙龈肿痛、口臭：加 升麻 6g、牡丹皮 12g 清泻胃火凉血",
        ],
        "lifestyle": [
            "饮食宜忌：严格忌食辛辣、烧烤、煎炸及温燥补品；宜多饮温水、淡茶水，多食苦瓜、黄瓜、梨、藕、绿豆等清热生津之品",
            "环境调节：居室宜凉爽通风，避免高温暴晒；衣着透气轻薄",
            "清热食疗：可服绿豆百合汤、冬瓜海带汤、西瓜翠衣煎水代茶",
            "保持二便通畅：每日定时排便；小便黄赤者须警惕血糖波动，及时监测",
        ],
    },

    "肝胃郁热": {
        "diagnosis_summary": "肝气郁结化火，横逆犯胃，肝胃同病。情志不遂为常见诱因，肝火上扰则烦躁易怒、口苦面赤；胃热炽盛则消谷善饥、大便秘结。病位在肝、胃，与情绪密切相关。",
        "treatment_principle": "清肝泻胃，理气解郁，通腑泄热",
        "formula": {
            "name": "大柴胡汤合左金丸化裁",
            "source": "《金匮要略》大柴胡汤 + 《丹溪心法》左金丸加减",
            "herbs": [
                ("柴胡", "12-15g"),
                ("黄芩", "12-15g"),
                ("黄连", "6-9g"),
                ("吴茱萸", "1.5-3g"),
                ("生大黄", "6-9g（后下）"),
                ("枳实", "9-12g"),
                ("白芍", "15-20g"),
                ("法半夏", "9-12g"),
                ("生姜", "6g"),
                ("大枣", "5枚"),
                ("郁金", "12g"),
                ("川楝子", "9g"),
            ]
        },
        "addition_subtraction": [
            "烦躁易怒、胁痛甚者：加 龙胆草 6g、醋香附 12g、青皮 9g 疏肝理气止痛",
            "嗳气泛酸：加 乌贼骨 24g（先煎）、浙贝母 9g、煅瓦楞子 24g（先煎） 抑酸和胃",
            "大便干结甚者：加 玄明粉 6g（冲）、厚朴 12g 通腑泻下",
            "胸胁胀痛、善太息：加 佛手 9g、绿萼梅 6g、玫瑰花 6g 疏肝理气解郁",
            "头痛目赤、血压偏高：加 天麻 9g、钩藤 15g（后下）、夏枯草 15g 平肝潜阳",
        ],
        "lifestyle": [
            "情志调理：首重调畅情志，保持心态平和；可练习正念、深呼吸、听舒缓音乐；遇事戒怒",
            "饮食有节：忌暴饮暴食、忌醇酒辛辣、忌咖啡浓茶；宜少食多餐，晚餐七成饱",
            "解郁食疗：可饮玫瑰花茶、佛手茶、陈皮菊花茶；多食萝卜、金橘、柚子等理气之品",
            "运动疏泄：每日快走或慢跑 30 分钟，或练习瑜伽、八段锦「疏肝式」以疏解肝郁",
        ],
    },

    "阴阳两虚": {
        "diagnosis_summary": "阴损及阳，阴阳俱虚。多为病程迁延日久，由阴虚发展而来。阴虚本在，又见畏寒肢冷、面浮肢肿等阳虚失温之象。病位深及下焦肾元，为病之后期阶段。",
        "treatment_principle": "温阳滋阴，补肾固摄，兼补脾益气",
        "formula": {
            "name": "金匮肾气丸合金匮肾气丸/水陆二仙丹化裁",
            "source": "《金匮要略》肾气丸 + 《洪氏集验方》水陆二仙丹化裁",
            "herbs": [
                ("熟附子", "6-9g（先煎）"),
                ("桂枝", "6-9g"),
                ("熟地黄", "18-24g"),
                ("山萸肉", "12g"),
                ("怀山药", "20-30g"),
                ("茯苓", "15g"),
                ("牡丹皮", "9g"),
                ("泽泻", "12g"),
                ("金樱子", "15g"),
                ("芡实", "15g"),
                ("补骨脂", "12g"),
                ("淫羊藿", "12g"),
                ("黄芪", "20-30g"),
                ("白术", "12-15g"),
            ]
        },
        "addition_subtraction": [
            "畏寒肢冷甚者：加 肉桂粉 2g（冲）、干姜 6g 增温阳之力",
            "下肢浮肿明显：加 车前子 15g（包）、冬瓜皮 30g、川牛膝 12g 利水消肿",
            "夜尿频多：加 桑螵蛸 12g、益智仁 12g、乌药 9g 温肾缩尿",
            "五更泄、便溏：加 肉豆蔻 9g、吴茱萸 3g、五味子 6g 温脾暖肾涩肠",
            "胸闷心悸、唇舌紫暗：加 丹参 15g、桃仁 9g、红花 6g 活血化瘀通络",
        ],
        "lifestyle": [
            "防寒保暖：特别注意腰腹、下肢及足部保暖；睡前温水泡脚 15-20 分钟",
            "饮食温补：宜食羊肉、牛肉、韭菜、生姜、桂圆等温阳之品；严忌生冷瓜果、冰饮凉菜",
            "节制养精：严格节制房事，休养生息，保养元阴元阳",
            "适度运动：以不出汗或微汗的柔和运动为宜，如晒太阳「天灸」、八段锦、静坐养生；避免大汗淋漓耗散阳气",
            "水肿护理：下肢浮肿者宜抬高患肢；每日监测体重及出入量，发现异常及时就医",
        ],
    },
}


# ============================================================
# 辅助：6个典型病例特征（与药方/证型一一对应）
# ============================================================
CASE_STUDIES = {
    "病例A · 气阴两虚型": {
        'description': '男，58岁，病程5年。口渴多饮，倦怠乏力，动则气短，时自汗出，腰膝酸软，视物昏花，舌质淡红偏干，苔薄白少津，脉细弱。',
        'features': [0.70, 0.30, 0.82, 0.22, 0.20, 0.28, 0.32, 0.40, 0.72, 0.65, 0.20, 0.18, 0.28, 0.55, 0.72],
    },
    "病例B · 痰热互结型": {
        'description': '女，46岁，病程4年，BMI 29.1。口苦口黏，脘腹胀满，形体肥胖，心烦急躁，大便黏滞不爽，舌红苔黄腻，脉滑数。',
        'features': [0.62, 0.42, 0.48, 0.72, 0.82, 0.52, 0.72, 0.90, 0.30, 0.28, 0.18, 0.28, 0.92, 0.72, 0.42],
    },
    "病例C · 肝肾阴虚型": {
        'description': '男，66岁，病程11年。腰膝酸软，双目干涩视物模糊，头晕耳鸣，口干不欲多饮，失眠多梦，舌红少苔，脉沉细略数。',
        'features': [0.62, 0.30, 0.52, 0.28, 0.32, 0.38, 0.40, 0.62, 0.92, 0.90, 0.38, 0.28, 0.38, 0.72, 0.82],
    },
    "病例D · 热盛伤津型": {
        'description': '男，51岁，病程2年。大渴引饮，消谷善饥，口苦口臭，小便黄赤量多，大便燥结，舌红苔黄燥，脉洪数有力。',
        'features': [0.96, 0.92, 0.32, 0.52, 0.80, 0.92, 0.92, 0.32, 0.22, 0.28, 0.10, 0.10, 0.62, 0.92, 0.42],
    },
    "病例E · 肝胃郁热型": {
        'description': '女，47岁，病程4年。平素急躁易怒，胸胁胀满，口苦反酸，多食易饥，大便秘结3-4日一行，舌红苔黄，脉弦数。',
        'features': [0.72, 0.82, 0.38, 0.96, 0.92, 0.70, 0.92, 0.42, 0.32, 0.28, 0.12, 0.18, 0.50, 0.92, 0.32],
    },
    "病例F · 阴阳两虚型": {
        'description': '男，72岁，病程16年。畏寒肢冷，足跗浮肿，夜尿频多，腰膝酸软，神疲嗜睡，视物不清，舌淡胖苔白，脉沉细无力。',
        'features': [0.48, 0.22, 0.86, 0.28, 0.18, 0.48, 0.46, 0.58, 0.92, 0.90, 0.96, 0.90, 0.26, 0.38, 0.86],
    },
}


# ============================================================
# 数据加载（静默，无任何标识输出）
# ============================================================
@st.cache_data(show_spinner=False)
def _load_ref_data():
    # 依次尝试：项目目录下的「补充」子目录 → 脚本同级目录（找不到则静默降级）
    _base = os.path.dirname(os.path.abspath(__file__))
    for _path in [os.path.join(_base, '补充'), _base]:
        try:
            _csv = os.path.join(_path, 'dataset_final_6class.csv')
            if not os.path.exists(_csv):
                continue
            _df = pd.read_csv(_csv)
            _res = {}
            for _s in SYNDROME_LABELS:
                _sub = _df[_df['syndrome_guideline'] == _s]
                if len(_sub) > 0:
                    _res[_s] = {c: _sub[c].mean() for c in FEATURE_NAMES}
            return _res
        except Exception:
            continue
    return {}

@st.cache_resource
def get_model():
    return load_model_and_explainer()


# ============================================================
# 诊断证据生成：基于输入特征值 vs 典型特征范围 → 输出有数据依据的证据
# ============================================================
def build_evidence_list(feature_dict, syndrome):
    """生成支持证据与不支持证据"""
    refs = SYNDROME_FEATURE_EVIDENCE.get(syndrome, [])
    support, against = [], []
    for feat, (lo, hi), ref_mean in refs:
        val = feature_dict.get(feat, 0.5)
        if lo <= val <= hi:
            # 支持：值落在典型范围
            if val >= ref_mean:
                level = "高度典型"
            else:
                level = "符合特征"
            support.append((feat, val, ref_mean, level))
        else:
            # 不支持
            if val < lo:
                reason = f"偏弱（{val:.2f} < 典型下限 {lo:.2f}）"
            else:
                reason = f"偏高（{val:.2f} > 典型上限 {hi:.2f}）"
            against.append((feat, val, ref_mean, reason))
    # 按偏离程度排序
    support.sort(key=lambda x: -x[1])
    against.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    return support[:6], against[:6]


def create_prob_chart(prob_dict):
    """证型概率水平条形图（现代风格）"""
    sorted_kv = sorted(prob_dict.items(), key=lambda x: x[1])
    ys = [f"{get_syndrome_icon(k)}  {k}" for k, _ in sorted_kv]
    xs = [v for _, v in sorted_kv]
    cs = [get_syndrome_color(k) for k, _ in sorted_kv]
    fig = go.Figure(go.Bar(
        x=xs, y=ys, orientation='h',
        marker_color=cs,
        marker_line_width=0,
        text=[f"<b>{v*100:.1f}%</b>" for v in xs],
        textposition='outside',
        textfont=dict(size=13, color="#263238", family="Arial Black"),
        hovertemplate='%{y}<br>概率：<b>%{x:.1%}</b><extra></extra>',
        width=0.65,
    ))
    fig.update_layout(
        height=360, showlegend=False,
        xaxis=dict(range=[0, 1.08], tickformat=".0%",
                   gridcolor="rgba(0,0,0,0.05)", zeroline=False,
                   title="<b>预测概率</b>"),
        yaxis=dict(title=None, tickfont=dict(size=12, color="#263238", family="Microsoft YaHei")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=0),
    )
    return fig


def create_shap_fig(shap_values, feature_values, syndrome_name):
    sv = np.asarray(shap_values); fv = np.asarray(feature_values)
    df = pd.DataFrame({'feat': FEATURE_NAMES[:len(sv)], 'sv': sv[:len(FEATURE_NAMES)], 'fv': fv[:len(FEATURE_NAMES)]})
    df = df.sort_values('sv', key=abs, ascending=False).head(10)
    colors = ['#e53935' if v>0 else '#1e88e5' for v in df['sv']]
    fig = go.Figure(go.Bar(
        x=df['sv'], y=df['feat'], orientation='h',
        marker_color=colors,
        text=[f"<b>{v:+.3f}</b>" for v in df['sv']],
        textposition='outside',
        textfont=dict(size=11),
        customdata=df['fv'],
        hovertemplate='<b>%{y}</b><br>SHAP贡献=%{x:+.4f}<br>输入值=%{customdata:.2f}<extra></extra>',
        width=0.65,
    ))
    fig.add_vline(x=0, line_color="#bdbdbd", line_width=1.5, line_dash="dash")
    fig.update_layout(
        height=380, showlegend=False,
        title=dict(text=f"<b>{syndrome_name}</b> · 关键特征贡献（SHAP）",
                   font=dict(size=14, color="#1b5e20"), x=0.02, xanchor="left"),
        xaxis=dict(title="<b>SHAP 值（正=支持该证型 / 负=不支持）</b>", zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(family="Microsoft YaHei")),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=0),
    )
    return fig


def create_shap_heatmap(shap_dict, feature_values):
    """
    构造【特征 × 证型】SHAP 贡献热力图（红蓝色系：深蓝正支持 / 深红负不支持 / 中间浅白）。
    色阶与图例、annotation 判定阈值共享同一配置，保持语义一致。
    """
    labels = [s for s in SYNDROME_LABELS if s in shap_dict]
    if not labels:
        labels = list(SYNDROME_LABELS)
    n_feat = len(FEATURE_NAMES)

    # 构建 z 矩阵（行=特征，列=证型）
    z = np.zeros((n_feat, len(labels)), dtype=float)
    for j, s in enumerate(labels):
        sv = shap_dict.get(s, np.zeros(n_feat))
        sv = np.asarray(sv).flatten()
        if len(sv) >= n_feat:
            z[:, j] = sv[:n_feat]
        else:
            z[:len(sv), j] = sv
    fv = np.asarray(feature_values).flatten()
    if len(fv) < n_feat:
        fv = np.pad(fv, (0, n_feat - len(fv)), mode="constant")

    # 颜色对称范围（使 0 为色带中心）
    vmax = float(np.max(np.abs(z))) if np.size(z) else 1.0
    vmax = max(vmax, 1e-6)

    # ====== 共享配色配置（热力图 + annotation + 降级cmap 三处复用） ======
    # 参考标准 RdBu reversed：深蓝(正) -> 白 -> 深红(负)，共 7 段锚点
    HEATMAP_COLORSCALE = [
        [0.0,  "#8B0000"],  # -vmax  最深红（强不支持）
        [0.2,  "#c92a2a"],
        [0.4,  "#f8b195"],
        [0.5,  "#ffffff"],  # 0      白（中性）
        [0.6,  "#92c5de"],
        [0.8,  "#2166ac"],
        [1.0,  "#053061"],  # +vmax  最深蓝（强支持）
    ]
    # annotation 显示阈值 & 字体颜色判定（与 colorscale 绑定）
    ANNOT_SHOW_FRAC = 0.22   # |v|/vmax >= 22% 时显示数值
    WHITE_FONT_FRAC = 0.55   # |v|/vmax >= 55% 时深色底用白字
    POS_THEME_DARK = "#0d47a1"   # 深蓝主题色，用于坐标轴标题
    NEG_THEME_DARK = "#8B0000"   # 深红主题色，用于图例语义

    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=FEATURE_NAMES,
        zmin=-vmax, zmax=vmax,
        zmid=0,
        colorscale=HEATMAP_COLORSCALE,
        colorbar=dict(
            title=dict(
                text=(f"<b style='color:{POS_THEME_DARK}'>⬆ 正 = 支持该证型</b><br>"
                      f"<b style='color:{NEG_THEME_DARK}'>⬇ 负 = 不支持该证型</b><br>"
                      f"<span style='font-size:10px;color:#546e7a'>SHAP 贡献值</span>"),
                side="right",
                font=dict(size=10, family="Microsoft YaHei"),
            ),
            thickness=16,
            lenmode="pixels", len=320,
            outlinewidth=0,
            tickfont=dict(size=10, color="#455a64"),
            xanchor="left", x=1.02,
            ticksuffix="  ",
        ),
        hoverongaps=False,
        hovertemplate=(
            "<b>特征：</b>%{y}<br>"
            "<b>证型：</b>%{x}<br>"
            "<b>SHAP 贡献：</b>%{z:+.4f}<br>"
            "<b>当前输入值：</b>%{customdata:.2f}<extra></extra>"
        ),
        customdata=np.stack([fv for _ in range(len(labels))], axis=1),
        xgap=2, ygap=2,
        showscale=True,
    ))

    # 单元格数值标注（与共享阈值绑定）
    annotations = []
    for i in range(n_feat):
        for j in range(len(labels)):
            v = z[i, j]
            if abs(v) >= vmax * ANNOT_SHOW_FRAC:
                # 正深蓝 / 负深红 达到一定浓度就用白字，否则深字
                if abs(v) >= vmax * WHITE_FONT_FRAC:
                    col = "#ffffff"
                else:
                    col = "#1a1a1a"
                annotations.append(dict(
                    x=labels[j], y=FEATURE_NAMES[i],
                    text=f"{v:+.2f}",
                    showarrow=False,
                    font=dict(size=10, color=col, family="Microsoft YaHei"),
                    xref="x1", yref="y1",
                ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=120, t=40, b=10),
        xaxis=dict(
            title=dict(text="<b>六类中医证型</b>", font=dict(size=12, color=POS_THEME_DARK)),
            tickfont=dict(family="Microsoft YaHei", size=12, color="#37474f"),
            side="bottom",
            gridcolor="rgba(13,71,161,0.05)",
            linecolor="rgba(13,71,161,0.15)",
        ),
        yaxis=dict(
            title=dict(text="<b>临床特征（15 项）</b>", font=dict(size=12, color=POS_THEME_DARK)),
            tickfont=dict(family="Microsoft YaHei", size=11, color="#37474f"),
            autorange="reversed",
            gridcolor="rgba(13,71,161,0.05)",
            linecolor="rgba(13,71,161,0.15)",
        ),
        plot_bgcolor="#fbfcff",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=annotations,
        font=dict(family="Microsoft YaHei"),
    )
    # 降级时用的 pandas Styler 配色（与上方 colorscale 语义同构）
    fig._share_pandas_cmap = "RdBu"
    return fig


# ============================================================
# 渲染：品牌标题
# ============================================================
def render_header():
    c1, c2, c3 = st.columns([1, 5, 1])
    with c2:
        st.markdown('<h1 class="brand-title">症智明辨</h1>', unsafe_allow_html=True)
        st.markdown('<p class="brand-subtitle">🌿 中医智能辅助辨证系统 · 基于 15 项临床特征 · 六类中医证型 · 可解释性决策引擎 · 个性化方药建议</p>', unsafe_allow_html=True)


# ============================================================
# 渲染：诊断主流程
# ============================================================
def render_diagnosis():
    # 加载模型
    with st.spinner('正在初始化诊断引擎...'):
        model, explainer = get_model()

    # 侧边栏：典型病例
    st.sidebar.markdown('<div class="sidebar-brand">📚 典型病例库</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-brand-sub">精选 6 类证型典型病例<br>一键加载完整特征</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="height:1px;background:rgba(255,255,255,0.2);margin:0.6rem 0;"></div>', unsafe_allow_html=True)

    if 'feature_values' not in st.session_state:
        st.session_state.feature_values = {f: 0.5 for f in FEATURE_NAMES}
    if 'selected_case' not in st.session_state:
        st.session_state.selected_case = "手动输入特征"

    case_opt = st.sidebar.selectbox(
        "加载典型病例：",
        ["手动输入特征"] + list(CASE_STUDIES.keys()),
        help="选择后将自动填充对应证型的特征值",
    )
    if case_opt != "手动输入特征" and case_opt != st.session_state.selected_case:
        st.session_state.selected_case = case_opt
        case = CASE_STUDIES[case_opt]
        for i, fn in enumerate(FEATURE_NAMES):
            st.session_state.feature_values[fn] = case['features'][i]
        st.sidebar.success(f"✅ 已加载：{case_opt}")
        st.sidebar.info(case['description'])

    # ========== 输入区域 ==========
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📝 临床特征采集（15 项）</div>', unsafe_allow_html=True)
        st.caption("请根据患者实际情况，对以下 15 项症状/体征的程度进行评分（0 = 无 / 1 = 极重）。每项均附有中医诊断学释义。")
        st.markdown("")

        feature_inputs = {}
        col_clinical, col_tongue = st.columns([3, 2])

        with col_clinical:
            st.markdown("#### 🩺 临床症状（12 项）")
            for fn in FEATURE_GROUPS['临床症状']:
                feature_inputs[fn] = st.slider(
                    fn, min_value=0.0, max_value=1.0, step=0.05,
                    value=float(st.session_state.feature_values[fn]),
                    help=FEATURE_DESCRIPTIONS.get(fn, ''),
                    key=f"sld_{fn}",
                )

        with col_tongue:
            st.markdown("#### 👅 舌象与脉象（3 项）")
            for fn in FEATURE_GROUPS['舌脉特征']:
                feature_inputs[fn] = st.slider(
                    fn, min_value=0.0, max_value=1.0, step=0.05,
                    value=float(st.session_state.feature_values[fn]),
                    help=FEATURE_DESCRIPTIONS.get(fn, ''),
                    key=f"sld_{fn}",
                )
            st.markdown("")
            st.markdown("""
            <div style="padding:14px 16px;background:rgba(27,94,32,0.06);border-radius:12px;border-left:4px solid #43a047;">
            <b style="color:#1b5e20;">💡 操作提示</b><br>
            <span style="font-size:0.85rem;color:#37474f;line-height:1.6;">
            可先从左侧「典型病例库」一键加载证型特征，再点击下方「🔍 开始辨证分析」查看完整诊断报告。
            </span></div>
            """, unsafe_allow_html=True)

        for fn in FEATURE_NAMES:
            st.session_state.feature_values[fn] = feature_inputs[fn]

        st.markdown("")
        _, c_btn, _ = st.columns([1, 3, 1])
        with c_btn:
            predict_clicked = st.button("🔍 开始辨证分析 · 生成诊断报告")
            if predict_clicked:
                st.session_state["diagnosis_done"] = True

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== 诊断结果 ==========
    if not st.session_state.get("diagnosis_done", False):
        return

    fv = np.array([feature_inputs[f] for f in FEATURE_NAMES])
    result = predict_syndrome(model, fv)
    shap_dict = get_shap_values(explainer, fv)
    pred_s = result['predicted_syndrome']
    probs = result['probabilities']
    conf = result['confidence']
    risk_level, risk_desc = calculate_risk_score(probs)
    info = SYNDROME_INFO.get(pred_s, {})
    treatment = SYNDROME_TREATMENT.get(pred_s, {})

    # ----- 结果 Hero -----
    icon = get_syndrome_icon(pred_s)
    st.markdown(f"""
    <div class="diagnosis-hero">
        <div class="hero-label">◆ 辨证诊断结论</div>
        <div class="hero-value">{icon} {pred_s}证</div>
        <div class="hero-desc">{info.get('description', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ----- 4 KPI -----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🎯</div>
        <div class="kpi-value">{conf*100:.1f}%</div><div class="kpi-label">主证匹配度</div></div>""", unsafe_allow_html=True)
    with c2:
        order = sorted(probs.values(), reverse=True)
        gap = (order[0] - order[1]) if len(order)>=2 else 0
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">📏</div>
        <div class="kpi-value">{gap*100:.1f}%</div><div class="kpi-label">与次证差距</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">⚖️</div>
        <div class="kpi-value">{risk_level}</div><div class="kpi-label">诊断把握度评级</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{sum(v>=0.15 for v in probs.values())} 类</div><div class="kpi-label">可疑相关证型</div></div>""", unsafe_allow_html=True)

    # ----- 概率图 + 辨证证据 -----
    col_prob, col_ev = st.columns([3, 4])
    with col_prob:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 六类证型概率分布</div>', unsafe_allow_html=True)
            st.plotly_chart(create_prob_chart(probs), width='stretch')
            # 胶囊条
            st.markdown("")
            for s in SYNDROME_LABELS:
                p = probs[s]
                c = get_syndrome_color(s)
                st.markdown(f"""
                <div class="prob-bar">
                  <div class="prob-bar-name">{get_syndrome_icon(s)} {s}</div>
                  <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{p*100:.1f}%;background:linear-gradient(90deg,{c}dd,{c}88);"></div>
                  </div>
                  <div class="prob-bar-value">{p*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_ev:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🧾 诊断证据分析（{pred_s}）</div>', unsafe_allow_html=True)
            st.caption("基于典型特征范围与当前输入值对比，给出「支持该证型的证据」与「不支持该证型的证据」")
            support, against = build_evidence_list(feature_inputs, pred_s)
            st.markdown(f"<b style='color:#1b5e20;'>✅ 支持该证型的特征（{len(support)} 项）</b>", unsafe_allow_html=True)
            for f, v, ref, lvl in support:
                st.markdown(f'''
                <div class="evidence-item evidence-support">
                  <span class="evidence-icon">✓</span>
                  <span><b>{f}</b> = {v:.2f}（参考典型均值 {ref:.2f}）· <b>{lvl}</b></span>
                </div>''', unsafe_allow_html=True)
            st.markdown("")
            if against:
                st.markdown(f"<b style='color:#b71c1c;'>⚠️ 不完全支持的特征（{len(against)} 项）</b>", unsafe_allow_html=True)
                for f, v, ref, reason in against:
                    st.markdown(f'''
                    <div class="evidence-item evidence-against">
                      <span class="evidence-icon">!</span>
                      <span><b>{f}</b> = {v:.2f}（参考典型均值 {ref:.2f}）· {reason}</span>
                    </div>''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ----- SHAP 多证型热力图（指定位置：概率+证据之后，核心病机之前） -----
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔥 各证型 SHAP 值分布 · 特征-证型关联热力图</div>', unsafe_allow_html=True)
        st.caption("热力图展示当前患者各特征对 <b>六类证型预测</b> 的 SHAP 贡献方向与强度。<span style='color:#053061'>■ 深蓝</span>=强烈支持该证型（正向贡献），<span style='color:#8B0000'>■ 深红</span>=强烈不支持该证型（负向贡献），<span style='color:#eceff1'>■ 浅白</span>=中性接近 0。悬停单元格可查看精确数值与当前输入特征值。")
        try:
            fig_hm = create_shap_heatmap(shap_dict, fv)
            st.plotly_chart(fig_hm, width='stretch')
        except Exception as _e:
            st.warning("SHAP 热力图渲染失败：%s" % str(_e))
            # 输出结构化预览表格作为降级
            try:
                labels = [s for s in SYNDROME_LABELS if s in shap_dict]
                if not labels:
                    labels = list(SYNDROME_LABELS)
                rows = {}
                for s in labels:
                    arr = np.asarray(shap_dict.get(s, np.zeros(len(FEATURE_NAMES)))).flatten()
                    if len(arr) < len(FEATURE_NAMES):
                        arr = np.pad(arr, (0, len(FEATURE_NAMES)-len(arr)))
                    rows[s] = arr[:len(FEATURE_NAMES)]
                df_hm = pd.DataFrame(rows, index=FEATURE_NAMES).round(4)
                vmax_ab = float(df_hm.abs().max().max())
                vmax_ab = max(vmax_ab, 1e-6)
                st.dataframe(df_hm.style.background_gradient(cmap="RdBu", axis=None, vmin=-vmax_ab, vmax=vmax_ab), width='stretch', height=520)
            except Exception:
                pass
        st.markdown('</div>', unsafe_allow_html=True)

    # ----- 核心病机 + SHAP -----
    col_shap, col_patho = st.columns([3, 2])
    with col_shap:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            try:
                ps = shap_dict[pred_s]
                if hasattr(ps, 'shape') and len(ps.shape) > 1:
                    ps = ps.flatten()
            except Exception:
                ps = np.zeros(len(FEATURE_NAMES))
            st.plotly_chart(create_shap_fig(ps, fv, pred_s), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

    with col_patho:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📘 核心病机与辨证要点</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="line-height:1.8;color:#37474f;font-size:0.95rem;">
              <b style="color:#e65100;">▸ 核心病机：</b>{treatment.get('diagnosis_summary', '')}
            </div>
            <div style="height:1px;background:rgba(0,0,0,0.08);margin:14px 0;"></div>
            <div style="line-height:1.8;color:#37474f;font-size:0.95rem;">
              <b style="color:#e65100;">▸ 治疗立法：</b>{treatment.get('treatment_principle', '')}
            </div>
            <div style="height:1px;background:rgba(0,0,0,0.08);margin:14px 0;"></div>
            <div style="line-height:1.7;color:#37474f;font-size:0.92rem;">
              <b style="color:#e65100;">▸ 辨证关键词：</b>
              {' · '.join(info.get('keywords', []))}
            </div>
            <div style="height:1px;background:rgba(0,0,0,0.08);margin:14px 0;"></div>
            <div style="line-height:1.7;color:#546e7a;font-size:0.88rem;">
              <b>📌 诊断把握说明：</b>{risk_desc}。
              建议结合舌诊、脉诊、问诊四诊合参，综合判断后确立最终治则方药。
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================
    # AI 智能解读 + 多轮聊天（模块1 + 模块2）
    # ========================================================================
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title" style="font-size:1.3rem;">🧠 AI 智能解读 &amp; 多轮对话</div>',
        unsafe_allow_html=True,
    )

    llm_cfg = get_llm_config()
    if llm_cfg.get("configured"):
        st.caption("本模块仅解读 ML 模型已输出的证型、概率与 SHAP 贡献，不会独立重新诊断。"
                   "当前已接入大模型（"
                   f"{llm_cfg.get('provider','?')} / {llm_cfg.get('model','?')}"
                   "），可自由追问。")
    else:
        st.caption("本模块仅解读 ML 模型已输出的证型、概率与 SHAP 贡献，不会独立重新诊断。"
                   "⚠️ 当前未配置大模型，使用规则化降级版；自由追问能力有限，"
                   "可通过环境变量 LLM_API_KEY / LLM_PROVIDER / LLM_BASE_URL / LLM_MODEL 启用完整对话。")

    # --- 会话状态：按「当前辨证结果」绑定，辨证变了自动清空 ---
    _session_key = (
        f"chat_session__{pred_s}__"
        f"{hash(tuple(sorted((probs or {}).items())))}"
    )
    if st.session_state.get("_last_diag_key") != _session_key:
        st.session_state["chat_messages"] = []
        st.session_state["chat_system_prompt"] = None
        st.session_state["_first_interp_meta"] = None
        st.session_state["_last_diag_key"] = _session_key

    # --- ① 一键解读按钮 + 清空 + 徽章 ---
    c1, c2, c3 = st.columns([3, 1, 1])
    _btn_disabled = (probs is None or shap_dict is None)
    with c1:
        if st.button(
            "🔍 大模型智能解读本次辨证结果",
            disabled=_btn_disabled,
            width='stretch',
            key="btn_first_interp",
        ):
            try:
                with st.spinner("AI 正在解读本次辨证结果，请稍候…"):
                    symptoms = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[:12]}
                    tongue   = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[12:]}

                    st.session_state["chat_system_prompt"] = build_chat_system_prompt(
                        symptoms=symptoms, tongue_pulse=tongue,
                        probabilities=probs, shap_dict=shap_dict,
                        feature_names=FEATURE_NAMES,
                        syndrome_labels=SYNDROME_LABELS,
                    )
                    st.session_state["chat_messages"] = []

                    ai_result = generate_llm_interpretation(
                        symptoms=symptoms, tongue_pulse=tongue,
                        probabilities=probs, shap_dict=shap_dict,
                        feature_names=FEATURE_NAMES,
                        syndrome_labels=SYNDROME_LABELS,
                        use_llm=True,
                    )
                    st.session_state["chat_messages"].append({
                        "role": "assistant", "content": ai_result["text"],
                    })
                    st.session_state["_first_interp_meta"] = {
                        "source": ai_result["source"],
                        "error": ai_result.get("error"),
                    }
            except Exception as e:
                st.error(f"一键解读失败：{e}")
    with c2:
        if st.button("🗑 清空当前会话", width='stretch', key="btn_clear_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()
    with c3:
        _badge_color = "#1b5e20" if llm_cfg.get("configured") else "#8d6e63"
        _badge_text  = ("🛰 已接入大模型" if llm_cfg.get("configured")
                        else "📝 规则化降级")
        st.markdown(
            f'<span style="display:inline-block;background:{_badge_color}1a;'
            f'color:{_badge_color};border:1px solid {_badge_color}40;'
            f'padding:6px 12px;border-radius:999px;font-size:0.82rem;'
            f'font-weight:600;width:100%;text-align:center;margin-top:2px;">{_badge_text}</span>',
            unsafe_allow_html=True,
        )

    # --- 会话历史渲染 ---
    history = st.session_state.get("chat_messages", [])
    if not history:
        st.info("👆 点击上方「🔍 大模型智能解读本次辨证结果」按钮，即可自动生成第一轮解读回答；之后可继续自由追问。")
    else:
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        _meta = st.session_state.get("_first_interp_meta")
        if _meta and _meta.get("error"):
            st.warning(f"⚠️ 大模型调用已降级：{_meta['error'][:120]}")

    # --- ② 聊天输入框（多轮追问） ---
    if probs is not None:
        prompt = st.chat_input(
            "💬 继续追问：例如「气阴两虚和阴虚热盛怎么鉴别？」「饮食还有哪些禁忌？」",
            key="chat_input",
        )
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.get("chat_system_prompt") is None:
                symptoms = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[:12]}
                tongue   = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[12:]}
                st.session_state["chat_system_prompt"] = build_chat_system_prompt(
                    symptoms=symptoms, tongue_pulse=tongue,
                    probabilities=probs, shap_dict=shap_dict,
                    feature_names=FEATURE_NAMES,
                    syndrome_labels=SYNDROME_LABELS,
                )

            with st.chat_message("assistant"):
                with st.spinner("思考中…"):
                    try:
                        if llm_cfg.get("configured"):
                            reply = chat_with_patient_data(
                                user_question=prompt,
                                history=st.session_state["chat_messages"][:-1],
                                system_prompt=st.session_state["chat_system_prompt"],
                            )
                        else:
                            reply = fallback_chat_reply(prompt)
                    except Exception as e:
                        # API 失败：友好提示 + 规则化降级，ML+SHAP 不受影响
                        reply = (
                            f"⚠️ {str(e)}\n\n"
                            + fallback_chat_reply(prompt)
                        )

                st.markdown(reply)
                st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
    else:
        st.chat_input("请先完成辨证分析（点击页面底部的「🔍 开始辨证分析」按钮）", disabled=True)

    # ----- 处方建议 -----
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title" style="font-size:1.3rem;">💊 个性化处方建议（{pred_s}证）</div>', unsafe_allow_html=True)
    st.caption("以下内容基于「辨证论治」原则，针对当前证型给出经典方剂与常用药物；实际处方须由执业中医师四诊合参后化裁使用。")

    formula_info = treatment.get("formula", {})
    c_formula, c_adjust = st.columns([3, 2])
    with c_formula:
        with st.container():
            st.markdown('<div class="formula-card">', unsafe_allow_html=True)
            st.markdown(f"""<div class="formula-title">推荐主方：{formula_info.get('name', '')}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:0.85rem;color:#8d6e63;margin-bottom:14px;">🕮 {formula_info.get('source', '')}</div>""", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.9rem;color:#5d4037;font-weight:600;margin-bottom:8px;">◆ 药物组成及参考剂量：</div>', unsafe_allow_html=True)
            # ===== 渲染 herb-grid：3列栅格 + 君臣佐使视觉分组分隔线（不改任何药味信息） =====
            herbs = list(formula_info.get('herbs', []))
            total = len(herbs)
            n_zhujun = max(1, round(total * 0.25)) if total >= 6 else min(2, total)   # 君 25%
            n_zuochen = max(n_zhujun + 1, round(total * 0.60)) if total >= 10 else total
            parts_html = []
            for idx, (h, d) in enumerate(herbs):
                parts_html.append(
                    f'<div class="herb-tag"><span class="herb-tag-name">{h}</span>'
                    f'<span class="herb-dose">{d}</span></div>'
                )
                # 视觉分组线：君->臣 之后插入
                if idx + 1 == n_zhujun and n_zhujun < total:
                    parts_html.append('<div class="herb-group-divider" aria-hidden="true"></div>')
                # 视觉分组线：臣->佐使 之后插入
                elif idx + 1 == n_zuochen and n_zuochen < total:
                    parts_html.append('<div class="herb-group-divider" aria-hidden="true"></div>')
            st.markdown(f'<div class="herb-grid">{"".join(parts_html)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with c_adjust:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⚖️ 随症加减</div>', unsafe_allow_html=True)
            for item in treatment.get('addition_subtraction', []):
                st.markdown(f'<div class="care-item"><span class="care-icon">▸</span><span>{item}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ----- 生活调摄 -----
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏡 生活调摄与饮食宜忌</div>', unsafe_allow_html=True)
        lifes = treatment.get('lifestyle', [])
        cols = st.columns(min(len(lifes), 4)) if lifes else []
        for i, life in enumerate(lifes):
            with cols[i % len(cols)]:
                icons = ['🍽️', '🏃', '🧘', '😴']
                top_colors = ['#ffb74d', '#81c784', '#64b5f6', '#ba68c8']
                title_colors = ['#e65100', '#2e7d32', '#1565c0', '#6a1b9a']
                icon_bgs = ['linear-gradient(135deg,#fff5e6 0%,#ffe0b2 100%)',
                            'linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%)',
                            'linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%)',
                            'linear-gradient(135deg,#f3e5f5 0%,#e1bee7 100%)']
                parts = life.split("：", 1)
                title, desc = (parts[0], parts[1]) if len(parts) == 2 else ("调摄建议", life)
                st.markdown(f"""
                <div style="background:linear-gradient(180deg,#ffffff 0%,#fefefe 100%);
                            padding:18px 18px 16px 18px;
                            border-radius:16px;
                            border-top:4px solid {top_colors[i%4]};
                            height:100%;
                            margin:6px 8px 14px 8px;
                            box-shadow:0 3px 10px rgba(16,42,18,0.05), 0 1px 2px rgba(16,42,18,0.03);
                            transition: transform .2s ease, box-shadow .2s ease;"
                     onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 18px rgba(16,42,18,0.08), 0 2px 4px rgba(16,42,18,0.04)';"
                     onmouseout="this.style.transform='';this.style.boxShadow='';">
                  <div style="font-size:1.75rem;margin-bottom:6px;
                              width:40px;height:40px;display:flex;align-items:center;justify-content:center;
                              border-radius:12px;background:{icon_bgs[i%4]};
                              box-shadow:inset 0 1px 0 rgba(255,255,255,0.9);">{icons[i%4]}</div>
                  <div style="font-weight:750;color:{title_colors[i%4]};margin-bottom:6px;font-size:0.98rem;
                              letter-spacing:0.2px;">{title}</div>
                  <div style="font-size:0.86rem;color:#5d4037;line-height:1.8;word-break:break-word;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown("""
        <div style="padding:16px 20px 16px 20px;
                    background:linear-gradient(135deg,#fff5f5 0%,#ffe9e9 100%);
                    border-radius:16px;
                    border-left:4px solid #e53935;
                    border-top:1px solid rgba(229,57,53,0.15);
                    box-shadow:0 2px 10px rgba(183,28,28,0.06), inset 0 1px 0 rgba(255,255,255,0.9);
                    font-size:0.88rem;color:#8a1a1a;line-height:1.85;">
            <div style="display:flex;align-items:flex-start;gap:12px;">
              <div style="flex-shrink:0;width:34px;height:34px;border-radius:10px;
                          background:rgba(229,57,53,0.10);
                          display:flex;align-items:center;justify-content:center;
                          font-size:1.1rem;">⚕️</div>
              <div><b style="font-size:0.92rem;color:#b71c1c;">免责声明：</b>本系统为中医辅助辨证科研教学工具，所提供的证型分析、概率分布、SHAP 可解释性分析、方剂建议与调摄指导均<b>仅供科研教学参考</b>，不构成医疗处方、诊断意见或治疗方案。本系统中的证型预测由机器学习模型（随机森林）输出，大模型仅做科普解读，不能替代执业中医师的四诊合参与辨证论治。具体诊断与用药请务必由<b>执业中医师</b>面诊后确定，切勿自行抓药服用或延误就医。</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 主函数
# ============================================================
def main():
    # 静默加载参考数据（不输出任何提示）
    _load_ref_data()

    # 标题
    render_header()

    # 侧边栏：版权/底部信息（无数据集提及）
    st.sidebar.markdown('<div style="height:1px;background:rgba(255,255,255,0.18);margin:1.2rem 0;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="color:rgba(255,255,255,0.8);text-align:center;font-size:0.78rem;line-height:1.7;">
    🌿 中医智能辨证系统<br>
    辅助辨证 · 辨证论治 · 对症下药<br>
    <span style="opacity:0.7;">仅供临床参考与学习使用</span>
    </div>""", unsafe_allow_html=True)

    # 主功能区（单页应用，无任何数据集信息页面）
    render_diagnosis()


if __name__ == "__main__":
    main()
