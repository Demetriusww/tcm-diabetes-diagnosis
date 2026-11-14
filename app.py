"""
中医智能辅助辨证系统 — 专业诊断版 V3.0
功能：6类证型智能辨证 + 经典方剂知识溯源
设计：现代渐变配色 / 玻璃拟态卡片 / 信息层级清晰
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import os
import re
import html
from datetime import datetime
import streamlit.components.v1 as components
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
    FEATURE_ATTRIBUTES,
    get_syndrome_color,
    get_syndrome_icon,
    create_feature_tooltip,
    calculate_confidence_level,
)

# 首次进入页面时所有特征均按“无症状”处理，避免 0.5 全满的假数据观感
DEFAULT_FEATURE_VALUE = 0.0
DEFAULT_FEATURE_LEVEL = 1
FEATURE_LEVEL_LABELS = {
    1: "无",
    2: "轻",
    3: "中",
    4: "重",
    5: "极重",
}
FEATURE_STATE_VERSION = "2026-11-five-level-v2"
# 顶部“一键加载预设演示病例”默认使用的病例
DEMO_CASE = "病例A · 气阴两虚型"

# 左侧模块导航定义：输入区 + 五个结果模块
MODULE_NAV_ITEMS = [
    ("input", "00", "🧾", "临床特征采集"),
    ("conclusion", "01", "📊", "辨证结论"),
    ("attribution", "02", "🔎", "归因依据"),
    ("formula", "03", "📜", "经典方剂溯源与方义推演"),
    ("care", "04", "🌿", "辨证施护与起居调摄推演"),
    ("ai", "05", "🧠", "AI 解读与报告"),
]
VALID_MODULES = {item[0] for item in MODULE_NAV_ITEMS}

# 图表统一使用滚动安全配置；概率图额外使用静态模式，避免滚轮和拖动误触缩放。
PLOTLY_SCROLL_SAFE_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "displaylogo": False,
    "doubleClick": False,
}
PLOTLY_STATIC_CONFIG = {
    **PLOTLY_SCROLL_SAFE_CONFIG,
    "staticPlot": True,
}

# 这些属于内部实现用语，不应出现在教师和学生看到的解读正文中。
AI_DEVELOPMENT_TERMS = (
    "四段式标准回答：",
    "四段式标准回答",
    "四段式教学内容：",
    "四段式教学内容",
    "四段式辨证解读：",
    "四段式辨证解读",
    "四段式：",
    "四段式",
    "本地 Mock：",
    "本地 Mock",
    "本地Mock：",
    "本地Mock",
)

# ============================================================
# 页面配置 + 主题配置
# ============================================================
st.set_page_config(
    page_title="中医智能辨证系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto"
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
  --n-300: #c6cec9; --n-500: #566159; --n-700: #39443c; --n-900: #1a231d;
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

/* ========== 全局背景（青瓷竹青 × 宣纸米白 × 淡赭纹理） ========== */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #edf0e5;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {
    background-color: #f1f1e7 !important;
    background-image:
      repeating-linear-gradient(
        0deg,
        rgba(103,82,48,0.022) 0,
        rgba(103,82,48,0.022) 1px,
        transparent 1px,
        transparent 6px
      ),
      repeating-linear-gradient(
        90deg,
        rgba(46,125,50,0.026) 0,
        rgba(46,125,50,0.026) 1px,
        transparent 1px,
        transparent 88px
      ),
      linear-gradient(
        132deg,
        #e8f0e5 0%,
        #f2f2e7 46%,
        #f6eddf 100%
      ) !important;
    border-left: 1px solid rgba(46,125,50,0.07);
    box-shadow:
      inset 22px 0 38px -38px rgba(27,94,32,0.24),
      inset 0 1px 0 rgba(255,255,255,0.55);
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}
[data-testid="stHeader"] {
    background: rgba(239,242,231,0.86) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
/* 隐藏 Streamlit 默认菜单、部署按钮和 GitHub / Fork 入口 */
#MainMenu,
[data-testid="stMainMenu"],
[data-testid="stToolbarActions"],
[data-testid="stDeployButton"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
.stAppDeployButton {
    display: none !important;
    visibility: hidden !important;
}
/* 保留侧边栏展开入口：收起后可通过顶部“展开导航”重新打开 */
[data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    align-items: center;
    gap: 6px;
    min-width: 40px !important;
    min-height: 38px !important;
    padding: 6px 12px !important;
    border: 1px solid rgba(27,94,32,0.24) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.92) !important;
    color: #1b5e20 !important;
    box-shadow: 0 3px 12px rgba(27,94,32,0.10) !important;
}
[data-testid="stExpandSidebarButton"]:hover {
    background: #e8f4e9 !important;
    border-color: rgba(27,94,32,0.42) !important;
}
[data-testid="stExpandSidebarButton"]::after {
    content: "展开导航";
    font-family: inherit;
    font-size: 0.8rem;
    font-weight: 750;
    letter-spacing: 0;
    color: #1b5e20;
    white-space: nowrap;
}
[data-testid="stAppViewContainer"] > .main > [data-testid="block-container"],
[data-testid="stMain"] [data-testid="block-container"] {
    position: relative; z-index: 1;
    max-width: 1320px;
    background: transparent !important;
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

/* ========== 首次使用引导 ========== */
.usage-guide {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    max-width: 920px;
    margin: -0.35rem auto 1.35rem auto;
    padding: 12px 18px;
    border-radius: 12px;
    border: 1px solid rgba(255,152,0,0.24);
    border-left: 4px solid var(--a-500);
    background: linear-gradient(90deg, rgba(255,248,225,0.96), rgba(255,255,255,0.78));
    color: #5d4037;
    font-size: 0.9rem;
    font-weight: 550;
    line-height: 1.75;
    box-shadow: 0 4px 14px rgba(93,64,55,0.05);
}
.usage-guide-badge {
    flex-shrink: 0;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--a-500);
    color: #ffffff;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.6px;
}
.privacy-note {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin: 8px 0 14px 0;
    padding: 7px 12px;
    border-radius: 8px;
    background: rgba(232,245,233,0.72);
    border: 1px solid rgba(46,125,50,0.13);
    color: #2e5d34 !important;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ========== 典型病例详情 ========== */
.clinical-case-card {
    position: relative;
    margin: 2px 0 18px 0;
    padding: 18px 20px 16px 20px;
    border-radius: 14px;
    border: 1px solid rgba(46,125,50,0.15);
    background:
      linear-gradient(115deg, rgba(248,252,246,0.98), rgba(255,250,239,0.96));
    box-shadow: 0 5px 18px rgba(27,94,32,0.055);
    overflow: hidden;
}
.clinical-case-card::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: linear-gradient(180deg, var(--p-600), var(--a-400));
}
.clinical-case-kicker {
    color: var(--p-700);
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1.2px;
    margin-bottom: 5px;
}
.clinical-case-title {
    color: var(--p-900);
    font-size: 1.08rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.clinical-case-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-bottom: 10px;
}
.clinical-case-meta span {
    display: inline-flex;
    padding: 3px 9px;
    border-radius: 999px;
    color: #4e5f50 !important;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(46,125,50,0.10);
    font-size: 0.75rem;
    font-weight: 650;
}
.clinical-case-description {
    color: #39443c;
    font-size: 0.9rem;
    line-height: 1.8;
}
.clinical-case-footnote {
    margin-top: 10px;
    padding-top: 9px;
    border-top: 1px dashed rgba(46,125,50,0.16);
    color: #5f6d62;
    font-size: 0.78rem;
}

/* ========== 低置信度提示 ========== */
.low-confidence-alert {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 0 0 16px 0;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid rgba(198,40,40,0.24);
    border-left: 4px solid #c62828;
    background: linear-gradient(90deg, rgba(255,235,238,0.96), rgba(255,250,250,0.92));
    color: #b71c1c;
    font-size: 0.9rem;
    font-weight: 700;
    line-height: 1.7;
    box-shadow: 0 3px 10px rgba(183,28,28,0.045);
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

/* Streamlit 原生边框容器统一为轻量模块面板 */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.92) 0%,
        rgba(251,250,245,0.84) 100%
    );
    border-color: rgba(46,125,50,0.11) !important;
    border-radius: 12px;
    box-shadow:
        0 7px 22px rgba(31,65,38,0.035),
        inset 0 1px 0 rgba(255,255,255,0.92);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* ========== 结果摘要条 ========== */
.result-summary {
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(360px, 1fr);
    gap: 28px;
    align-items: center;
    padding: 24px 28px;
    margin: 4px 0 18px 0;
    border-radius: 16px;
    border: 1px solid rgba(27,94,32,0.12);
    background: linear-gradient(120deg, rgba(255,255,255,0.96), rgba(240,248,242,0.94));
    box-shadow: var(--sh-1);
}
.result-summary-kicker {
    color: var(--p-700);
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 1.4px;
    margin-bottom: 6px;
}
.result-summary-title {
    color: var(--p-900);
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.35;
    margin-bottom: 5px;
}
.result-summary-desc {
    color: var(--n-500);
    font-size: 0.9rem;
    line-height: 1.75;
    max-width: 720px;
}
.result-summary-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
}
.result-summary-metric {
    min-height: 74px;
    padding: 12px 10px;
    text-align: center;
    border-radius: 10px;
    background: #ffffff;
    border: 1px solid rgba(27,94,32,0.09);
}
.result-summary-metric strong {
    display: block;
    color: var(--p-800);
    font-size: 1.25rem;
    line-height: 1.3;
    white-space: nowrap;
}
.result-summary-metric span {
    display: block;
    margin-top: 4px;
    color: var(--n-500);
    font-size: 0.74rem;
    line-height: 1.35;
}

/* ========== 结果模块分区标题 ========== */
.result-module-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 18px;
    margin: 22px 0 14px 0;
    border-left: 4px solid var(--p-600);
    border-radius: 0 12px 12px 0;
    background: linear-gradient(90deg, rgba(232,245,233,0.92), rgba(255,255,255,0.72));
}
.result-module-index {
    flex: 0 0 42px;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: var(--p-700);
    color: #ffffff;
    font-size: 0.88rem;
    font-weight: 800;
}
.result-module-title {
    color: var(--p-900);
    font-size: 1.12rem;
    font-weight: 800;
    line-height: 1.4;
}
.result-module-subtitle {
    margin-top: 3px;
    color: var(--n-700);
    font-size: 0.82rem;
    line-height: 1.55;
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
    color: var(--n-700);
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
.formula-rationale {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(255, 183, 77, 0.30);
    border-left: 4px solid var(--a-700);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 4px 0 18px 0;
    color: #4e342e;
    font-size: 0.9rem;
    line-height: 1.85;
}
.formula-rationale-title {
    color: var(--a-700);
    font-weight: 750;
    margin-bottom: 6px;
}
.herb-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px 14px;
    margin-top: 6px;
    padding: 4px 2px;
    overflow: visible;
}
.herb-tag {
    position: relative;
    display: block;
    background: linear-gradient(180deg, #ffffff 0%, #fff8ec 100%);
    padding: 0;
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
    cursor: pointer;
    z-index: 1;
}
.herb-tag:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(191,54,12,0.09), inset 0 1px 0 rgba(255,255,255,0.9);
    border-left-color: var(--a-600);
    z-index: 30;
}
.herb-tag[open] {
    z-index: 40;
    border-left-color: var(--a-700);
    box-shadow: 0 8px 20px rgba(191,54,12,0.12), inset 0 1px 0 rgba(255,255,255,0.9);
}
.herb-tag-name {
    display: flex;
    align-items: center;
    gap: 6px;
    min-height: 40px;
    padding: 8px 13px 8px 12px;
    list-style: none;
    cursor: pointer;
    outline: none;
}
.herb-tag-name::-webkit-details-marker {
    display: none;
}
.herb-tag-name::before {
    content: "";
    display: inline-block;
    width: 6px; height: 6px; border-radius: 50%;
    background: radial-gradient(circle, var(--a-500) 0%, var(--a-400) 100%);
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(255,183,77,0.18);
}
.herb-tag-name::after {
    content: "?";
    margin-left: auto;
    width: 18px;
    height: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    color: #9a5a1b;
    background: rgba(255,183,77,0.16);
    border: 1px solid rgba(230,81,0,0.15);
    font-size: 0.7rem;
    font-weight: 800;
}
.herb-explanation {
    position: absolute;
    top: calc(100% + 8px);
    left: 0;
    z-index: 60;
    width: min(280px, 76vw);
    padding: 11px 13px;
    border-radius: 10px;
    border: 1px solid rgba(230,81,0,0.18);
    background: rgba(255,252,246,0.99);
    box-shadow: 0 10px 28px rgba(93,64,55,0.16);
    color: #5d4037 !important;
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1.7;
    white-space: normal;
    word-break: break-word;
}
.herb-explanation::before {
    content: "";
    position: absolute;
    top: -6px;
    left: 20px;
    width: 10px;
    height: 10px;
    background: #fffcf6;
    border-left: 1px solid rgba(230,81,0,0.18);
    border-top: 1px solid rgba(230,81,0,0.18);
    transform: rotate(45deg);
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
.sidebar-nav-heading {
    margin: 0 0 4px 0;
    color: #ffffff;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 1px;
}
.sidebar-nav-caption {
    margin: 0 0 10px 0;
    color: rgba(255,255,255,0.68) !important;
    font-size: 0.74rem;
    line-height: 1.55;
}
.sidebar-section-label {
    margin: 6px 0 2px 0;
    color: rgba(255,255,255,0.82) !important;
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.8px;
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
[data-testid="stSidebar"] .stButton > button {
    min-height: 42px;
    height: auto;
    padding: 10px 13px;
    justify-content: flex-start;
    text-align: left;
    border-radius: 9px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.055);
    box-shadow: none;
    font-size: 0.86rem;
    font-weight: 650;
    letter-spacing: 0;
}
[data-testid="stSidebar"] .stButton > button p {
    white-space: normal !important;
    line-height: 1.45;
    text-align: left;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(255,255,255,0.24);
    background: rgba(255,255,255,0.10);
    transform: none;
    box-shadow: none;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    color: #123816 !important;
    background: linear-gradient(135deg, #f1fbf2 0%, #ccebd0 100%) !important;
    border-color: rgba(255,255,255,0.72) !important;
    box-shadow: 0 5px 16px rgba(0,0,0,0.14) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] span {
    color: #123816 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #ffffff 0%, #d9f2dc 100%) !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    color: rgba(255,255,255,0.42) !important;
    border-color: rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.025);
    opacity: 1;
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
.feature-level-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin: 8px 0 16px 0;
    color: #355e3b;
    font-size: 0.84rem;
}
.feature-level-legend span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 9px;
    border-radius: 999px;
    background: rgba(232, 245, 233, 0.92);
    border: 1px solid rgba(76, 175, 80, 0.22);
}
.feature-level-legend b {
    color: var(--p-700);
}
div[data-testid="stNumberInput"] label p {
    color: var(--p-700);
    font-size: 0.8rem;
    font-weight: 650;
}
div[data-testid="stNumberInput"] input {
    color: var(--p-700);
    font-weight: 700;
    text-align: center;
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

/* ========== 原生 Caption / 辅助文字（提高白底对比度） ========== */
[data-testid="stCaptionContainer"] {
    font-size: 0.86rem !important;
    color: var(--n-700) !important;
    line-height: 1.8 !important;
}
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
    color: var(--n-700) !important;
}
small, .stMarkdown small { color: var(--n-700) !important; }
/* 说明性正文 → 统一加深，避免浅灰/浅绿在白底上看不清 */
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: var(--n-900);
}
[data-testid="stWidgetLabel"] p {
    color: var(--n-900) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: var(--n-900) !important;
    font-weight: 650;
}
.stAlert p, .stAlert span, .stAlert div {
    color: #1f2a23 !important;
}

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
    .result-summary {
        grid-template-columns: 1fr;
        gap: 16px;
        padding: 20px 18px;
    }
    .result-summary-title { font-size: 1.45rem; }
    .result-summary-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .result-module-header {
        align-items: flex-start;
        padding: 14px;
        gap: 11px;
    }
    .result-module-index {
        flex-basis: 36px;
        width: 36px;
        height: 36px;
        border-radius: 9px;
    }
    .result-module-title { font-size: 1rem; }
    .stTabs [data-baseweb="tab-list"] {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 4px;
        padding: 4px;
        overflow-x: visible;
    }
    .stTabs [data-baseweb="tab"] {
        width: 100%;
        min-height: 42px;
        padding: 8px 12px;
        font-size: 0.82rem;
        justify-content: center;
    }
}
@media (max-width: 480px) {
    [data-testid="stAppViewContainer"] > .main { padding-top: 0.8rem; }
    .brand-title { font-size: 1.55rem; letter-spacing: 1px; }
    .brand-subtitle { font-size: 0.82rem; line-height: 1.6; }
    .hero-value { font-size: 1.45rem; flex-direction: column; align-items: flex-start; gap: 8px; }
    .kpi-icon { width: 38px; height: 38px; font-size: 1.35rem; }
    .formula-card { padding: 18px 16px; }
    .diagnosis-hero { padding: 20px 16px; }
    .result-summary { padding: 18px 14px; }
    .result-summary-metrics { gap: 6px; }
    .result-summary-metric { min-height: 68px; padding: 10px 6px; }
    .result-summary-metric strong { font-size: 1.05rem; }
    .result-summary-metric span { font-size: 0.68rem; }
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
# 6证型 经典方剂知识（仅展示方名、出处与药材名录，不含剂量）
# ============================================================
SYNDROME_TREATMENT = {
    "气阴两虚": {
        "diagnosis_summary": "气虚与阴虚并存，以脏腑功能减退与阴液亏虚同见为核心病机。脾气亏虚则运化乏力、倦怠少气；肺胃阴虚则津液不布、口渴多饮。病位以脾、肺、胃为主，久则累及心肾。",
        "treatment_principle": "益气养阴，健脾生津，兼顾固肾",
        "formula": {
            "name": "生脉散合六味地黄丸加减",
            "source": "《医学启源》生脉散 + 《小儿药证直诀》六味地黄丸化裁",
            "rationale": "结合患者“气阴两虚”病机，治以益气养阴、健脾生津，兼顾固肾。方中以生脉散（太子参、麦冬、五味子）益气养阴、敛汗生脉，配伍黄芪补气升阳，针对气虚津伤之本；辅以六味地黄丸（熟地黄、山萸肉、怀山药等）滋补肝肾之阴，针对阴虚内热之标；白术、茯苓、甘草健脾和中，使气血生化有源。诸药合用，共奏气阴双补、标本兼顾之效。（教学提示：此推演旨在训练学生“以法统方”的辨证思维。）",
            "herbs": [
                "黄芪", "太子参", "麦冬", "五味子",
                "生地黄", "山萸肉", "怀山药", "茯苓",
                "牡丹皮", "泽泻", "白术", "甘草",
            ]
        },
        "addition_subtraction": [
            "口渴甚者：加 天花粉、知母 清热生津",
            "自汗明显者：加 浮小麦、煅牡蛎 固表敛汗",
            "便溏者：减生地黄，加 炒扁豆、薏苡仁 健脾止泻",
            "失眠心悸者：加 酸枣仁、远志 养心安神",
            "腰膝酸软甚者：加 桑寄生、川牛膝 补肝肾强筋骨",
        ],
        "care_theory": [
            {
                "icon": "🏃",
                "title": "运动治则",
                "text": "气阴两虚者，气不摄津，阴不敛阳，宜选择太极拳、八段锦等柔和运动以益气养阴，切忌大汗淋漓。大汗不仅伤阴，更易耗气，加重虚损。",
            },
            {
                "icon": "🍽️",
                "title": "施膳逻辑",
                "text": "阴虚则生内热，故饮食当以甘润生津为原则。山药粥健脾益气，银耳百合汤滋阴清热。忌食辛辣燥烈之品，以防“火上浇油”更伤阴津。",
            },
            {
                "icon": "🧘",
                "title": "情志与病机",
                "text": "患者常因久病而思虑过度。中医认为“思伤脾”，脾虚则气血生化之源不足，加重气阴两虚。故需保持心境平和。",
            },
            {
                "icon": "🌙",
                "title": "起居治则",
                "text": "肝开窍于目，肝肾阴虚则目失所养。应避免久视伤血，子时（23点）前入睡，以顺应肝胆经气之生发，滋养肝阴。",
            },
        ],
        "reflection": {
            "question": "若该患者（气阴两虚证）近期因情绪波动，突然出现口苦咽干、心烦易怒、舌边红等新症状，在临床带教中，应提示学生考虑合并了哪种病机？在饮食调护上应做何调整？",
            "hint": "应提示学生考虑肝郁化火、横逆犯胃，兼有郁热伤津之象。饮食调护宜在益气养阴的基础上兼顾清肝解郁，少食辛辣燥烈与醇酒，可适量选用玫瑰花、菊花等疏肝清热之品，并重视情志疏导。",
        },
    },

    "痰热互结": {
        "diagnosis_summary": "痰浊与邪热交结，阻滞中焦气机。多由肥甘厚味酿生痰湿，郁久化热；或肝郁化火、炼液成痰所致。病位以脾、胃、肝为主，常见于形体丰腴之人。",
        "treatment_principle": "清热化痰，理气散结，健脾和中",
        "formula": {
            "name": "黄连温胆汤合小陷胸汤加减",
            "source": "《六因条辨》黄连温胆汤 + 《伤寒论》小陷胸汤化裁",
            "rationale": "结合“痰热互结、阻滞中焦”的病机，治以清热化痰、理气散结、健脾和中。方中黄连温胆汤（黄连、法半夏、竹茹、枳实、陈皮、茯苓等）清化痰热、和胃降逆；小陷胸汤（黄连、法半夏、瓜蒌）辛开苦降、宽胸散结；柴胡、黄芩疏肝清胆，浙贝母、胆南星增强化痰散结之力，甘草调和诸药。全方清热而不忘理气，化痰而兼顾健脾，使痰热分消、气机通畅。（教学提示：此推演旨在训练学生“以法统方”的辨证思维。）",
            "herbs": [
                "黄连", "黄芩", "法半夏", "竹茹",
                "枳实", "全瓜蒌", "陈皮", "茯苓",
                "浙贝母", "胆南星", "柴胡", "甘草",
            ]
        },
        "addition_subtraction": [
            "脘腹胀甚者：加 厚朴、大腹皮 理气消胀",
            "大便黏滞不爽者：加 木香、槟榔 导滞通腑",
            "口苦胁痛者：加 龙胆草、郁金 清胆疏肝",
            "纳呆食少者：加 神曲、炒谷麦芽 健胃消食",
            "痰浊壅盛、胸闷咯痰者：加 薤白、桔梗 开胸化痰",
        ],
        "care_theory": [
            {
                "icon": "🏃",
                "title": "运动治则",
                "text": "痰湿内阻者宜以适度运动振奋阳气、运化水湿。可选择快走、八段锦或舒展训练，以微微汗出为度；痰热未清时不宜骤然剧烈运动，以免气机逆乱、津随汗泄。",
            },
            {
                "icon": "🍽️",
                "title": "施膳逻辑",
                "text": "脾为生痰之源，痰热互结者当以清热化痰、健脾利湿为原则。宜选芹菜、冬瓜、海带等清淡之品，少食肥甘厚味、甜腻糕点及醇酒辛辣，以断痰湿化热之源。",
            },
            {
                "icon": "🧘",
                "title": "情志与病机",
                "text": "肝主疏泄，情志郁结则气机不畅，郁久化火并可炼液成痰。故应保持心境舒畅，避免忧思郁怒，使肝气条达、脾胃升降有序。",
            },
            {
                "icon": "🌙",
                "title": "起居治则",
                "text": "痰湿得温则行、得寒则凝，居室宜干燥通风，避免久居潮湿环境。作息当规律，避免熬夜伤及肝脾，使水湿运化有常。",
            },
        ],
        "reflection": {
            "question": "若患者痰热证候减轻后，逐渐出现食欲不振、便溏、舌苔白腻，教学时应提示学生考虑何种病机转化？调护重点应如何调整？",
            "hint": "应提示学生考虑痰热渐化而脾虚湿困未复。调护重点宜由单纯清热化痰转向健脾化湿、理气和中，饮食更应避免生冷甜腻，并观察痰湿与脾虚的消长。",
        },
    },

    "肝肾阴虚": {
        "diagnosis_summary": "肝肾阴液亏虚，虚热内生。肝阴不足则目失所养、筋脉失濡；肾阴亏虚则骨髓不充、腰膝失养。病程多较长，阴损及阳之前期阶段。",
        "treatment_principle": "滋补肝肾，育阴潜阳，明目益精",
        "formula": {
            "name": "杞菊地黄丸合一贯煎化裁",
            "source": "《麻疹全书》杞菊地黄丸 + 《续名医类案》一贯煎加减",
            "rationale": "结合“肝肾阴虚、虚热内生”的病机，治以滋补肝肾、育阴潜阳、明目益精。方中杞菊地黄丸以六味地黄丸（熟地黄、山萸肉、怀山药、茯苓、牡丹皮、泽泻）滋补肝肾之阴，枸杞子、杭菊花养肝明目；一贯煎以北沙参、麦冬、当归身滋阴养血，川楝子疏肝理气，使滋养而不滞。诸药合用，既补肝肾之本，又兼顾肝郁与目窍失养之标。（教学提示：此推演旨在训练学生“以法统方”的辨证思维。）",
            "herbs": [
                "枸杞子", "杭菊花", "熟地黄", "山萸肉",
                "怀山药", "茯苓", "牡丹皮", "泽泻",
                "北沙参", "麦冬", "当归身", "川楝子",
            ]
        },
        "addition_subtraction": [
            "视物昏花甚者：加 女贞子、旱莲草、决明子 益肝明目",
            "潮热盗汗者：加 地骨皮、银柴胡、浮小麦 清虚热敛汗",
            "头晕耳鸣者：加 天麻、钩藤（后下）、磁石（先煎）平肝潜阳",
            "遗精滑泄者：加 金樱子、芡实、莲须 固肾涩精",
            "失眠多梦：加 酸枣仁、夜交藤 养血安神",
        ],
        "care_theory": [
            {
                "icon": "🏃",
                "title": "运动治则",
                "text": "肝主筋、肾主骨，肝肾阴虚者宜练太极拳、八段锦等柔和功法，以养筋健骨、调畅气机。运动宜舒缓有度，切忌大汗淋漓，以免汗出伤阴、耗损肝肾之精。",
            },
            {
                "icon": "🍽️",
                "title": "施膳逻辑",
                "text": "阴虚则内热，食疗当以甘润滋补为原则。枸杞子、黑芝麻、桑椹、黑豆等可滋养肝肾；少食辛辣温燥、煎炸炙煿之品，以防助火伤阴。",
            },
            {
                "icon": "🧘",
                "title": "情志与病机",
                "text": "肝藏血、主疏泄，郁怒日久易化火伤阴。应保持情绪平稳，避免长期紧张与恼怒，使肝血得养、肝气条达。",
            },
            {
                "icon": "🌙",
                "title": "起居治则",
                "text": "肝开窍于目，久视则伤血；肾主藏精，熬夜则耗精。宜子时前入睡，避免久视劳神，使肝肾精血得以休养。",
            },
        ],
        "reflection": {
            "question": "若患者新出现头晕耳鸣、五心烦热、舌红少苔，教学时应如何判断阴虚与虚火的主次？饮食调护应侧重什么？",
            "hint": "应提示学生考虑肝肾阴虚、虚火偏亢，病机重点由单纯阴亏转向阴虚火旺。饮食调护宜坚持甘润滋养，避免辛辣温燥，并兼顾清虚热与养肝明目。",
        },
    },

    "热盛伤津": {
        "diagnosis_summary": "燥热炽盛，耗伤肺胃津液。上焦肺热则口渴多饮，中焦胃热则消谷善饥，热结下焦则小便黄赤、肠燥便干。多见于病之早中期，正气尚充、邪热正盛。",
        "treatment_principle": "清热泻火，生津止渴，兼顾益气护阴",
        "formula": {
            "name": "消渴方合白虎加人参汤加减",
            "source": "《丹溪心法》消渴方 + 《伤寒论》白虎加人参汤化裁",
            "rationale": "结合“燥热炽盛、耗伤肺胃津液”的病机，治以清热泻火、生津止渴，兼顾益气护阴。方中白虎加人参汤以生石膏、知母清阳明气分实热，西洋参益气生津，粳米、炙甘草护胃和中；消渴方以黄连、天花粉、生地黄清热泻火、养阴生津，麦冬、葛根、黄芩、栀子清热润燥。全方清中有养，使热退而津复、气阴不伤。（教学提示：此推演旨在训练学生“以法统方”的辨证思维。）",
            "herbs": [
                "生石膏（先煎）", "知母", "西洋参（另炖兑服）", "天花粉",
                "生地黄", "黄连", "麦冬", "葛根",
                "黄芩", "栀子", "粳米", "炙甘草",
            ]
        },
        "addition_subtraction": [
            "口渴甚者：加 鲜芦根、乌梅 生津止渴",
            "多食易饥甚者：加 黄连，配 升麻 清胃泻火",
            "大便秘结、舌苔黄燥者：加 生大黄（后下）、玄明粉（冲） 急下存阴",
            "心烦失眠、口舌生疮：加 竹叶心、莲子心、连翘心 清泻心火",
            "牙龈肿痛、口臭：加 升麻、牡丹皮 清泻胃火凉血",
        ],
        "care_theory": [
            {
                "icon": "🏃",
                "title": "运动治则",
                "text": "热盛伤津时应避免高温环境下的剧烈运动，可选清晨或傍晚缓步行走、呼吸吐纳等和缓活动，以不疲劳、不大汗为度。热邪未清而强行运动，易致津液更伤。",
            },
            {
                "icon": "🍽️",
                "title": "施膳逻辑",
                "text": "燥热伤津，饮食当以清热生津、甘润为宜。苦瓜、黄瓜、梨、藕等可辅助清热生津，少食辛辣、烧烤、煎炸及温燥补品，以防助热伤津。",
            },
            {
                "icon": "🧘",
                "title": "情志与病机",
                "text": "火热易扰心神，患者可出现心烦、急躁。应保持环境安静，避免情绪激动与过度劳神，使心神安定、火热渐平。",
            },
            {
                "icon": "🌙",
                "title": "起居治则",
                "text": "热邪易伤津，起居宜凉爽通风，避免高温暴晒与熬夜。衣着宜透气，保持二便通畅，使邪热有外出之路。",
            },
        ],
        "reflection": {
            "question": "若患者壮热渐退后反而出现神疲乏力、口渴自汗、脉细无力，教学时应提示学生考虑何种病机变化？调护重点应如何调整？",
            "hint": "应提示学生考虑热邪耗气伤津，病机由热盛伤津转为气阴两伤。调护重点宜从单纯清热转向益气养阴、生津护胃，并避免过用苦寒损伤脾胃。",
        },
    },

    "肝胃郁热": {
        "diagnosis_summary": "肝气郁结化火，横逆犯胃，肝胃同病。情志不遂为常见诱因，肝火上扰则烦躁易怒、口苦面赤；胃热炽盛则消谷善饥、大便秘结。病位在肝、胃，与情绪密切相关。",
        "treatment_principle": "清肝泻胃，理气解郁，通腑泄热",
        "formula": {
            "name": "大柴胡汤合左金丸化裁",
            "source": "《金匮要略》大柴胡汤 + 《丹溪心法》左金丸加减",
            "rationale": "结合“肝郁化火、横逆犯胃”的病机，治以清肝泻胃、理气解郁、通腑泄热。方中大柴胡汤以柴胡、黄芩和解少阳、清泄郁热，生大黄、枳实通腑泄热，法半夏、生姜、大枣和胃降逆；左金丸以黄连、吴茱萸清肝泻火、降逆制酸；白芍、郁金、川楝子柔肝疏肝、理气止痛。诸药合用，使肝火得清、胃气得降、腑气通畅。（教学提示：此推演旨在训练学生“以法统方”的辨证思维。）",
            "herbs": [
                "柴胡", "黄芩", "黄连", "吴茱萸",
                "生大黄（后下）", "枳实", "白芍", "法半夏",
                "生姜", "大枣", "郁金", "川楝子",
            ]
        },
        "addition_subtraction": [
            "烦躁易怒、胁痛甚者：加 龙胆草、醋香附、青皮 疏肝理气止痛",
            "嗳气泛酸：加 乌贼骨（先煎）、浙贝母、煅瓦楞子（先煎） 抑酸和胃",
            "大便干结甚者：加 玄明粉（冲）、厚朴 通腑泻下",
            "胸胁胀痛、善太息：加 佛手、绿萼梅、玫瑰花 疏肝理气解郁",
            "头痛目赤、血压偏高：加 天麻、钩藤（后下）、夏枯草 平肝潜阳",
        ],
        "care_theory": [
            {
                "icon": "🏃",
                "title": "运动治则",
                "text": "肝喜条达而恶抑郁，宜选择快走、八段锦等节律舒缓的运动以疏肝理气。运动以微汗为度，避免在愤怒或情绪激动时剧烈活动，以免肝火更盛。",
            },
            {
                "icon": "🍽️",
                "title": "施膳逻辑",
                "text": "肝胃郁热者饮食宜有节，少食多餐，避免暴饮暴食及醇酒、辛辣、浓茶咖啡。可适量选用萝卜、金橘、佛手等理气之品，以助肝胃气机和降。",
            },
            {
                "icon": "🧘",
                "title": "情志与病机",
                "text": "情志不遂是肝郁化火的重要诱因。应重视情志疏导，避免恼怒忧思，使肝气条达、胃气和降，防止郁火进一步伤及胃阴。",
            },
            {
                "icon": "🌙",
                "title": "起居治则",
                "text": "肝胃郁热与作息失调、进食不节相关。宜规律作息、避免熬夜，饭后缓行以助运化，居室保持安静舒适，使肝胃得以调养。",
            },
        ],
        "reflection": {
            "question": "若患者在清肝泻热后出现食欲减退、便溏、畏寒、舌淡苔白，教学时应提示学生警惕什么？后续调护应如何避免偏颇？",
            "hint": "应提示学生警惕苦寒清降太过而损伤脾胃阳气，出现肝胃郁热未清与脾胃虚寒并见。调护应顾护脾胃，饮食避免生冷，清热不可过剂，并动态复核证候。",
        },
    },

    "阴阳两虚": {
        "diagnosis_summary": "阴损及阳，阴阳俱虚。多为病程迁延日久，由阴虚发展而来。阴虚本在，又见畏寒肢冷、面浮肢肿等阳虚失温之象。病位深及下焦肾元，为病之后期阶段。",
        "treatment_principle": "温阳滋阴，补肾固摄，兼补脾益气",
        "formula": {
            "name": "金匮肾气丸合水陆二仙丹化裁",
            "source": "《金匮要略》肾气丸 + 《洪氏集验方》水陆二仙丹化裁",
            "rationale": "结合“阴损及阳、阴阳俱虚”的病机，治以温阳滋阴、补肾固摄，兼补脾益气。方中金匮肾气丸以熟附子、桂枝温补肾阳，熟地黄、山萸肉、怀山药滋阴填精，取“阴中求阳”之意；茯苓、泽泻、牡丹皮利水渗湿、泄浊行滞；水陆二仙丹以金樱子、芡实固肾涩精，黄芪、白术健脾益气，补骨脂、淫羊藿温肾助阳。全方阴阳并补、脾肾兼顾，使阳气渐复而阴精不伤。（教学提示：此推演旨在训练学生“以法统方”的辨证思维。）",
            "herbs": [
                "熟附子（先煎）", "桂枝", "熟地黄", "山萸肉",
                "怀山药", "茯苓", "牡丹皮", "泽泻",
                "金樱子", "芡实", "补骨脂", "淫羊藿",
                "黄芪", "白术",
            ]
        },
        "addition_subtraction": [
            "畏寒肢冷甚者：加 肉桂粉（冲）、干姜 增温阳之力",
            "下肢浮肿明显：加 车前子（包）、冬瓜皮、川牛膝 利水消肿",
            "夜尿频多：加 桑螵蛸、益智仁、乌药 温肾缩尿",
            "五更泄、便溏：加 肉豆蔻、吴茱萸、五味子 温脾暖肾涩肠",
            "胸闷心悸、唇舌紫暗：加 丹参、桃仁、红花 活血化瘀通络",
        ],
        "care_theory": [
            {
                "icon": "🏃",
                "title": "运动治则",
                "text": "阴阳两虚者宜温养而不宜耗散，可选择八段锦、静坐、缓步等柔和活动，并适度接受温和日晒以助阳气。运动以微汗为度，切忌大汗淋漓，以免阳随汗泄、阴液更伤。",
            },
            {
                "icon": "🍽️",
                "title": "施膳逻辑",
                "text": "阴阳俱虚者宜平补阴阳、温而不燥。山药、黑豆、核桃等可健脾补肾，羊肉、生姜等温性食物宜适量，少食生冷瓜果与冰饮，也不可过用辛辣燥烈之品。",
            },
            {
                "icon": "🧘",
                "title": "情志与病机",
                "text": "久病易致思虑忧惧，恐则伤肾、思则伤脾。应给予情志支持，帮助患者保持心境平和，使脾肾气机调畅、阴阳渐复。",
            },
            {
                "icon": "🌙",
                "title": "起居治则",
                "text": "阳虚失温则畏寒，阴虚失养则易疲劳。应注意腰腹与足部保暖，规律作息，睡前可温水泡脚，避免熬夜与过度劳累，以护养元阴元阳。",
            },
        ],
        "reflection": {
            "question": "若患者在畏寒肢冷、下肢浮肿的同时新出现口干、夜间盗汗、舌红少苔，教学时应如何理解阴阳两虚中的虚火表现？调护应避免哪些偏颇？",
            "hint": "应提示学生考虑阴阳两虚而虚火偏亢，属于阴阳俱损、虚热内生之复杂证候。调护应避免温阳过于温燥、滋阴过于滋腻，饮食宜平补阴阳并动态观察水肿、口干与舌象变化。",
        },
    },
}


# 经典方剂药味释义：仅用于教学识药，不提供剂量与用法。
HERB_EXPLANATIONS = {
    "黄芪": "补气升阳、固表止汗、利水消肿，常用于气虚乏力、表虚自汗。",
    "太子参": "益气健脾、生津润肺，药性平和，常用于气阴不足而虚不受补者。",
    "麦冬": "养阴生津、润肺清心，常用于肺胃阴虚、口燥咽干。",
    "五味子": "敛肺滋肾、生津敛汗、宁心安神，酸收之中兼有补益。",
    "生地黄": "清热凉血、养阴生津，为滋阴清热常用药。",
    "山萸肉": "补益肝肾、收涩固脱，常用于肝肾亏虚、精气不固。",
    "怀山药": "补脾养胃、生津益肺、补肾涩精，属平补脾肺肾之品。",
    "茯苓": "利水渗湿、健脾宁心，常助脾运而化湿浊。",
    "牡丹皮": "清热凉血、活血化瘀，常用于阴虚内热或血热瘀滞。",
    "泽泻": "利水渗湿、泄热化浊，常与补益药同用以防滋腻留湿。",
    "白术": "健脾益气、燥湿利水，为健脾运湿的常用药。",
    "甘草": "补脾益气、调和诸药，生用与炙用功效侧重不同。",
    "炙甘草": "补脾益气、缓急止痛、调和诸药，偏于温补和中。",
    "黄连": "清热燥湿、泻火解毒，尤善清中焦湿热与胃火。",
    "黄芩": "清热燥湿、泻火解毒，善清中上焦湿热。",
    "法半夏": "燥湿化痰、降逆止呕，为化痰和胃常用药。",
    "竹茹": "清热化痰、除烦止呕，常用于痰热烦扰与胃热呕逆。",
    "枳实": "破气消积、化痰散痞，常用于气滞痰阻、胸脘痞满。",
    "全瓜蒌": "清热涤痰、宽胸散结、润燥滑肠，善治痰热胸痹。",
    "陈皮": "理气健脾、燥湿化痰，可调畅中焦气机。",
    "浙贝母": "清热化痰、散结消痈，偏于清肺化痰散结。",
    "胆南星": "清热化痰、息风定惊，常用于痰热内盛。",
    "柴胡": "疏肝解郁、升举阳气、和解少阳，为调畅气机要药。",
    "枸杞子": "滋补肝肾、益精明目，常用于肝肾阴虚、目暗昏花。",
    "杭菊花": "疏散风热、平肝明目、清热解毒，常用于目赤昏花。",
    "熟地黄": "补血滋阴、益精填髓，为滋补肝肾阴血的重要药味。",
    "北沙参": "养阴清肺、益胃生津，常用于肺胃阴伤。",
    "当归身": "补血活血、润肠通便，偏于养血和血。",
    "川楝子": "疏肝泄热、行气止痛，常用于肝郁化火之胁痛。",
    "生石膏": "清热泻火、除烦止渴，善清气分实热。",
    "知母": "清热泻火、滋阴润燥，常与石膏配伍清气分热。",
    "西洋参": "补气养阴、清热生津，适用于气阴两伤而兼热象者。",
    "天花粉": "清热泻火、生津止渴、消肿排脓，常用于热盛津伤。",
    "葛根": "解肌退热、生津止渴、升阳止泻，兼能升发脾胃清阳。",
    "栀子": "泻火除烦、清热利湿、凉血解毒，可清三焦郁热。",
    "粳米": "益气和胃、养阴生津，常作方中护胃和中之品。",
    "吴茱萸": "散寒止痛、降逆止呕、疏肝下气，常与黄连配伍调肝胃。",
    "生大黄": "泻下攻积、清热泻火、逐瘀通经，药力较峻。",
    "白芍": "养血调经、敛阴止汗、柔肝止痛，常用于肝血不足。",
    "生姜": "解表散寒、温中止呕、化痰止咳，可调和胃气。",
    "大枣": "补中益气、养血安神，常用于脾胃虚弱与方剂调和。",
    "郁金": "活血止痛、行气解郁、清心凉血、利胆退黄。",
    "熟附子": "回阳救逆、补火助阳、散寒止痛，属温阳要药。",
    "桂枝": "发汗解肌、温通经脉、助阳化气，常用于寒凝与阳虚。",
    "金樱子": "固精缩尿、涩肠止泻，偏于收敛固摄。",
    "芡实": "益肾固精、补脾止泻、除湿止带，脾肾双补而兼收涩。",
    "补骨脂": "温肾助阳、纳气平喘、温脾止泻，常用于肾阳不足。",
    "淫羊藿": "补肾阳、强筋骨、祛风湿，常用于肾阳虚衰。",
}


# ============================================================
# 辅助：6个典型病例特征（与药方/证型一一对应）
# ============================================================
CASE_STUDIES = {
    "病例A · 气阴两虚型": {
        'patient': '张某', 'sex': '男', 'age': 58, 'course': '5年',
        'description': '男，58岁，病程5年。口渴多饮，倦怠乏力，动则气短，时自汗出，腰膝酸软，视物昏花，舌质淡红偏干，苔薄白少津，脉细弱。',
        'features': [0.70, 0.30, 0.82, 0.22, 0.20, 0.28, 0.32, 0.40, 0.72, 0.65, 0.20, 0.18, 0.28, 0.55, 0.72],
    },
    "病例B · 痰热互结型": {
        'patient': '李某', 'sex': '女', 'age': 46, 'course': '4年',
        'description': '女，46岁，病程4年，BMI 29.1。口苦口黏，脘腹胀满，形体肥胖，心烦急躁，大便黏滞不爽，舌红苔黄腻，脉滑数。',
        'features': [0.62, 0.42, 0.48, 0.72, 0.82, 0.52, 0.72, 0.90, 0.30, 0.28, 0.18, 0.28, 0.92, 0.72, 0.42],
    },
    "病例C · 肝肾阴虚型": {
        'patient': '王某', 'sex': '男', 'age': 66, 'course': '11年',
        'description': '男，66岁，病程11年。腰膝酸软，双目干涩视物模糊，头晕耳鸣，口干不欲多饮，失眠多梦，舌红少苔，脉沉细略数。',
        'features': [0.62, 0.30, 0.52, 0.28, 0.32, 0.38, 0.40, 0.62, 0.92, 0.90, 0.38, 0.28, 0.38, 0.72, 0.82],
    },
    "病例D · 热盛伤津型": {
        'patient': '赵某', 'sex': '男', 'age': 51, 'course': '2年',
        'description': '男，51岁，病程2年。大渴引饮，消谷善饥，口苦口臭，小便黄赤量多，大便燥结，舌红苔黄燥，脉洪数有力。',
        'features': [0.96, 0.92, 0.32, 0.52, 0.80, 0.92, 0.92, 0.32, 0.22, 0.28, 0.10, 0.10, 0.62, 0.92, 0.42],
    },
    "病例E · 肝胃郁热型": {
        'patient': '陈某', 'sex': '女', 'age': 47, 'course': '4年',
        'description': '女，47岁，病程4年。平素急躁易怒，胸胁胀满，口苦反酸，多食易饥，大便秘结3-4日一行，舌红苔黄，脉弦数。',
        'features': [0.72, 0.82, 0.38, 0.96, 0.92, 0.70, 0.92, 0.42, 0.32, 0.28, 0.12, 0.18, 0.50, 0.92, 0.32],
    },
    "病例F · 阴阳两虚型": {
        'patient': '刘某', 'sex': '男', 'age': 72, 'course': '16年',
        'description': '男，72岁，病程16年。畏寒肢冷，足跗浮肿，夜尿频多，腰膝酸软，神疲嗜睡，视物不清，舌淡胖苔白，脉沉细无力。',
        'features': [0.48, 0.22, 0.86, 0.28, 0.18, 0.48, 0.46, 0.58, 0.92, 0.90, 0.96, 0.90, 0.26, 0.38, 0.86],
    },
}


def _clean_ai_text(text):
    """移除面向开发者的内部术语，只保留教师可见的教学解读。"""
    cleaned = str(text or "")
    for term in AI_DEVELOPMENT_TERMS:
        cleaned = cleaned.replace(term, "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _render_case_detail(case_name):
    """在输入页固定展示当前典型病案的完整文本信息。"""
    case = CASE_STUDIES.get(case_name)
    if not case:
        return
    case_type = case_name.split(" · ", 1)[-1]
    st.markdown(
        f"""
        <div class="clinical-case-card">
            <div class="clinical-case-kicker">典型病案已加载 · 请先核对四诊信息</div>
            <div class="clinical-case-title">📋 {case_name}</div>
            <div class="clinical-case-meta">
                <span>患者{case.get('patient', '—')}</span>
                <span>{case.get('sex', '—')} · {case.get('age', '—')}岁</span>
                <span>病程 {case.get('course', '—')}</span>
                <span>教学证型：{case_type}</span>
            </div>
            <div class="clinical-case-description">{case.get('description', '')}</div>
            <div class="clinical-case-footnote">
                ✓ 系统已自动填充 15 项四诊特征。请核对病案与滑块信息，再点击“开始辨证分析”。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_herb_explanation(herb_name):
    """按基础药名匹配释义，兼容“生石膏（先煎）”等炮制说明。"""
    base_name = re.sub(r"（.*?）|\(.*?\)", "", str(herb_name)).strip()
    return HERB_EXPLANATIONS.get(
        base_name,
        "本品为经典方剂组成药味，具体性味归经与功效请结合《中药学》教材学习。",
    )


# ============================================================
# 会话状态：保证侧栏病例、顶部预设按钮与滑块始终同步
# ============================================================
def _normalize_feature_level(value):
    """将界面输入限制为 1-5 的整数程度等级。"""
    try:
        level = int(round(float(value)))
    except (TypeError, ValueError):
        level = DEFAULT_FEATURE_LEVEL
    return min(5, max(1, level))


def _level_to_feature_value(level):
    """仅模型内部使用：1-5 级映射为原有 0.0-1.0 输入范围。"""
    return (_normalize_feature_level(level) - 1) / 4.0


def _value_to_feature_level(value):
    """将病例中的 0.0-1.0 特征映射到最接近的 1-5 级。"""
    try:
        normalized = min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        normalized = DEFAULT_FEATURE_VALUE
    return _normalize_feature_level(round(normalized * 4) + 1)


def _feature_levels_from_values(values):
    return {
        fn: _value_to_feature_level(values.get(fn, DEFAULT_FEATURE_VALUE))
        for fn in FEATURE_NAMES
    }


def _set_feature_levels(levels):
    """同时更新等级、模型输入和两类控件，避免旧 widget state 覆盖新值。"""
    normalized_levels = {
        fn: _normalize_feature_level(levels.get(fn, DEFAULT_FEATURE_LEVEL))
        for fn in FEATURE_NAMES
    }
    st.session_state.feature_levels = normalized_levels
    st.session_state.feature_values = {
        fn: _level_to_feature_value(level)
        for fn, level in normalized_levels.items()
    }
    for fn, level in normalized_levels.items():
        st.session_state[f"lvl_slider_{fn}"] = level
        st.session_state[f"lvl_input_{fn}"] = level
    return normalized_levels


def _set_feature_values(values):
    """兼容原有 0.0-1.0 病例数据，载入时转换为 1-5 级。"""
    return _set_feature_levels(_feature_levels_from_values(values))


def _record_feature_level(feature_name, level):
    """只更新业务状态；控件同步由各 on_change 回调在重绘前完成。"""
    level = _normalize_feature_level(level)
    levels = st.session_state.get("feature_levels")
    if not isinstance(levels, dict):
        levels = {}
    levels[feature_name] = level
    st.session_state.feature_levels = levels

    values = st.session_state.get("feature_values")
    if not isinstance(values, dict):
        values = {}
    values[feature_name] = _level_to_feature_value(level)
    st.session_state.feature_values = values


def _on_feature_level_slider_change(feature_name):
    level = _normalize_feature_level(
        st.session_state.get(f"lvl_slider_{feature_name}", DEFAULT_FEATURE_LEVEL)
    )
    st.session_state[f"lvl_input_{feature_name}"] = level
    _record_feature_level(feature_name, level)


def _on_feature_level_input_change(feature_name):
    level = _normalize_feature_level(
        st.session_state.get(f"lvl_input_{feature_name}", DEFAULT_FEATURE_LEVEL)
    )
    st.session_state[f"lvl_slider_{feature_name}"] = level
    _record_feature_level(feature_name, level)


def _initialize_feature_state():
    """首次进入或版本升级时将 15 项特征统一初始化为 1 级“无症状”。"""
    if st.session_state.get("_feature_state_version") != FEATURE_STATE_VERSION:
        _set_feature_levels(
            {fn: DEFAULT_FEATURE_LEVEL for fn in FEATURE_NAMES}
        )
        st.session_state.selected_case = "手动输入特征"
        st.session_state.case_selectbox = "手动输入特征"
        st.session_state["diagnosis_done"] = False
        st.session_state["current_module"] = "input"
        st.session_state["_feature_state_version"] = FEATURE_STATE_VERSION
        return

    values = st.session_state.get("feature_values", {})
    levels = st.session_state.get("feature_levels")
    if not isinstance(levels, dict):
        levels = _feature_levels_from_values(
            values if isinstance(values, dict) else {}
        )
    levels = {
        fn: _normalize_feature_level(
            levels.get(
                fn,
                _value_to_feature_level(
                    values.get(fn, DEFAULT_FEATURE_VALUE)
                    if isinstance(values, dict)
                    else DEFAULT_FEATURE_VALUE
                ),
            )
        )
        for fn in FEATURE_NAMES
    }
    st.session_state.feature_levels = levels
    st.session_state.feature_values = {
        fn: _level_to_feature_value(level)
        for fn, level in levels.items()
    }

    if "selected_case" not in st.session_state:
        st.session_state.selected_case = "手动输入特征"
    if "case_selectbox" not in st.session_state:
        st.session_state.case_selectbox = st.session_state.selected_case
    if st.session_state.get("current_module") not in VALID_MODULES:
        st.session_state.current_module = "input"


def _apply_case_to_session(case_name):
    """病例加载回调：在控件重绘前强制刷新业务值与全部滑块值。"""
    case = CASE_STUDIES.get(case_name)
    if not case:
        return

    values = {
        fn: float(case["features"][idx])
        for idx, fn in enumerate(FEATURE_NAMES)
    }
    _set_feature_values(values)
    st.session_state.selected_case = case_name
    if "case_selectbox" in st.session_state:
        st.session_state.case_selectbox = case_name
    st.session_state["_loaded_case_notice"] = (
        case_name,
        case.get("description", ""),
    )
    st.session_state["diagnosis_done"] = False
    st.session_state["current_module"] = "input"
    st.session_state["scroll_to_module_top"] = True


def _on_case_change():
    case_name = st.session_state.get("case_selectbox", "手动输入特征")
    if case_name != "手动输入特征":
        _apply_case_to_session(case_name)
    else:
        st.session_state.selected_case = "手动输入特征"
        st.session_state.pop("_loaded_case_notice", None)


def _navigate_to_module(module_id):
    """侧边栏导航回调：切换模块并在重绘后回到当前模块顶部。"""
    if module_id not in VALID_MODULES:
        return
    if module_id != "input" and not st.session_state.get("diagnosis_done", False):
        return
    st.session_state.current_module = module_id
    st.session_state["scroll_to_module_top"] = True


def _current_feature_inputs():
    """从会话状态读取当前 15 项特征，供切回结果模块时继续使用。"""
    stored = st.session_state.get("feature_values", {})
    return {
        fn: float(stored.get(fn, DEFAULT_FEATURE_VALUE))
        for fn in FEATURE_NAMES
    }


def _current_feature_levels():
    """读取当前 1-5 级状态；缺失时从模型输入值安全回推。"""
    stored_levels = st.session_state.get("feature_levels", {})
    values = _current_feature_inputs()
    return {
        fn: _normalize_feature_level(
            stored_levels.get(fn, _value_to_feature_level(values[fn]))
            if isinstance(stored_levels, dict)
            else _value_to_feature_level(values[fn])
        )
        for fn in FEATURE_NAMES
    }


def _restore_feature_level_widget_state(feature_inputs):
    """恢复被 Streamlit 清理的隐藏滑块与数字输入控件状态。"""
    levels = _current_feature_levels()
    for fn in FEATURE_NAMES:
        level = levels.get(fn, _value_to_feature_level(feature_inputs[fn]))
        for widget_key in (f"lvl_slider_{fn}", f"lvl_input_{fn}"):
            if widget_key not in st.session_state:
                st.session_state[widget_key] = level


def _render_feature_level_control(feature_name):
    """渲染单个特征的 1-5 级滑块和可直接输入的等级框。"""
    label = f"{feature_name}（{FEATURE_ATTRIBUTES.get(feature_name, '')}）"
    slider_col, input_col = st.columns(
        [4.7, 1.3],
        vertical_alignment="bottom",
    )
    with slider_col:
        st.slider(
            label,
            min_value=1,
            max_value=5,
            step=1,
            help=create_feature_tooltip(feature_name),
            key=f"lvl_slider_{feature_name}",
            on_change=_on_feature_level_slider_change,
            args=(feature_name,),
        )
    with input_col:
        st.number_input(
            "等级",
            min_value=1,
            max_value=5,
            step=1,
            help="也可直接输入 1-5：1 无，2 轻，3 中，4 重，5 极重。",
            key=f"lvl_input_{feature_name}",
            on_change=_on_feature_level_input_change,
            args=(feature_name,),
        )
    return _normalize_feature_level(
        st.session_state.get(f"lvl_slider_{feature_name}", DEFAULT_FEATURE_LEVEL)
    )


def _schedule_scroll_to_top():
    """在模块内容渲染完成后执行一次短时、可中断的滚动复位。"""
    # Streamlit 会复用内容完全相同的组件 iframe。加入唯一标记后，
    # 每次导航都会重新加载脚本，确保复位逻辑真正执行。
    reset_token = datetime.now().strftime("%Y%m%d%H%M%S%f")
    components.html(
        f"""
        <div id="module-scroll-reset-{reset_token}" aria-hidden="true"></div>
        <script>
        (() => {{
          const parentWindow = window.parent;
          const parentDocument = parentWindow.document;
          const main = parentDocument.querySelector('[data-testid="stMain"]');
          if (!main) return;

          let userInterrupted = false;

          const scrollToTop = () => {{
            if (userInterrupted) return;
            main.scrollTop = 0;
            main.scrollTo({{top: 0, left: 0, behavior: "auto"}});
            parentWindow.scrollTo({{top: 0, left: 0, behavior: "auto"}});
          }};

          // 用户一旦开始正常翻页，立即终止尚未执行的复位，避免“向上跳”。
          const cancelReset = () => {{
            userInterrupted = true;
          }};
          parentWindow.addEventListener("wheel", cancelReset, {{once: true, passive: true}});
          parentWindow.addEventListener("touchstart", cancelReset, {{once: true, passive: true}});
          parentWindow.addEventListener("keydown", cancelReset, {{once: true}});

          scrollToTop();
          parentWindow.requestAnimationFrame(() => {{
            scrollToTop();
            parentWindow.requestAnimationFrame(scrollToTop);
          }});
          parentWindow.setTimeout(scrollToTop, 80);
          parentWindow.setTimeout(scrollToTop, 180);
        }})();
        </script>
        """,
        height=0,
    )


def _render_sidebar_navigation(diagnosis_done, current_module):
    st.sidebar.markdown(
        '<div class="sidebar-nav-heading">🧭 功能导航</div>'
        '<div class="sidebar-nav-caption">选择模块后，右侧仅显示当前内容</div>',
        unsafe_allow_html=True,
    )
    for module_id, number, icon, title in MODULE_NAV_ITEMS:
        is_result_module = module_id != "input"
        st.sidebar.button(
            f"{number} · {icon} {title}",
            key=f"nav_module_{module_id}",
            type="primary" if current_module == module_id else "secondary",
            disabled=is_result_module and not diagnosis_done,
            help=(
                "完成辨证分析后可查看"
                if is_result_module and not diagnosis_done
                else None
            ),
            on_click=_navigate_to_module,
            args=(module_id,),
        )


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
        height=360, showlegend=False, dragmode=False,
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
        st.markdown('<p class="brand-subtitle">🌿 中医智能辅助辨证系统 · 基于 15 项临床特征 · 六类中医证型 · 可解释性决策引擎 · 经典方剂知识溯源</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="usage-guide">
            <span class="usage-guide-badge">操作提示</span>
            <span>请先在左侧“典型病例库”选择教学病案，或手动调整 15 项四诊特征；
            核对病案后点击“开始辨证分析”，再通过左侧功能导航逐模块查看结果。</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_module_header(anchor, number, icon, title, subtitle):
    """渲染顺序展示的结果模块标题，避免标签页隐藏原有功能。"""
    st.markdown(
        f"""
        <div id="{anchor}" class="result-module-header">
            <div class="result-module-index">{number}</div>
            <div>
                <div class="result-module-title">{icon} {title}</div>
                <div class="result-module-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 渲染：AI 解读与多轮追问（固定放在结果页最后）
# ============================================================
def render_ai_interpretation(feature_inputs, probs, shap_dict, pred_s):
    st.markdown(
        '<div class="section-title" style="font-size:1.3rem;">🧠 AI 教研解读与追问</div>',
        unsafe_allow_html=True,
    )

    llm_cfg = get_llm_config()
    if llm_cfg.get("configured"):
        st.caption(
            "本模块仅解读 ML 模型已输出的证型、概率与 SHAP 线索，不会独立重新诊断。"
            f"当前配置：{llm_cfg.get('provider', '?')} / {llm_cfg.get('model', '?')}；"
            "网络或 API 异常时自动切换到本地标准解读。"
        )
    else:
        st.caption(
            "本模块仅解读 ML 模型已输出的证型、概率与 SHAP 线索，不会独立重新诊断。"
            "当前使用本地标准解读，断网时仍可即时输出中医理论教学分析。"
        )

    _session_key = (
        f"chat_session__{pred_s}__"
        f"{hash(tuple(sorted((probs or {}).items())))}"
    )
    if st.session_state.get("_last_diag_key") != _session_key:
        st.session_state["chat_messages"] = []
        st.session_state["chat_system_prompt"] = None
        st.session_state["_first_interp_meta"] = None
        st.session_state["_first_interp_text"] = None
        st.session_state["_last_diag_key"] = _session_key

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        if st.button(
            "🔍 大模型智能解读本次辨证结果",
            disabled=(probs is None or shap_dict is None),
            width="stretch",
            key="btn_first_interp",
        ):
            try:
                with st.spinner("正在检索中医理论并生成辨证解读…"):
                    symptoms = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[:12]}
                    tongue = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[12:]}

                    st.session_state["chat_system_prompt"] = build_chat_system_prompt(
                        symptoms=symptoms,
                        tongue_pulse=tongue,
                        probabilities=probs,
                        shap_dict=shap_dict,
                        feature_names=FEATURE_NAMES,
                        syndrome_labels=SYNDROME_LABELS,
                    )
                    st.session_state["chat_messages"] = []

                    ai_result = generate_llm_interpretation(
                        symptoms=symptoms,
                        tongue_pulse=tongue,
                        probabilities=probs,
                        shap_dict=shap_dict,
                        feature_names=FEATURE_NAMES,
                        syndrome_labels=SYNDROME_LABELS,
                        use_llm=True,
                    )
                    cleaned_text = _clean_ai_text(ai_result["text"])
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": cleaned_text,
                    })
                    st.session_state["_first_interp_text"] = cleaned_text
                    st.session_state["_first_interp_meta"] = {
                        "source": ai_result["source"],
                        "error": ai_result.get("error"),
                    }
            except Exception as e:
                st.error(f"AI 解读失败：{e}")
    with c2:
        if st.button("🗑 清空当前会话", width="stretch", key="btn_clear_chat"):
            st.session_state["chat_messages"] = []
            st.session_state["_first_interp_meta"] = None
            st.session_state["_first_interp_text"] = None
            st.rerun()
    with c3:
        meta = st.session_state.get("_first_interp_meta") or {}
        if meta.get("source") == "mock":
            badge_text = "📝 本地标准解读"
            badge_color = "#8d6e63"
        elif llm_cfg.get("configured"):
            badge_text = "🛰 已接入大模型"
            badge_color = "#1b5e20"
        else:
            badge_text = "📝 本地标准解读"
            badge_color = "#8d6e63"
        st.markdown(
            f'<span style="display:inline-block;background:{badge_color}1a;'
            f'color:{badge_color};border:1px solid {badge_color}40;'
            f'padding:6px 12px;border-radius:999px;font-size:0.82rem;'
            f'font-weight:650;width:100%;text-align:center;margin-top:2px;">{badge_text}</span>',
            unsafe_allow_html=True,
        )

    history = st.session_state.get("chat_messages", [])
    if not history:
        st.info("点击上方按钮可生成首轮解读；网络不可用时，系统会即时使用本地标准答案。")
    else:
        for msg in history:
            with st.chat_message(msg["role"]):
                content = msg["content"]
                if msg["role"] == "assistant":
                    content = _clean_ai_text(content)
                st.markdown(content)
        meta = st.session_state.get("_first_interp_meta")
        if meta and meta.get("error"):
            st.warning(f"大模型调用已降级：{meta['error'][:120]}")

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
            tongue = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[12:]}
            st.session_state["chat_system_prompt"] = build_chat_system_prompt(
                symptoms=symptoms,
                tongue_pulse=tongue,
                probabilities=probs,
                shap_dict=shap_dict,
                feature_names=FEATURE_NAMES,
                syndrome_labels=SYNDROME_LABELS,
            )

        with st.chat_message("assistant"):
            with st.spinner("正在结合四诊信息思考…"):
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
                    reply = f"⚠️ {str(e)}\n\n" + fallback_chat_reply(prompt)

            cleaned_reply = _clean_ai_text(reply)
            st.markdown(cleaned_reply)
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": cleaned_reply,
            })


def build_teaching_report(pred_s, probabilities, feature_values, shap_dict, ai_text):
    """生成可下载的纯文本教研报告，内容不包含任何处方或剂量。"""
    ordered_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    shap_arr = np.asarray(
        shap_dict.get(pred_s, np.zeros(len(FEATURE_NAMES))), dtype=float
    ).flatten()
    if len(shap_arr) < len(FEATURE_NAMES):
        shap_arr = np.pad(shap_arr, (0, len(FEATURE_NAMES) - len(shap_arr)))
    top_idx = np.argsort(-np.abs(shap_arr))[:5]

    feature_lines = [
        f"- {name}：{float(feature_values.get(name, 0.0)):.2f}"
        for name in FEATURE_NAMES
    ]
    probability_lines = [
        f"- {name}：{float(probability):.1%}"
        for name, probability in ordered_probs
    ]
    shap_lines = [
        f"- {FEATURE_NAMES[int(idx)]}：{shap_arr[int(idx)]:+.4f}"
        for idx in top_idx
    ]
    clean_ai_text = _clean_ai_text(ai_text)
    clean_ai_text = clean_ai_text.replace("###", "").replace("**", "").strip()

    return "\n".join([
        "症智明辨·中医辨证教学分析报告",
        "=" * 42,
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "一、主证结论",
        f"- 主证：{pred_s}证",
        f"- 主证匹配度：{float(probabilities[pred_s]):.1%}",
        "",
        "二、六类证型概率分布",
        *probability_lines,
        "",
        "三、临床四诊特征采集（15 项）",
        *feature_lines,
        "",
        "四、主证 SHAP Top 5",
        *shap_lines,
        "",
        "五、AI / 本地标准教研解读",
        clean_ai_text,
        "",
        "六、免责声明",
        "本报告为中医辅助辨证科研教学工具输出，仅供院校师生教研参考，"
        "不构成任何医疗处方、诊断意见或治疗方案。具体诊断与治疗须由执业中医师"
        "四诊合参后确定，严禁照方抓药或自行用药。",
    ])


# ============================================================
# 渲染：诊断主流程
# ============================================================
def render_diagnosis():
    # 加载模型
    with st.spinner('正在初始化诊断引擎...'):
        model, explainer = get_model()

    _initialize_feature_state()

    diagnosis_done = st.session_state.get("diagnosis_done", False)
    if not diagnosis_done and st.session_state.get("current_module") != "input":
        st.session_state.current_module = "input"
    if st.session_state.get("current_module") not in VALID_MODULES:
        st.session_state.current_module = "input"
    current_module = st.session_state.current_module

    # 侧边栏：模块导航
    _render_sidebar_navigation(diagnosis_done, current_module)
    st.sidebar.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.2);margin:1rem 0 0.8rem 0;"></div>',
        unsafe_allow_html=True,
    )

    # 侧边栏：典型病例
    st.sidebar.markdown('<div class="sidebar-brand">📚 典型病例库</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-brand-sub">精选 6 类证型典型病例<br>一键加载完整特征</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="height:1px;background:rgba(255,255,255,0.2);margin:0.6rem 0;"></div>', unsafe_allow_html=True)

    case_opt = st.sidebar.selectbox(
        "加载典型病例：",
        ["手动输入特征"] + list(CASE_STUDIES.keys()),
        help="选择后将自动填充对应证型的特征值",
        key="case_selectbox",
        on_change=_on_case_change,
    )
    _notice = st.session_state.pop("_loaded_case_notice", None)
    if _notice:
        st.sidebar.success(f"✅ 已加载：{_notice[0]}")
        if _notice[1]:
            st.sidebar.info(_notice[1])
    if st.session_state.selected_case != "手动输入特征":
        st.sidebar.button(
            "↻ 重新加载当前病例",
            key="btn_reload_case",
            on_click=_apply_case_to_session,
            args=(st.session_state.selected_case,),
        )

    # ========== 输入区域 ==========
    feature_inputs = _current_feature_inputs()
    feature_levels = _current_feature_levels()
    _restore_feature_level_widget_state(feature_inputs)
    if current_module == "input":
        st.markdown('<div id="feature-input"></div>', unsafe_allow_html=True)
        selected_case = st.session_state.get("selected_case", "手动输入特征")
        if selected_case != "手动输入特征":
            _render_case_detail(selected_case)
        with st.container(border=True):
            st.markdown('<div class="section-title">📝 临床特征采集（15 项）</div>', unsafe_allow_html=True)
            c_intro, c_demo = st.columns([4, 1])
            with c_intro:
                st.caption(
                    "每项按 1-5 级采集：1 无、2 轻、3 中、4 重、5 极重。"
                    "可滑动选择，也可直接输入等级；分析后仍可修改并重新生成结果。"
                )
                st.markdown(
                    '<div class="feature-level-legend">'
                    + "".join(
                        f"<span><b>{level}</b>{label}</span>"
                        for level, label in FEATURE_LEVEL_LABELS.items()
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="privacy-note">🔒 系统已脱敏处理，保护患者隐私，符合教学伦理规范。</div>',
                    unsafe_allow_html=True,
                )
            with c_demo:
                st.button(
                    "一键加载演示病例",
                    key="btn_load_demo_case",
                    on_click=_apply_case_to_session,
                    args=(DEMO_CASE,),
                    help=f"加载「{DEMO_CASE}」的完整四诊特征，适合现场路演快速演示。",
                    width="stretch",
                )

            clinical_features = FEATURE_GROUPS['临床症状']
            col_clinical_1, col_clinical_2, col_tongue = st.columns([2.4, 2.4, 2])

            with col_clinical_1:
                st.markdown("#### 🩺 临床症状 · 1/2")
                for fn in clinical_features[:6]:
                    feature_levels[fn] = _render_feature_level_control(fn)
                    feature_inputs[fn] = _level_to_feature_value(feature_levels[fn])

            with col_clinical_2:
                st.markdown("#### 🩺 临床症状 · 2/2")
                for fn in clinical_features[6:]:
                    feature_levels[fn] = _render_feature_level_control(fn)
                    feature_inputs[fn] = _level_to_feature_value(feature_levels[fn])

            with col_tongue:
                st.markdown("#### 👅 舌象与脉象（3 项）")
                for fn in FEATURE_GROUPS['舌脉特征']:
                    feature_levels[fn] = _render_feature_level_control(fn)
                    feature_inputs[fn] = _level_to_feature_value(feature_levels[fn])

            st.session_state.feature_levels = feature_levels
            for fn in FEATURE_NAMES:
                st.session_state.feature_values[fn] = feature_inputs[fn]

            _, c_btn, _ = st.columns([2, 3, 2])
            with c_btn:
                predict_clicked = st.button(
                    "🔍 开始辨证分析 · 生成诊断报告",
                    width="stretch",
                )
                if predict_clicked:
                    st.session_state["diagnosis_done"] = True
                    st.session_state["current_module"] = "conclusion"
                    st.session_state["scroll_to_module_top"] = True
                    st.rerun()


        if st.session_state.pop("scroll_to_module_top", False):
            _schedule_scroll_to_top()

        return
    # ========== 诊断结果 ==========
    if not diagnosis_done:
        return

    fv = np.array([feature_inputs[f] for f in FEATURE_NAMES])
    with st.spinner("正在进行机器学习辨证与 SHAP 归因分析…"):
        result = predict_syndrome(model, fv)
        shap_dict = get_shap_values(explainer, fv)
    pred_s = result['predicted_syndrome']
    probs = result['probabilities']
    conf = result['confidence']
    confidence_level, confidence_desc = calculate_confidence_level(probs)
    info = SYNDROME_INFO.get(pred_s, {})
    treatment = SYNDROME_TREATMENT.get(pred_s, {})

    # ----- 结果摘要 -----
    icon = get_syndrome_icon(pred_s)
    probability_order = sorted(probs.values(), reverse=True)
    probability_gap = (
        probability_order[0] - probability_order[1]
        if len(probability_order) >= 2 else 0
    )
    related_count = sum(value >= 0.15 for value in probs.values())
    result_summary_html = f"""
    <div class="result-summary">
        <div>
            <div class="result-summary-kicker">辨证分析结果</div>
            <div class="result-summary-title">{icon} {pred_s}证</div>
            <div class="result-summary-desc">{info.get('description', '')}</div>
        </div>
        <div class="result-summary-metrics">
            <div class="result-summary-metric">
                <strong>{conf*100:.1f}%</strong>
                <span>主证匹配度</span>
            </div>
            <div class="result-summary-metric">
                <strong>{confidence_level}</strong>
                <span>辨证置信度</span>
            </div>
            <div class="result-summary-metric">
                <strong>{probability_gap*100:.1f}%</strong>
                <span>与次证差距</span>
            </div>
            <div class="result-summary-metric">
                <strong>{related_count} 类</strong>
                <span>可疑相关证型</span>
            </div>
        </div>
    </div>
    """
    if current_module == "conclusion":
        st.markdown(result_summary_html, unsafe_allow_html=True)
        if confidence_level == "低":
            st.markdown(
                """
                <div class="low-confidence-alert">
                    <span>⚠️</span>
                    <span><b>模型提示：</b>当前输入特征存在交叉，辨证边界模糊，
                    建议结合四诊合参复核。</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----- 仅渲染左侧导航当前选中的结果模块 -----
    if current_module == "conclusion":
        with st.container():
            render_result_module_header(
                "module-conclusion", "01", "📊", "辨证结论",
                "查看六类证型概率、诊断证据与本次辨证的可靠程度。",
            )
            col_prob, col_ev = st.columns([2, 3])
            with col_prob:
                with st.container(border=True):
                    st.markdown('<div class="section-title">📊 六类证型概率分布</div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        create_prob_chart(probs),
                        use_container_width=True,
                        config=PLOTLY_STATIC_CONFIG,
                    )
                    st.markdown("")
                    for syndrome in SYNDROME_LABELS:
                        probability = probs[syndrome]
                        color = get_syndrome_color(syndrome)
                        st.markdown(f"""
                        <div class="prob-bar">
                          <div class="prob-bar-name">{get_syndrome_icon(syndrome)} {syndrome}</div>
                          <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width:{probability*100:.1f}%;background:linear-gradient(90deg,{color}dd,{color}88);"></div>
                          </div>
                          <div class="prob-bar-value">{probability*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

            with col_ev:
                with st.container(border=True):
                    st.markdown(f'<div class="section-title">🧾 诊断证据分析（{pred_s}）</div>', unsafe_allow_html=True)
                    st.caption("基于典型特征范围与当前输入值对比，列出支持与存疑证据。")
                    support, against = build_evidence_list(feature_inputs, pred_s)
                    st.markdown(f"<b style='color:#1b5e20;'>✅ 支持该证型的特征（{len(support)} 项）</b>", unsafe_allow_html=True)
                    for f, v, ref, lvl in support:
                        st.markdown(f'''
                        <div class="evidence-item evidence-support">
                          <span class="evidence-icon">✓</span>
                          <span><b>{f}</b> = {v:.2f}（参考典型均值 {ref:.2f}）· <b>{lvl}</b></span>
                        </div>''', unsafe_allow_html=True)
                    if against:
                        st.markdown(f"<b style='color:#b71c1c;'>⚠️ 不完全支持的特征（{len(against)} 项）</b>", unsafe_allow_html=True)
                        for f, v, ref, reason in against:
                            st.markdown(f'''
                            <div class="evidence-item evidence-against">
                              <span class="evidence-icon">!</span>
                              <span><b>{f}</b> = {v:.2f}（参考典型均值 {ref:.2f}）· {reason}</span>
                            </div>''', unsafe_allow_html=True)

    if current_module == "attribution":
        with st.container():
            render_result_module_header(
                "module-attribution", "02", "🔎", "归因依据",
                "通过 SHAP 判断哪些四诊特征推动了当前证型判断。",
            )
            col_shap, col_patho = st.columns([3, 2])
            with col_shap:
                with st.container(border=True):
                    st.markdown(f'<div class="section-title">📈 主证型关键特征 SHAP 贡献（{pred_s}）</div>', unsafe_allow_html=True)
                    try:
                        ps = shap_dict[pred_s]
                        if hasattr(ps, 'shape') and len(ps.shape) > 1:
                            ps = ps.flatten()
                    except Exception:
                        ps = np.zeros(len(FEATURE_NAMES))
                    st.plotly_chart(
                        create_shap_fig(ps, fv, pred_s),
                        use_container_width=True,
                        config=PLOTLY_SCROLL_SAFE_CONFIG,
                    )

            with col_patho:
                with st.container(border=True):
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
                      <b>📌 诊断把握说明：</b>{confidence_desc}。
                      建议结合舌诊、脉诊、问诊四诊合参，综合判断后确立最终治则方药。
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("点击查看全量 SHAP 特征热力图", expanded=False):
                st.markdown('<div class="section-title">🔥 特征 × 证型 SHAP 贡献矩阵（15 × 6）</div>', unsafe_allow_html=True)
                st.markdown(
                    "热力图展示当前患者各特征对 **六类证型预测** 的 SHAP 贡献方向与强度。"
                    "<span style='color:#053061;font-weight:700;'>■ 深蓝</span> 表示强烈支持该证型，"
                    "<span style='color:#8B0000;font-weight:700;'>■ 深红</span> 表示强烈不支持该证型，"
                    "<span style='color:#455a64;font-weight:700;'>■ 浅色</span> 表示贡献接近中性。"
                    "悬停单元格可查看精确数值与当前输入特征值。",
                    unsafe_allow_html=True,
                )
                try:
                    fig_hm = create_shap_heatmap(shap_dict, fv)
                    st.plotly_chart(
                        fig_hm,
                        use_container_width=True,
                        config=PLOTLY_SCROLL_SAFE_CONFIG,
                    )
                except Exception as _e:
                    st.warning("SHAP 热力图渲染失败：%s" % str(_e))
                    try:
                        labels = [s for s in SYNDROME_LABELS if s in shap_dict]
                        if not labels:
                            labels = list(SYNDROME_LABELS)
                        rows = {}
                        for s in labels:
                            arr = np.asarray(shap_dict.get(s, np.zeros(len(FEATURE_NAMES)))).flatten()
                            if len(arr) < len(FEATURE_NAMES):
                                arr = np.pad(arr, (0, len(FEATURE_NAMES) - len(arr)))
                            rows[s] = arr[:len(FEATURE_NAMES)]
                        df_hm = pd.DataFrame(rows, index=FEATURE_NAMES).round(4)
                        vmax_ab = float(df_hm.abs().max().max())
                        vmax_ab = max(vmax_ab, 1e-6)
                        st.dataframe(
                            df_hm.style.background_gradient(
                                cmap="RdBu", axis=None, vmin=-vmax_ab, vmax=vmax_ab
                            ),
                            width='stretch',
                            height=520,
                        )
                    except Exception:
                        pass

    if current_module == "formula":
        with st.container():
            render_result_module_header(
                "module-formula", "03", "📜", "经典方剂溯源与方义推演",
                "展示经典方剂源流、组方思路与药味释义，不提供临床剂量。",
            )
            st.warning("本系统仅展示经典方剂知识，不提供临床剂量，严禁照方抓药")
            formula_info = treatment.get("formula", {})
            with st.container(border=True):
                st.markdown(f'<div class="section-title">📜 经典方剂溯源（{pred_s}证）</div>', unsafe_allow_html=True)
                st.markdown(f"""<div class="formula-title">经典方剂：{formula_info.get('name', '')}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="font-size:0.88rem;color:#5d4037;margin-bottom:14px;">🕮 出处：{formula_info.get('source', '')}</div>""", unsafe_allow_html=True)
                rationale = formula_info.get("rationale", "")
                if rationale:
                    st.markdown(
                        '<div class="formula-rationale">'
                        '<div class="formula-rationale-title">【方义推演】</div>'
                        f'{html.escape(rationale)}'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('<div style="font-size:0.9rem;color:#4e342e;font-weight:700;margin-bottom:8px;">◆ 经典方剂组成（药材名录，不含剂量）</div>', unsafe_allow_html=True)
                st.caption("点击药名可查看该药味的教学释义，不提供剂量与用法。")
                herbs = list(formula_info.get('herbs', []))
                total = len(herbs)
                n_zhujun = max(1, round(total * 0.25)) if total >= 6 else min(2, total)
                n_zuochen = max(n_zhujun + 1, round(total * 0.60)) if total >= 10 else total
                parts_html = []
                for idx, herb_name in enumerate(herbs):
                    safe_herb_name = html.escape(str(herb_name))
                    safe_explanation = html.escape(_get_herb_explanation(herb_name))
                    parts_html.append(
                        f'<details class="herb-tag">'
                        f'<summary class="herb-tag-name">{safe_herb_name}</summary>'
                        f'<div class="herb-explanation"><b>功效释义：</b>'
                        f'{safe_explanation}</div></details>'
                    )
                    if idx + 1 == n_zhujun and n_zhujun < total:
                        parts_html.append('<div class="herb-group-divider" aria-hidden="true"></div>')
                    elif idx + 1 == n_zuochen and n_zuochen < total:
                        parts_html.append('<div class="herb-group-divider" aria-hidden="true"></div>')
                st.markdown(f'<div class="herb-grid">{"".join(parts_html)}</div>', unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="section-title">⚖️ 随证加减思路（不含剂量）</div>', unsafe_allow_html=True)
                for item in treatment.get('addition_subtraction', []):
                    st.markdown(f'<div class="care-item"><span class="care-icon">▸</span><span>{item}</span></div>', unsafe_allow_html=True)

    if current_module == "care":
        with st.container():
            render_result_module_header(
                "module-care", "04", "🌿", "辨证施护与起居调摄推演",
                "从运动、施膳、情志与起居四个方面推演中医调护思路。",
            )
            care_theory = treatment.get("care_theory", [])
            icons = ['🏃', '🍽️', '🧘', '🌙']
            top_colors = ['#81c784', '#ffb74d', '#64b5f6', '#ba68c8']
            title_colors = ['#2e7d32', '#e65100', '#1565c0', '#6a1b9a']
            icon_bgs = [
                'linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%)',
                'linear-gradient(135deg,#fff5e6 0%,#ffe0b2 100%)',
                'linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%)',
                'linear-gradient(135deg,#f3e5f5 0%,#e1bee7 100%)',
            ]
            for row_start in range(0, len(care_theory), 2):
                cols = st.columns(2)
                for col_index, item_index in enumerate(
                    range(row_start, min(row_start + 2, len(care_theory)))
                ):
                    item = care_theory[item_index]
                    color_index = item_index % 4
                    icon = html.escape(str(item.get("icon", icons[color_index])))
                    title = html.escape(str(item.get("title", "调护推演")))
                    desc = html.escape(str(item.get("text", "")))
                    with cols[col_index]:
                        st.markdown(f"""
                        <div style="background:linear-gradient(180deg,#ffffff 0%,#fefefe 100%);
                                    padding:20px 20px 18px 20px;
                                    border-radius:12px;
                                    border-top:4px solid {top_colors[color_index]};
                                    height:100%;
                                    margin:6px 0 14px 0;
                                    box-shadow:0 3px 10px rgba(16,42,18,0.05), 0 1px 2px rgba(16,42,18,0.03);">
                          <div style="font-size:1.75rem;margin-bottom:8px;
                                      width:42px;height:42px;display:flex;align-items:center;justify-content:center;
                                      border-radius:10px;background:{icon_bgs[color_index]};">
                            {icon}
                          </div>
                          <div style="font-weight:750;color:{title_colors[color_index]};margin-bottom:8px;font-size:1rem;">{title}</div>
                          <div style="font-size:0.9rem;color:#4e342e;line-height:1.9;word-break:break-word;">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)

            reflection = treatment.get("reflection", {})
            if reflection:
                with st.container(border=True):
                    st.markdown('<div class="section-title">💭 课堂思考与教学提示</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="padding:16px 18px;background:linear-gradient(135deg,#f7fbf5 0%,#eef7ee 100%);'
                        'border-left:4px solid #66bb6a;border-radius:10px;color:#2f4f34;'
                        'font-size:0.92rem;line-height:1.9;margin-bottom:10px;">'
                        f'{html.escape(str(reflection.get("question", "")))}'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("查看教学提示"):
                        st.markdown(str(reflection.get("hint", "")))
                    st.caption("本互动仅用于课堂病机推演，不构成诊疗建议。")

    if current_module == "ai":
        with st.container():
            render_result_module_header(
                "module-ai", "05", "🧠", "AI 解读与报告",
                "结合中医理论解读病机，并可导出本病例教学分析报告。",
            )
            render_ai_interpretation(feature_inputs, probs, shap_dict, pred_s)

            ai_text = st.session_state.get("_first_interp_text") or ""
            if not ai_text:
                symptoms = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[:12]}
                tongue = {fn: feature_inputs[fn] for fn in FEATURE_NAMES[12:]}
                ai_text = generate_llm_interpretation(
                    symptoms=symptoms,
                    tongue_pulse=tongue,
                    probabilities=probs,
                    shap_dict=shap_dict,
                    feature_names=FEATURE_NAMES,
                    syndrome_labels=SYNDROME_LABELS,
                    use_llm=False,
                )["text"]

            report_text = build_teaching_report(
                pred_s=pred_s,
                probabilities=probs,
                feature_values=feature_inputs,
                shap_dict=shap_dict,
                ai_text=ai_text,
            )
            with st.container(border=True):
                st.markdown('<div class="section-title">📄 一键导出教学报告</div>', unsafe_allow_html=True)
                st.caption(
                    "TXT 报告包含当前四诊特征、六类证型概率、主证 SHAP Top 5、"
                    "AI / 本地标准解读和完整免责声明。"
                )
                report_bytes = ("\ufeff" + report_text).encode("utf-8")
                c_report_download, c_compare = st.columns(2)
                with c_report_download:
                    st.download_button(
                        "⬇️ 导出本病例教学分析报告（TXT）",
                        data=report_bytes,
                        file_name=f"症智明辨教学分析报告_{datetime.now():%Y%m%d_%H%M}.txt",
                        mime="text/plain; charset=utf-8",
                        width="stretch",
                        key="download_teaching_report",
                    )
                with c_compare:
                    st.button(
                        "生成阴虚与阳虚病案对比报告",
                        disabled=True,
                        help="V2.0 功能开发中",
                        width="stretch",
                        key="btn_compare_v2",
                    )
                    st.caption("病案对比教学功能将在 V2.0 开放。")

            st.markdown(
                """
                <div style="padding:16px 20px 16px 20px;
                            background:linear-gradient(135deg,#fff5f5 0%,#ffe9e9 100%);
                            border-radius:12px;
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
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.pop("scroll_to_module_top", False):
        _schedule_scroll_to_top()


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
    辅助辨证 · 辨证论治 · 方剂溯源<br>
    <span style="opacity:0.7;">仅供临床参考与学习使用</span>
    </div>""", unsafe_allow_html=True)

    # 主功能区（单页应用，无任何数据集信息页面）
    render_diagnosis()


if __name__ == "__main__":
    main()
