"""
llm_service.py —— 大模型 API 调用统一封装模块

设计原则：
  1. 密钥安全：只从 Streamlit secrets 读取 api_key / base_url / model，绝不明文硬编码。
  2. OpenAI 兼容：统一使用 openai SDK，通过 base_url 同时支持通义千问 / 讯飞星火 / 智谱 / DeepSeek / 本地 Ollama。
  3. 只读解读：大模型不做证型预测，只对 ML 模型已输出的 证型 + 概率 + SHAP 值 做临床解读。
  4. 可选增强：API 不可用或密钥缺失时，自动降级为规则化的本地解读文本，不阻塞核心页面。

------------------------------------------------------------------
部署配置（任选其一，推荐 secrets.toml）：

  方式 A —— 本地开发：在项目根目录创建 .streamlit/secrets.toml
      [llm]
      provider  = "qwen"                    # qwen | spark | zhipu | deepseek | ollama | openai
      api_key  = "sk-xxxxxxxxxxxxxxxxxxxx"  # 本地占位，请勿提交到仓库
      base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
      model    = "qwen-plus"
      temperature = 0.3

  方式 B —— Streamlit Community Cloud：部署平台设置 secrets
      左侧菜单 → Advanced settings → Secrets → 填入同名 TOML 段落即可。

  方式 C —— 环境变量（无 Streamlit 时可直接用 os.environ）：
      LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_PROVIDER
------------------------------------------------------------------
"""

from __future__ import annotations

import os
import json
import logging
import socket
from typing import Optional
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# 1. 配置读取（st.secrets → os.environ → 默认值）
# ======================================================================
def _read_secrets():
    """从 Streamlit secrets 读取 [llm] 段；若没有 secrets 文件返回空 dict。"""
    try:
        import streamlit as st
        sec = st.secrets.get("llm", None)
        # st.secrets 返回的是 Secrets 对象，.get 可能也返回 Secrets，转成纯 dict
        if sec is not None and hasattr(sec, "items"):
            return {k: v for k, v in sec.items()}
    except Exception as _e:
        logger.warning("_read_secrets failed: %s: %s", type(_e).__name__, _e)
    return {}


# 预设 provider → base_url / model 映射（仅作为默认提示，真正生效以 secrets 为准）
_PROVIDER_PRESETS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "spark": {
        "base_url": "https://spark-api-open.xfyun.cn/v1",
        "model": "generalv3.5",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}


def get_llm_config() -> dict:
    """
    返回统一配置 dict：{api_key, base_url, model, temperature, provider, configured}

    configured=False 表示既没 secrets 也没环境变量，可以安全跳过 API 调用走降级分支。
    """
    sec = _read_secrets()

    api_key  = (sec.get("api_key")
                or os.environ.get("LLM_API_KEY")
                or "").strip()

    provider = (sec.get("provider")
                or os.environ.get("LLM_PROVIDER")
                or "qwen").strip().lower()

    base_url = (sec.get("base_url")
                or os.environ.get("LLM_BASE_URL")
                or _PROVIDER_PRESETS.get(provider, {}).get("base_url", "")).strip()

    model    = (sec.get("model")
                or os.environ.get("LLM_MODEL")
                or _PROVIDER_PRESETS.get(provider, {}).get("model", "")).strip()

    temperature = float(sec.get("temperature") if sec.get("temperature") is not None
                        else os.environ.get("LLM_TEMPERATURE", 0.3))

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "provider": provider,
        "configured": bool(api_key and base_url and model),
    }


# ======================================================================
# 2. Prompt 组装 —— 明确禁止 LLM 重新预测证型
# ======================================================================
SYSTEM_PROMPT = """\
【角色定位】
你是一名熟悉《中医基础理论》《中医诊断学》《中医内科学》的中医教研助手，服务于 2 型糖尿病中医证型的科研教学场景。
证型预测由已训练好的随机森林 / LightGBM(EBM) 模型完成，你只解释模型结果和中医理论，不重新预测证型。

【安全硬红线（不可逾越）】
1. 不得重新预测证型，不得推翻机器学习模型给出的主证型结论。
2. 不得开具处方，不得推荐具体药物，不得输出任何药物剂量、克数、用量或煎服法。
3. 不得输出“确诊”“明确诊断”“建议立即就医”等临床决策性表述。
4. 本解读不能替代执业中医师四诊合参和辨证论治。

【理论约束（必须执行）】
1. 必须以《中医基础理论》《中医诊断学》《中医内科学》的证候、病机、舌脉理论解释本病例。
2. 第 ①②③ 段每一段都必须以“根据中医理论，”作为开头，先讲病机与四诊依据，再解释模型线索。
3. 不得单纯罗列 SHAP 数值。SHAP 最多作为辅助证据，不要连续出现多个 ± 小数；应把它们翻译成“哪些症状支持主证、哪些症状提示兼证或边界”。
4. 必须结合患者当前已有症状与舌脉，不得只复述模型概率。

【固定四段结构（每段必须有小标题）】
① SHAP 通俗翻译与四诊证据
   说明模型为何倾向当前主证；把正向、负向 SHAP 翻译为支持证据与存疑证据，并联系症状和舌脉。

② 本证型病机与相似证鉴别
   依据教材理论说明核心病机、病位、虚实寒热和典型舌脉；与第二候选证型作 2~3 点鉴别；解释次证概率提示兼证还是模型边界。

③ 饮食、起居与情志调摄
   给出不含药物、不含剂量的生活调摄建议，并按本证型病机解释其依据。

④ 免责声明
   ⚠️ 本内容仅供科研教学参考，不构成任何诊疗依据。
   具体诊断与治疗须由执业中医师四诊合参后确定，切勿自行判断、照方抓药或延误就医。

【文风与篇幅】
全文中文，500~900 字。术语首次出现时附一句通俗解释。不要写空泛套话，不要出现药物名称或剂量。
"""


def assemble_full_shap_context(
    shap_dict: dict,
    feature_names: list,
    syndrome_labels: list,
    max_features: int = 15,
) -> str:
    """
    把全部 SHAP 贡献格式化成紧凑文本：15 特征 × 6 证型 = 90 项，
    每项带 ↑/↓ 表示方向，便于大模型完整感知特征-证型映射。

    格式紧凑，控制在 ~800 字符内，避免 token 爆炸。
    """
    feat_names = feature_names[:max_features]
    lines = []
    lines.append("【全量 SHAP 贡献矩阵 —— 特征 × 证型（共 %d 项）】" % (len(feat_names) * len(syndrome_labels)))
    lines.append("符号说明：↑=正向（支持）  ↓=负向（不支持）  =0 表示该特征对该证型几乎无贡献")

    for feat_idx, feat_name in enumerate(feat_names):
        row_parts = []
        for syn in syndrome_labels:
            arr = np.asarray(shap_dict.get(syn, np.zeros(len(feat_names)))).flatten()
            v = float(arr[feat_idx]) if feat_idx < len(arr) else 0.0
            sign = "↑" if v > 0.02 else ("↓" if v < -0.02 else "·")
            row_parts.append(f"{syn}{sign}{v:+.4f}")
        lines.append(f"  · {feat_name}:  " + "  ".join(row_parts))

    return "\n".join(lines)


def assemble_user_prompt(
    symptoms: dict,           # {特征名: 0-1 浮点数}
    tongue_pulse: dict,       # 同上（苔黄腻 / 舌红 / 脉细）
    probabilities: dict,      # {证型名: 概率 0-1}
    shap_dict: dict,          # {证型名: np.ndarray(15,)}
    feature_names: list,      # 15 项特征名
    syndrome_labels: list,    # 6 个证型名
) -> str:
    """把 ML 模型全部输出格式化为一段自然语言，作为 user message。"""

    # --- 1) 患者症状（只列程度 >0.3 的，避免过长） ---
    symptom_lines = []
    for name in feature_names[:12]:  # 前 12 项是临床症状
        v = float(symptoms.get(name, 0.0))
        if v > 0.3:
            # 把 0~1 的浮点映射为轻/中/重
            level = "重度" if v >= 0.75 else ("中度" if v >= 0.55 else "轻度")
            symptom_lines.append(f"    • {name}：{level}（特征值 {v:.2f}）")

    tongue_lines = []
    for name in feature_names[12:]:  # 后 3 项是舌脉
        v = float(tongue_pulse.get(name, 0.0))
        if v > 0.3:
            level = "明显" if v >= 0.75 else ("较明显" if v >= 0.55 else "轻度")
            tongue_lines.append(f"    • {name}：{level}（特征值 {v:.2f}）")

    # --- 2) 六类证型预测概率 ---
    sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    prob_lines = []
    for syn, p in sorted_probs:
        prob_lines.append(f"    • {syn}：{p:.1%}")

    # --- 3) 主证型 SHAP 关键特征（Top-5 正 & Top-3 负） ---
    main_syn = sorted_probs[0][0]
    shap_arr = np.asarray(shap_dict.get(main_syn, np.zeros(len(feature_names)))).flatten()
    if len(shap_arr) < len(feature_names):
        shap_arr = np.pad(shap_arr, (0, len(feature_names) - len(shap_arr)))

    pos_idx = np.argsort(-shap_arr)[:5]   # 正向最大 5
    neg_idx = np.argsort(shap_arr)[:3]    # 负向最大 3

    shap_lines = []
    shap_lines.append(f"  主证型【{main_syn}】的 SHAP 关键特征：")
    shap_lines.append("    正向贡献（支持该证型）：")
    for i in pos_idx:
        if shap_arr[i] > 0:
            shap_lines.append(f"      • {feature_names[int(i)]}：SHAP = {shap_arr[i]:+.4f}")
    shap_lines.append("    负向贡献（不支持该证型）：")
    for i in neg_idx:
        if shap_arr[i] < 0:
            shap_lines.append(f"      • {feature_names[int(i)]}：SHAP = {shap_arr[i]:+.4f}")

    full_shap_ctx = assemble_full_shap_context(shap_dict, feature_names, syndrome_labels)

    return (
        "请基于以下机器学习模型输出，对 2 型糖尿病患者的中医证型进行科普解读。\n\n"
        "【机器学习模型来源】本预测来自 **随机森林 / LightGBM(EBM)** 模型，"
        "在 2 型糖尿病中医证型数据集上训练，可区分："
        + "、".join(syndrome_labels)
        + "。你是科普解读角色，不做二次诊断。\n\n"
        "【患者主要症状（程度中等以上）】\n"
        + ("\n".join(symptom_lines) if symptom_lines else "    （无明显中重度症状）")
        + "\n\n【舌脉表现】\n"
        + ("\n".join(tongue_lines) if tongue_lines else "    （无明显异常）")
        + "\n\n【六类中医证型预测概率（机器学习模型输出）】\n"
        + "\n".join(prob_lines)
        + "\n\n【主证型关键 SHAP 特征 Top-5 正 + Top-3 负】\n"
        + "\n".join(shap_lines)
        + "\n\n" + full_shap_ctx
    )


# ======================================================================
# 3. 安全与异常处理（模块4）
# ======================================================================

# ---- 3a. API 异常分类 → 友好提示 ----
def _classify_api_error(e: Exception) -> str:
    """
    把 openai SDK / 网络层抛出的异常分类，返回面向终端用户的友好提示。
    核心原则：无论哪种 API 异常，ML 辨证 + SHAP 功能不受影响。
    """
    err_str = str(e).lower()

    # 网络类：连接超时 / DNS 解析失败 / 断网
    if any(k in err_str for k in ("connection", "timeout", "timed out",
                                   "dns", "resolve", "refused", "unreachable")):
        return "大模型服务暂时不可用（网络连接异常），请稍后重试。"

    # 额度耗尽 / 限流
    if any(k in err_str for k in ("429", "quota", "rate limit", "insufficient_quota",
                                   "rate_limit", "too many requests")):
        return "大模型服务额度已耗尽或请求过于频繁，请稍后重试或联系管理员。"

    # 认证失败
    if any(k in err_str for k in ("401", "authentication", "invalid api key",
                                   "unauthorized", "invalid_api_key")):
        return "大模型 API 密钥无效，请联系管理员检查配置。"

    # 服务端错误
    if any(k in err_str for k in ("500", "502", "503", "server error",
                                   "internal error", "unavailable")):
        return "大模型服务端暂时不可用，请稍后重试。"

    # 兜底
    return "大模型调用异常，已自动切换为规则化降级解读，核心辨证功能不受影响。"


# ---- 3b. 危险提问过滤 ----
_DANGER_KEYWORDS = [
    "开药方", "开方", "处方", "给我开", "帮我开",
    "剂量", "用量", "多少克", "几克", "克数",
    "治疗方案", "怎么治", "如何治疗", "怎么治好",
    "吃什么药", "用什么药", "推荐药物", "具体用药",
    "帮我治病", "给我治病", "给我开药",
]

_DANGER_REPLY = (
    "⚠️ **抱歉，我无法回答此类问题。**\n\n"
    "根据安全规范，我不能开具处方、推荐具体药物或剂量、提供治疗方案。\n"
    "以上内容须由**执业中医师**四诊合参后确定。\n\n"
    "> 本系统仅对机器学习模型已输出的证型、概率与 SHAP 贡献做科普解读，"
    "> 不构成任何诊疗依据。具体诊断与治疗请务必咨询执业中医师。"
)


def filter_dangerous_question(user_question: str) -> bool:
    """
    检查用户提问是否涉及开方/剂量/治疗方案等危险内容。
    返回 True 表示该问题被拦截（调用方应直接返回 _DANGER_REPLY）。
    """
    q = user_question.strip().lower()
    for kw in _DANGER_KEYWORDS:
        if kw in q:
            return True
    return False


def get_danger_reply() -> str:
    """返回危险提问的标准拒绝回复（供 app.py 调用）。"""
    return _DANGER_REPLY


# ======================================================================
# 4. 统一调用入口（单轮 / 多轮共用）
# ======================================================================
def call_llm_chat(messages: list, config: Optional[dict] = None) -> str:
    """
    最底层：直接把 messages 数组交给 OpenAI 兼容端点。
    支持多轮对话 —— 调用方自己拼好 system / history / current user。
    """
    if config is None:
        config = get_llm_config()
    if not config.get("configured"):
        raise RuntimeError("LLM 未配置：缺少 api_key / base_url / model。请在 .streamlit/secrets.toml 中填写。")

    try:
        import openai as _openai
        OpenAI = _openai.OpenAI
    except ImportError as e:
        raise RuntimeError("缺少 openai SDK，请先运行：pip install openai") from e

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=20.0,
        max_retries=0,
    )
    try:
        resp = client.chat.completions.create(
            model=config["model"],
            temperature=config["temperature"],
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        logger.warning("LLM chat 调用失败（provider=%s, model=%s）：%s",
                       config.get("provider"), config.get("model"), e)
        raise RuntimeError(f"大模型 API 调用失败：{e}") from e


def call_llm(system_prompt: str, user_prompt: str, config: Optional[dict] = None) -> str:
    """单轮快捷封装：等价于 call_llm_chat([{system},{user}])。保留给 generate_llm_interpretation 使用。"""
    return call_llm_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config=config,
    )


# ======================================================================
# 4. 本地 Mock / 规则化降级（不依赖外部 API）
# ======================================================================
_SYNDROME_THEORY = {
    "气阴两虚": {
        "pathogenesis": "气阴两耗，脾气亏虚则运化无力，肺胃阴虚则津液不能输布，久则累及肾元。",
        "differentiation": "重点与阴虚热盛鉴别：前者多见神疲乏力、自汗、脉细弱；后者多见五心烦热、舌红少津、脉细数。",
        "care": "饮食宜清淡甘润，主食粗细搭配；活动以八段锦、太极拳等柔和运动为主，避免过劳和熬夜。",
    },
    "痰热互结": {
        "pathogenesis": "痰湿内蕴，郁久化热，痰热阻滞中焦，气机升降失司，故见口苦、苔黄腻及脘腹不适。",
        "differentiation": "重点与肝胃郁热鉴别：痰热证以形体肥胖、苔黄腻、胸闷痰多为要；肝胃郁热证以烦躁易怒、胁胀反酸为要。",
        "care": "饮食宜清淡，少食肥甘甜腻，主食粗细搭配；居住环境宜干爽，循序渐进增加活动量。",
    },
    "肝肾阴虚": {
        "pathogenesis": "肝肾阴液不足，精血不能上荣目窍、下养腰膝，虚热内生，故见视物模糊、腰膝酸软、舌红脉细。",
        "differentiation": "重点与气阴两虚鉴别：肝肾阴虚以腰膝酸软、目涩耳鸣、舌红少苔为主；气阴两虚以倦怠自汗、气短为主。",
        "care": "注意用眼休息，避免熬夜和过度劳神；饮食宜清润，保持情绪平和，运动以舒缓有氧活动为主。",
    },
    "热盛伤津": {
        "pathogenesis": "燥热炽盛，灼伤肺胃津液，胃热腐熟太过则多食易饥，热结肠道则便干，热盛津伤则口渴明显。",
        "differentiation": "重点与肝胃郁热鉴别：热盛伤津以口渴、多食、便结、舌红苔燥为主；肝胃郁热以情志不畅、口苦胁胀、脉弦为要。",
        "care": "避免辛辣煎炸和温燥食物，适量饮水；运动宜避开高温时段，避免大汗进一步耗伤津液。",
    },
    "肝胃郁热": {
        "pathogenesis": "肝气郁结化火，横逆犯胃，肝胃同病；肝火上扰则心烦易怒、口苦，胃热炽盛则多食易饥、便结。",
        "differentiation": "重点与痰热互结鉴别：肝胃郁热以情绪波动、口苦反酸、胁胀为要；痰热互结以肥胖、苔黄腻、胸闷痰多为要。",
        "care": "重视情志调畅，避免暴怒和长期压抑；饮食定时适量，少食辛辣油腻，可用舒缓运动帮助疏解压力。",
    },
    "阴阳两虚": {
        "pathogenesis": "病久阴损及阳，阴阳俱虚，命门温煦不足则畏寒肢冷，水湿不化则下肢浮肿，肾失固摄则夜尿增多。",
        "differentiation": "重点与肝肾阴虚鉴别：阴阳两虚兼见畏寒肢冷、浮肿、便溏；肝肾阴虚以虚热、目涩、舌红少苔为主。",
        "care": "注意腰腹和足部保暖，避免生冷饮食；活动宜温和，避免大汗耗气，出现明显水肿时应及时接受专业评估。",
    },
}


def _active_feature_names(symptoms: dict, tongue_pulse: dict, feature_names: list) -> list:
    """按当前输入提取阳性特征，保留原有特征顺序。"""
    merged = {**(symptoms or {}), **(tongue_pulse or {})}
    return [
        name for name in feature_names
        if float(merged.get(name, 0.0)) >= 0.3
    ]


def rule_based_interpretation(
    probabilities: dict,
    shap_dict: dict,
    feature_names: list,
    symptoms: Optional[dict] = None,
    tongue_pulse: Optional[dict] = None,
) -> str:
    """本地 Mock：按教材病机生成四段式标准解读，API 不可用时立即返回。"""
    sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    main_syn, main_prob = sorted_probs[0]
    sec_syn, sec_prob = sorted_probs[1] if len(sorted_probs) > 1 else ("—", 0.0)
    theory = _SYNDROME_THEORY.get(main_syn, {})

    shap_arr = np.asarray(shap_dict.get(main_syn, np.zeros(len(feature_names)))).flatten()
    if len(shap_arr) < len(feature_names):
        shap_arr = np.pad(shap_arr, (0, len(feature_names) - len(shap_arr)))
    pos_idx = [int(i) for i in np.argsort(-shap_arr)[:5] if shap_arr[i] > 0]
    neg_idx = [int(i) for i in np.argsort(shap_arr)[:3] if shap_arr[i] < 0]
    positive = [feature_names[i] for i in pos_idx]
    negative = [feature_names[i] for i in neg_idx]

    active = _active_feature_names(symptoms or {}, tongue_pulse or {}, feature_names)
    active_text = "、".join(active[:8]) if active else "当前阳性四诊特征较少"
    positive_text = "、".join(positive[:4]) if positive else "多项综合特征"
    negative_text = "、".join(negative[:3]) if negative else "未见明显反向证据"

    if main_prob >= 0.60:
        confidence = "较高"
    elif main_prob >= 0.30:
        confidence = "中等，仍需结合兼证与四诊合参"
    else:
        confidence = "偏低，应把结果作为线索而非结论"

    lines = [
        "### 一、SHAP 通俗翻译与四诊证据",
        f"根据中医理论，本病例采集到的阳性表现为{active_text}。机器学习模型据此倾向"
        f"**{main_syn}证**，主证匹配度为 {main_prob:.1%}，判断把握{confidence}。",
        f"从可解释性结果看，{positive_text}对主证形成支持；{negative_text}"
        "则提示仍需辨析兼夹或模型边界。这里不逐一罗列贡献小数，而是把 SHAP 理解为："
        "哪些四诊信息在推动模型作出当前判断。",
        "",
        "### 二、本证型病机与相似证鉴别",
        f"根据中医理论，{main_syn}证的核心病机可概括为：{theory.get('pathogenesis', '证候由气血阴阳失调、脏腑功能失常所致。')}",
        f"本病例第二候选为**{sec_syn}证**（{sec_prob:.1%}）。{theory.get('differentiation', '两种证候可能兼夹，应结合舌脉和症状轻重进一步辨析。')}"
        "次证概率存在并不等于同时确诊两种证型，更可能提示兼证、病程阶段变化或模型对边界样本的不确定性。",
        "",
        "### 三、饮食、起居与情志调摄",
        f"根据中医理论，{main_syn}证的调摄应围绕其病机展开。{theory.get('care', '饮食宜清淡有节，起居规律，避免过劳。')}"
        "建议观察症状变化并记录血糖监测结果；若出现明显不适或指标异常，应由专业人员评估。",
        "",
        "### 四、免责声明",
        "⚠️ **本内容仅供科研教学参考，不构成任何诊疗依据。**",
        "具体诊断与治疗须由执业中医师四诊合参后确定，切勿自行判断、照方抓药或延误就医。",
    ]
    return "\n".join(lines)


def _api_endpoint_reachable(base_url: str, timeout: float = 0.75) -> bool:
    """调用前快速探测端点，断网时立即切换到本地 Mock，避免路演长时间等待。"""
    try:
        parsed = urlparse(base_url)
        if not parsed.hostname:
            return True
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


# ======================================================================
# 5. 对外主入口（供 app.py 调用）
# ======================================================================
def generate_llm_interpretation(
    symptoms: dict,
    tongue_pulse: dict,
    probabilities: dict,
    shap_dict: dict,
    feature_names: list,
    syndrome_labels: list,
    use_llm: bool = True,
) -> dict:
    """
    生成一段 AI 解读文本。

    Returns:
        dict with keys:
            text       : str   最终可直接 Markdown 渲染的解读文本
            source     : str   "llm" | "rule"   —— 便于前端显示徽章
            configured : bool  当前是否配置了 LLM
            error      : str|None  LLM 调用失败时的错误摘要（用于日志）
    """
    config = get_llm_config()

    # ---- 未启用、未配置或端点不可达 → 直接走本地四段式 Mock ----
    if not use_llm or not config.get("configured"):
        return {
            "text": rule_based_interpretation(
                probabilities, shap_dict, feature_names, symptoms, tongue_pulse
            ),
            "source": "mock",
            "configured": config.get("configured", False),
            "error": None,
        }

    if not _api_endpoint_reachable(config.get("base_url", "")):
        return {
            "text": rule_based_interpretation(
                probabilities, shap_dict, feature_names, symptoms, tongue_pulse
            ),
            "source": "mock",
            "configured": True,
            "error": "网络连接异常，已自动切换到本地标准解读。",
        }

    try:
        user_prompt = assemble_user_prompt(
            symptoms=symptoms,
            tongue_pulse=tongue_pulse,
            probabilities=probabilities,
            shap_dict=shap_dict,
            feature_names=feature_names,
            syndrome_labels=syndrome_labels,
        )
        text = call_llm(SYSTEM_PROMPT, user_prompt, config)
        return {
            "text": text,
            "source": "llm",
            "configured": True,
            "error": None,
        }
    except Exception as e:
        logger.exception("LLM 调用异常，回退到本地标准解读")
        friendly = _classify_api_error(e)
        mock_text = rule_based_interpretation(
            probabilities, shap_dict, feature_names, symptoms, tongue_pulse
        )
        return {
            "text": (
                f"> 系统提示：{friendly} 已自动切换到本地标准解读。\n\n"
                + mock_text
            ),
            "source": "mock",
            "configured": True,
            "error": friendly,
        }


# ======================================================================
# 6. 聊天模块（模块2 对外入口）
# ======================================================================
CHAT_SYSTEM_BASE = """\
【角色定位】
你是一名熟悉《中医基础理论》《中医诊断学》《中医内科学》的中医教研助手，正在与 2 型糖尿病患者（或医学生）进行多轮对话。
本对话中的证型预测来自**随机森林 + LightGBM(EBM) 机器学习模型**，
模型基于 15 项临床特征对六类中医证型做多分类，**你只做科普解读，不做预测**。

【硬红线（不可逾越）】
  ✋ 不得重新独立预测证型、不得推翻机器学习模型给出的主证型结论；
  ✋ 不得开具任何具体处方、剂量、药物名称；
  ✋ 不得输出"确诊""明确诊断""建议立即就医"等临床决策性词汇；
  ✋ 不得声称本解读可替代执业中医师的四诊合参与辨证论治。

【输出要求（每次回答都要遵守）】
  1) 每次回答必须以“根据中医理论，”开头，并依据《中医基础理论》《中医诊断学》或《中医内科学》解释病机；
  2) 回答必须**结合已提供的本患者症状、舌脉、证型概率与 SHAP 线索**，不得空泛套话，也不得单纯罗列 SHAP 数值；
  3) SHAP 相关问题：↑ = 正向（支持该证型），↓ = 负向（不支持该证型），应翻译成四诊证据，而非连续报数；
  4) 每次回答末尾**必须出现免责声明**（可简写一行）：
     ⚠️ 本内容仅供科研教学参考，不构成任何诊疗依据；
  5) 全文中文，150~400 字，必要时分条，面向普通患者+医学生；
  6) 追问超出中医科普范围（如生活琐事、无关学科）时礼貌引导回到中医证型相关问题。
"""


def build_chat_system_prompt(
    symptoms: dict,
    tongue_pulse: dict,
    probabilities: dict,
    shap_dict: dict,
    feature_names: list,
    syndrome_labels: list,
) -> str:
    """
    把患者结构化数据 + ML 模型输出编码进 system prompt，
    这样后续多轮追问不需要再重复传结构化数据。
    包含：症状 / 舌脉 / 六类证型概率 / 主证型 Top-5+Top-3 SHAP / 全量 15×6 SHAP 矩阵。
    """
    sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    main_syn = sorted_probs[0][0]

    # 症状（>0.3 的）
    sym_lines = []
    for name in feature_names[:12]:
        v = float(symptoms.get(name, 0.0))
        if v > 0.3:
            level = "重度" if v >= 0.75 else ("中度" if v >= 0.55 else "轻度")
            sym_lines.append(f"  - {name}：{level}（{v:.2f}）")

    # 舌脉
    tp_lines = []
    for name in feature_names[12:]:
        v = float(tongue_pulse.get(name, 0.0))
        if v > 0.3:
            level = "明显" if v >= 0.75 else ("较明显" if v >= 0.55 else "轻度")
            tp_lines.append(f"  - {name}：{level}（{v:.2f}）")

    # 主证型 Top-5 正向 + Top-3 负向
    shap_arr = np.asarray(shap_dict.get(main_syn, np.zeros(len(feature_names)))).flatten()
    if len(shap_arr) < len(feature_names):
        shap_arr = np.pad(shap_arr, (0, len(feature_names) - len(shap_arr)))
    pos_idx = np.argsort(-shap_arr)[:5]
    neg_idx = np.argsort(shap_arr)[:3]
    top_shap_lines = ["  正向（支持）："]
    top_shap_lines += [f"    ↑ {feature_names[int(i)]}：{shap_arr[i]:+.4f}"
                       for i in pos_idx if shap_arr[i] > 0]
    top_shap_lines.append("  负向（不支持）：")
    top_shap_lines += [f"    ↓ {feature_names[int(i)]}：{shap_arr[i]:+.4f}"
                       for i in neg_idx if shap_arr[i] < 0]

    # 概率全表
    prob_lines = [f"  - {s}：{p:.1%}" for s, p in sorted_probs]

    # 全量 SHAP（15×6=90 项）
    full_shap_ctx = assemble_full_shap_context(shap_dict, feature_names, syndrome_labels)

    return (
        CHAT_SYSTEM_BASE
        + "\n\n【本患者结构化数据 —— 随机森林/EBM 机器学习模型输出（15 项特征 → 6 类证型），仅供回答追问参考】\n"
        + f"• ML 主证型判定：{main_syn}（置信度 {sorted_probs[0][1]:.1%}）\n"
        + "• 六类证型概率分布：\n" + "\n".join(prob_lines) + "\n"
        + "• 患者主要症状（中等以上）：\n" + ("\n".join(sym_lines) if sym_lines else "  （无明显中重度症状）") + "\n"
        + "• 舌脉表现：\n" + ("\n".join(tp_lines) if tp_lines else "  （无明显异常）") + "\n"
        + f"• 主证型【{main_syn}】Top SHAP 关键特征：\n"
        + "\n".join(top_shap_lines) + "\n"
        + full_shap_ctx
    )


def chat_with_patient_data(
    user_question: str,
    history: list,                # [{role, content}, ...]  仅 user / assistant
    system_prompt: str,           # 已编码患者数据的 system prompt
    config: Optional[dict] = None,
    max_history_turns: int = 10,
) -> str:
    """
    多轮对话主入口。把 system + 截断后的 history + 当前问题 组装成 messages，交给 API。

    安全机制：
      1. 先过滤危险提问（开方/剂量/治疗方案）→ 直接返回标准拒绝回复；
      2. API 调用失败时抛出 RuntimeError，携带友好提示，由调用方降级处理。

    history 自动截断：只保留最近 max_history_turns 轮，避免 prompt 膨胀。
    同时过滤掉 role 非法 / content 为空的脏条目。
    """
    # ---- 安全过滤：拦截危险提问 ----
    if filter_dangerous_question(user_question):
        return _DANGER_REPLY

    if config is None:
        config = get_llm_config()

    if not config.get("configured"):
        raise RuntimeError("LLM 未配置，请在 .streamlit/secrets.toml 中填写 api_key / base_url / model。")
    if not _api_endpoint_reachable(config.get("base_url", "")):
        raise RuntimeError("网络连接异常，已切换到本地标准解读。")

    # --- 拼装 messages ---
    valid_history = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            valid_history.append({"role": role, "content": content.strip()})

    # 保留最近 max_history_turns 轮（一轮 = user + assistant，所以乘 2）
    # 这里简化为：最后 N 条消息（无论角色）
    trimmed = valid_history[-(max_history_turns * 2):] if len(valid_history) > max_history_turns * 2 else valid_history

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_question.strip()})

    try:
        return call_llm_chat(messages, config)
    except Exception as e:
        # 把原始异常包装成带友好提示的 RuntimeError，供 app.py catch 后降级
        raise RuntimeError(_classify_api_error(e)) from e


# ----------------------------------------------------------------------
# 规则化聊天降级（configured=False 时用，能力有限但保证交互不崩）
# ----------------------------------------------------------------------
_FOLLOWUP_TEMPLATES = {
    "鉴别": (
        "根据中医理论，关于「{q}」，中医辨析证候强调四诊合参、病机互参。"
        "气阴两虚 vs 阴虚热盛的常见鉴别点包括：\n"
        "• 舌象：气阴两虚多见舌淡少苔，阴虚热盛多见舌红少苔或无苔；\n"
        "• 脉象：气阴两虚多偏细弱，阴虚热盛多偏细数；\n"
        "• 畏寒与否：气阴两虚常伴畏寒肢冷，阴虚热盛常伴五心烦热。\n"
        "以上是通用鉴别思路，具体仍需结合患者的其他症状综合判断。"
    ),
    "饮食": (
        "根据中医理论，饮食调摄应结合证候虚实寒热，以下原则可作为教学参考：\n"
        "• 控制总热量，主食定量，杂粮占一定比例；\n"
        "• 低 GI 食物：燕麦、藜麦、糙米、绿叶蔬菜优先；\n"
        "• 避免精制糖、油炸和高盐食物，酒和甜饮尽量不碰；\n"
        "• 适量饮水，根据口渴、汗出和活动量调整；\n"
        "• 气阴两虚者可适度吃山药、莲子、枸杞；阴虚热盛者可吃银耳、百合、冬瓜。"
    ),
    "运动": (
        "根据中医理论，动静有度、劳逸结合有助于调畅气血，糖尿病患者可参考：\n"
        "• 中等强度有氧运动：快走、慢跑、游泳、骑车，每周 150 分钟以上；\n"
        "• 餐后 30~60 分钟适度散步，有助于控制餐后血糖；\n"
        "• 阴虚或气阴不足者避免大汗淋漓，阳虚者宜注意保暖；\n"
        "• 合并心血管疾病时先咨询医生再运动。"
    ),
    "默认": (
        "根据中医理论，当前处于本地标准解读模式，未接入大模型 API，只能回答有限范围的问题。\n"
        "你问的「{q}」不在预置答案范围内。如需更丰富的个性化回答，"
        "请在 `.streamlit/secrets.toml` 中配置 LLM（DeepSeek/通义千问/讯飞星火等均可）。"
    ),
}

_LOCAL_CHAT_DISCLAIMER = (
    "\n\n⚠️ 本内容仅供科研教学参考，不构成任何诊疗依据。"
)


def fallback_chat_reply(user_question: str) -> str:
    """未配置 LLM 时的规则化聊天降级：按关键词匹配模板。"""
    # 安全过滤：危险提问统一拒绝
    if filter_dangerous_question(user_question):
        return _DANGER_REPLY
    q = user_question
    for key, tmpl in _FOLLOWUP_TEMPLATES.items():
        if key == "默认":
            continue
        if key in q:
            return tmpl.format(q=q[:40]) + _LOCAL_CHAT_DISCLAIMER
    return _FOLLOWUP_TEMPLATES["默认"].format(q=q[:40]) + _LOCAL_CHAT_DISCLAIMER
