"""
配置和工具函数 - 更新为6个证型和15个症状特征
"""
import streamlit as st

# 系统信息
SYSTEM_INFO = {
    'name': '症智明辨',
    'version': '2.0.0',
    'description': '基于机器学习与可解释性 AI 的 2 型糖尿病中医证型智能辅助辨证系统',
    'author': '大数据与人工智能创意赛参赛团队',
}

# 证型详细信息 - 6个证型（基于dataset_final_6class.csv）
SYNDROME_INFO = {
    '气阴两虚': {
        'color': '#FF6B6B',
        'icon': '🌸',
        'keywords': ['口渴', '神疲乏力', '腰膝酸软', '视物模糊', '舌红', '脉细'],
        'treatment': '益气养阴',
        'formula': '生脉散合六味地黄丸加减',
        'description': '气阴两虚证是糖尿病最常见证型，表现为气虚与阴虚并存'
    },
    '痰热互结': {
        'color': '#4ECDC4',
        'icon': '🌿',
        'keywords': ['口渴', '口苦', '苔黄腻', '形体肥胖', '心烦易怒', '大便干结'],
        'treatment': '清热化痰，散结和中',
        'formula': '黄连温胆汤合小陷胸汤加减',
        'description': '痰热互结证多见于肥胖型糖尿病患者，痰热交阻中焦'
    },
    '肝肾阴虚': {
        'color': '#45B7D1',
        'icon': '💧',
        'keywords': ['腰膝酸软', '视物模糊', '舌红', '口渴', '脉细', '形体肥胖'],
        'treatment': '滋补肝肾，养阴清热',
        'formula': '杞菊地黄丸合一贯煎加减',
        'description': '肝肾阴虚证多见于糖尿病病程较长者，肝肾阴液亏虚'
    },
    '热盛伤津': {
        'color': '#FFA07A',
        'icon': '🔥',
        'keywords': ['口渴', '多食易饥', '口苦', '小便色黄', '大便干结', '舌红'],
        'treatment': '清热泻火，养阴生津',
        'formula': '消渴方合白虎加人参汤加减',
        'description': '热盛伤津证多见于糖尿病早中期，燥热炽盛耗伤津液'
    },
    '肝胃郁热': {
        'color': '#9370DB',
        'icon': '⚡',
        'keywords': ['心烦易怒', '口苦', '口渴', '多食易饥', '大便干结', '舌红'],
        'treatment': '清肝泻胃，理气开郁',
        'formula': '大柴胡汤合左金丸加减',
        'description': '肝胃郁热证多见于情绪不畅患者，肝郁化火犯胃'
    },
    '阴阳两虚': {
        'color': '#6B8E23',
        'icon': '☯️',
        'keywords': ['畏寒肢冷', '下肢浮肿', '腰膝酸软', '视物模糊', '神疲乏力', '脉细'],
        'treatment': '温阳滋阴，补肾固摄',
        'formula': '金匮肾气丸合水陆二仙丹加减',
        'description': '阴阳两虚证多见于糖尿病后期，阴损及阳，阴阳俱虚'
    }
}

# 特征分组 - 基于dataset_final_6class.csv的15个症状特征
FEATURE_GROUPS = {
    '临床症状': ['口渴', '多食易饥', '神疲乏力', '心烦易怒', '口苦',
                '小便色黄', '大便干结', '形体肥胖', '腰膝酸软', '视物模糊',
                '畏寒肢冷', '下肢浮肿'],
    '舌脉特征': ['苔黄腻', '舌红', '脉细']
}

# 特征的中医属性（用于滑块标签与提示，突出四诊含义）
FEATURE_ATTRIBUTES = {
    '口渴': '主津伤燥热',
    '多食易饥': '主胃火炽盛',
    '神疲乏力': '主气虚',
    '心烦易怒': '主肝郁化火',
    '口苦': '主胆热湿热',
    '小便色黄': '主下焦热盛',
    '大便干结': '主肠燥腑实',
    '形体肥胖': '主痰湿内盛',
    '腰膝酸软': '主肾精亏虚',
    '视物模糊': '主肝肾阴虚',
    '畏寒肢冷': '主阳气亏虚',
    '下肢浮肿': '主脾肾阳虚',
    '苔黄腻': '主痰湿湿热',
    '舌红': '主热证阴虚',
    '脉细': '主阴血亏虚',
}

# 特征描述 —— 按《中医诊断学》四诊要点撰写，供滑块悬浮提示使用
FEATURE_DESCRIPTIONS = {
    '口渴': '口中干燥、渴欲饮水。多因津液亏损或热盛伤津所致，为消渴病上消常见症状。',
    '多食易饥': '食欲亢进、食后易饥。属胃火炽盛、腐熟太过，常见于消渴病中消。',
    '神疲乏力': '精神倦怠、肢体无力。多由脾气亏虚、清阳不升、四肢失养所致。',
    '心烦易怒': '心中烦热、性情急躁易怒。多因肝郁化火或心火内扰，与情志关系密切。',
    '口苦': '口中味苦。多属胆火上炎或湿热内蕴，常见于肝胆湿热及郁热证候。',
    '小便色黄': '小便颜色深黄。多提示下焦热盛或湿热下注，热伤津液则尿少色深。',
    '大便干结': '大便干燥、排便困难。多因肠燥津亏或腑气不通，热盛伤津者尤甚。',
    '形体肥胖': '体形肥盛。多因脾虚失运、痰湿内蕴，所谓“肥人多痰湿”。',
    '腰膝酸软': '腰部与膝部酸软无力。多属肾精亏虚、筋骨失养，腰为肾之府。',
    '视物模糊': '视物不清、目睛干涩。多因肝肾阴虚、精血不能上荣于目。',
    '畏寒肢冷': '怕冷、四肢不温。多属阳气亏虚、温煦失职，常见于阴阳两虚或脾肾阳虚。',
    '下肢浮肿': '下肢按之凹陷、久不复起。多因脾肾阳虚、水湿泛溢肌肤所致。',
    '苔黄腻': '舌苔黄而黏腻。主湿热或痰热内蕴，是痰热互结证的重要舌象依据。',
    '舌红': '舌质鲜红。主热证或阴虚内热；舌红少苔者多属阴虚，舌红苔黄者多属实热。',
    '脉细': '脉道细小如线、应指无力。主虚证，多见于阴血不足或气虚。',
}


def get_syndrome_color(syndrome_name):
    """获取证型颜色"""
    return SYNDROME_INFO.get(syndrome_name, {}).get('color', '#888888')


def get_syndrome_icon(syndrome_name):
    """获取证型图标"""
    return SYNDROME_INFO.get(syndrome_name, {}).get('icon', '⚕️')


def format_probability(prob):
    """格式化概率显示"""
    return f"{prob:.1%}"


def create_feature_tooltip(feature_name):
    """创建特征提示文本（含中医属性与《中医诊断学》释义）"""
    attr = FEATURE_ATTRIBUTES.get(feature_name, '')
    desc = FEATURE_DESCRIPTIONS.get(feature_name, '')
    prefix = f"{feature_name}（{attr}）" if attr else feature_name
    return f"{prefix}：{desc}"


def validate_input(feature_dict):
    """验证输入特征"""
    errors = []
    
    for feature, value in feature_dict.items():
        if not isinstance(value, (int, float)):
            errors.append(f"{feature} 必须是数值")
        elif value < 0 or value > 1:
            errors.append(f"{feature} 必须在 0-1 之间")
    
    return len(errors) == 0, errors


def calculate_risk_score(probabilities):
    """
    计算模型不确定性评级（保留旧接口）

    注意：该函数的等级描述的是“不确定性”高低，等级越高代表把握越低，
    与页面上的“诊断把握度评级”语义相反。新代码请使用 calculate_confidence_level。
    """
    max_prob = max(probabilities.values())
    uncertainty = 1 - max_prob
    
    if uncertainty < 0.1:
        return "低", "模型判断把握很高"
    elif uncertainty < 0.3:
        return "中", "模型判断有一定把握"
    else:
        return "高", "模型判断把握较低，建议结合其他诊断方法"


def calculate_confidence_level(probabilities):
    """
    按主证概率给出“诊断把握度”评级（与页面文案语义一致）。

    阈值：>=60% 高；30%~60% 中（提示可能存在兼证）；<30% 低。
    返回 (等级, 说明文字)。
    """
    max_prob = max(probabilities.values())

    if max_prob >= 0.60:
        return "高", "模型对主证判断把握较高"
    if max_prob >= 0.30:
        return "中", "主证概率中等，提示可能存在兼证"
    return "低", "主证概率偏低，模型判断把握有限"
