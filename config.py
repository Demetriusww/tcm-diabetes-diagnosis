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

# 特征描述
FEATURE_DESCRIPTIONS = {
    '口渴': '反映津液损伤程度，口渴多饮为热盛或阴虚',
    '多食易饥': '反映胃火炽盛程度，消谷善饥为中消特征',
    '神疲乏力': '反映气虚程度，脾气亏虚则倦怠乏力',
    '心烦易怒': '反映肝郁化火程度，肝气郁结化火',
    '口苦': '反映胆火上炎或湿热内蕴',
    '小便色黄': '反映下焦热盛程度',
    '大便干结': '反映肠燥津亏或腑气不通',
    '形体肥胖': '反映痰湿内蕴程度，脾虚生痰',
    '腰膝酸软': '反映肝肾亏虚程度，腰为肾之府',
    '视物模糊': '反映肝肾不足，目失所养',
    '畏寒肢冷': '反映阳气亏虚程度，阳虚则寒',
    '下肢浮肿': '反映脾肾阳虚，水湿内停',
    '苔黄腻': '反映湿热或痰热内蕴程度',
    '舌红': '反映热象或阴虚程度',
    '脉细': '反映阴虚或血虚程度，脉细为阴血不足'
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
    """创建特征提示文本"""
    desc = FEATURE_DESCRIPTIONS.get(feature_name, '')
    return f"{feature_name}: {desc}"


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
    """计算风险评分"""
    max_prob = max(probabilities.values())
    uncertainty = 1 - max_prob
    
    if uncertainty < 0.1:
        return "低", "模型判断把握很高"
    elif uncertainty < 0.3:
        return "中", "模型判断有一定把握"
    else:
        return "高", "模型判断把握较低，建议结合其他诊断方法"
