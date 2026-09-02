"""
模型服务模块 - 更新为6个证型和15个症状特征
提供训练好的模型和 SHAP 解释器
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import shap
import pickle
import os

# 特征名称列表 - 15个症状特征（基于dataset_final_6class.csv）
FEATURE_NAMES = [
    '口渴', '多食易饥', '神疲乏力', '心烦易怒', '口苦',
    '小便色黄', '大便干结', '形体肥胖', '腰膝酸软', '视物模糊',
    '畏寒肢冷', '下肢浮肿', '苔黄腻', '舌红', '脉细'
]

# 证型标签 - 6个证型（基于dataset_final_6class.csv）
SYNDROME_LABELS = ['气阴两虚', '痰热互结', '肝肾阴虚', '热盛伤津', '肝胃郁热', '阴阳两虚']


def create_mock_model():
    """
    创建模拟的训练好的模型（基于6个证型和15个特征）
    """
    np.random.seed(42)
    
    # 生成模拟数据
    n_samples = 2000
    X = np.random.rand(n_samples, len(FEATURE_NAMES))
    
    # 创建有区分度的标签（6个证型）
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # 6个证型的特征权重（基于CSV数据中的症状关联）
        scores = [
            # 气阴两虚：口渴、神疲乏力、腰膝酸软、视物模糊、舌红、脉细
            X[i, 0] + X[i, 2] + X[i, 8] + X[i, 9] + X[i, 13] + X[i, 14],
            # 痰热互结：口渴、口苦、苔黄腻、形体肥胖、心烦易怒、大便干结
            X[i, 0] + X[i, 4] + X[i, 12] + X[i, 7] + X[i, 3] + X[i, 6],
            # 肝肾阴虚：腰膝酸软、视物模糊、舌红、口渴、脉细、形体肥胖
            X[i, 8] + X[i, 9] + X[i, 13] + X[i, 0] + X[i, 14] + X[i, 7],
            # 热盛伤津：口渴、多食易饥、口苦、小便色黄、大便干结、舌红
            X[i, 0] + X[i, 1] + X[i, 4] + X[i, 5] + X[i, 6] + X[i, 13],
            # 肝胃郁热：心烦易怒、口苦、口渴、多食易饥、大便干结、舌红
            X[i, 3] + X[i, 4] + X[i, 0] + X[i, 1] + X[i, 6] + X[i, 13],
            # 阴阳两虚：畏寒肢冷、下肢浮肿、腰膝酸软、视物模糊、神疲乏力、脉细
            X[i, 10] + X[i, 11] + X[i, 8] + X[i, 9] + X[i, 2] + X[i, 14],
        ]
        y[i] = np.argmax(scores)
    
    # 训练随机森林模型
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    
    return model


def create_shap_explainer(model):
    """
    创建 SHAP 解释器（树模型推荐不传 background_data，
    避免多分类返回维度与旧版list不一致导致零值问题）
    """
    np.random.seed(42)
    explainer = shap.TreeExplainer(model)
    return explainer


def save_model_and_explainer(model, explainer, save_dir='models'):
    """
    保存模型和解释器
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(save_dir, 'shap_explainer.pkl'), 'wb') as f:
        pickle.dump(explainer, f)
    
    print(f"模型和解释器已保存到 {save_dir}/")


def load_model_and_explainer(model_dir='models'):
    """
    加载模型和解释器
    """
    model_path = os.path.join(model_dir, 'best_model.pkl')
    explainer_path = os.path.join(model_dir, 'shap_explainer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(explainer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(explainer_path, 'rb') as f:
            explainer = pickle.load(f)
        return model, explainer
    else:
        # 如果文件不存在，创建模拟模型
        print("未找到已保存的模型，正在创建模拟模型...")
        model = create_mock_model()
        explainer = create_shap_explainer(model)
        save_model_and_explainer(model, explainer, model_dir)
        return model, explainer


def predict_syndrome(model, input_features):
    """
    预测证型（6个证型）
    
    Parameters:
    -----------
    model : 训练好的模型
    input_features : array-like, 形状 (n_features,)
        输入特征向量
    
    Returns:
    --------
    dict : 包含预测结果和概率的字典
    """
    # 确保输入是二维数组
    if len(input_features.shape) == 1:
        input_features = input_features.reshape(1, -1)
    
    # 获取预测概率
    probabilities = model.predict_proba(input_features)[0]
    predicted_class = np.argmax(probabilities)
    
    result = {
        'predicted_syndrome': SYNDROME_LABELS[predicted_class],
        'probabilities': dict(zip(SYNDROME_LABELS, probabilities)),
        'confidence': probabilities[predicted_class]
    }
    
    return result


def get_shap_values(explainer, input_features):
    """
    计算 SHAP 值（兼容新版 SHAP：多分类返回 3D ndarray (sample, feat, class)，
    以及旧版 SHAP：返回 list，每个元素对应一类的 SHAP 值）

    Returns:
        dict : SHAP 值字典，键为证型名称，值为 1D numpy 数组 (n_features,)
    """
    if len(input_features.shape) == 1:
        input_features = input_features.reshape(1, -1)

    shap_out = explainer.shap_values(input_features)
    result = {}

    # ------------------------------------------------------
    # 新版 SHAP：输出 (n_samples, n_features, n_classes) 的 3D 数组
    # （这是目前全0问题的主因，原代码未处理）
    # ------------------------------------------------------
    if isinstance(shap_out, np.ndarray):
        sv = np.asarray(shap_out)
        # 如果是 3 维：(sample, feat, class)
        if sv.ndim == 3:
            n_classes = sv.shape[2]
            for i in range(min(n_classes, len(SYNDROME_LABELS))):
                feat_sv = sv[0, :, i]  # 取第 0 个样本，第 i 类
                # 对齐长度（防止维度不匹配）
                if len(feat_sv) >= len(FEATURE_NAMES):
                    result[SYNDROME_LABELS[i]] = np.asarray(feat_sv[:len(FEATURE_NAMES)], dtype=float)
                else:
                    pad = np.zeros(len(FEATURE_NAMES), dtype=float)
                    pad[:len(feat_sv)] = feat_sv
                    result[SYNDROME_LABELS[i]] = pad
            # 若类别不足，补齐剩余类别为零数组
            for i in range(n_classes, len(SYNDROME_LABELS)):
                result[SYNDROME_LABELS[i]] = np.zeros(len(FEATURE_NAMES))
            return result

        # 2 维：(feat, class) 或 (sample, feat)
        if sv.ndim == 2:
            if sv.shape[0] == 1 and sv.shape[1] == len(FEATURE_NAMES):
                # 二分类：只返回第 0 类
                return {SYNDROME_LABELS[0]: sv.flatten().astype(float)}
            if sv.shape[1] == len(SYNDROME_LABELS) and sv.shape[0] == len(FEATURE_NAMES):
                # (feat, class)
                for i in range(len(SYNDROME_LABELS)):
                    result[SYNDROME_LABELS[i]] = sv[:, i].flatten().astype(float)
                return result
            if sv.shape[0] == len(SYNDROME_LABELS) and sv.shape[1] == len(FEATURE_NAMES):
                # (class, feat)
                for i in range(len(SYNDROME_LABELS)):
                    result[SYNDROME_LABELS[i]] = sv[i, :].flatten().astype(float)
                return result

    # ------------------------------------------------------
    # 旧版 SHAP：list，每个元素对应一类的 SHAP 值
    # ------------------------------------------------------
    if isinstance(shap_out, list):
        for i, label in enumerate(SYNDROME_LABELS):
            try:
                s = shap_out[i]
                s = np.asarray(s, dtype=float).flatten()
                if len(s) >= len(FEATURE_NAMES):
                    result[label] = s[:len(FEATURE_NAMES)]
                else:
                    pad = np.zeros(len(FEATURE_NAMES), dtype=float)
                    pad[:len(s)] = s
                    result[label] = pad
            except (IndexError, AttributeError, ValueError):
                result[label] = np.zeros(len(FEATURE_NAMES), dtype=float)
        return result

    # ------------------------------------------------------
    # 兜底：返回零数组（每类都给一个零向量避免报错）
    # ------------------------------------------------------
    for lbl in SYNDROME_LABELS:
        result[lbl] = np.zeros(len(FEATURE_NAMES), dtype=float)
    return result


if __name__ == '__main__':
    # 测试模型创建
    print("正在创建6证型模型和 SHAP 解释器...")
    model = create_mock_model()
    explainer = create_shap_explainer(model)
    save_model_and_explainer(model, explainer)
    print("完成！")
    print(f"特征数量: {len(FEATURE_NAMES)}")
    print(f"证型数量: {len(SYNDROME_LABELS)}")
    print(f"证型列表: {SYNDROME_LABELS}")
