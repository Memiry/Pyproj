import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 设置绘图风格
sns.set(style="whitegrid")


# ==========================================
# 1. 模块：高级数据模拟器 (IoT Traffic Simulator) 测试上传
# ==========================================
# 在真实项目中，这里会替换为读取 .pcap 转出的 CSV 文件
# 这里我们模拟生成具有统计规律的“流特征”数据
# ==========================================
def generate_iot_data(n_samples=1000):
    print("🔄 [Step 1] 正在生成仿真 IoT 流量数据...")
    data = []
    labels = []
    
    # 定义不同设备的流量特征 (均值, 标准差)
    # 特征包括: [包大小均值, 包大小方差, 包间隔均值, 包间隔方差, 协议类型(0=TCP, 1=UDP)]
    devices = {
        'Smart_Camera': {'size_mu': 1200, 'size_std': 200, 'iat_mu': 0.05, 'iat_std': 0.02, 'proto': 1}, # 视频流: 包大, 间隔短, UDP
        'Smart_Plug':   {'size_mu': 64,   'size_std': 10,  'iat_mu': 5.0,  'iat_std': 2.0,  'proto': 0}, # 心跳包: 包小, 间隔长, TCP
        'Voice_Asst':   {'size_mu': 300,  'size_std': 150, 'iat_mu': 0.5,  'iat_std': 0.3,  'proto': 0}, # 语音助手: 中等, 突发
        'Thermostat':   {'size_mu': 128,  'size_std': 20,  'iat_mu': 60.0, 'iat_std': 5.0,  'proto': 0}  # 温控器: 很稀疏
    }

    for device, stats in devices.items():
        for _ in range(n_samples // 4):
            # 模拟生成特征，加入高斯噪声
            pkt_size_mean = np.random.normal(stats['size_mu'], stats['size_std'])
            pkt_size_std  = abs(np.random.normal(stats['size_std'], 5)) # 方差
            iat_mean      = abs(np.random.normal(stats['iat_mu'], stats['iat_std']))
            iat_std       = abs(np.random.normal(stats['iat_std'], 0.01))
            proto         = stats['proto'] # 这里简化为固定，实际会有波动
            
            # 加入一些异常点/噪声
            if np.random.random() < 0.05: 
                pkt_size_mean += 500
            
            data.append([pkt_size_mean, pkt_size_std, iat_mean, iat_std, proto])
            labels.append(device)

    columns = ['pkt_size_mean', 'pkt_size_std', 'iat_mean', 'iat_std', 'protocol']
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    return df

# ==========================================
# 2. 模块：数据预处理与分割
# ==========================================
df = generate_iot_data(2000)
X = df.drop('label', axis=1)
y = df['label']

# 划分: 训练集, 测试集, 和一个"验证集"(用于模拟未知设备)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化 (对于 SVM 和 KNN 非常重要)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ 数据准备完成。样本总数: {len(df)}")

# ==========================================
# 3. 模块：模型训练与超参数优化 (GridSearch)
# ==========================================
print("\n⚙️ [Step 2] 开始模型训练与优化...")

models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]} # 寻找最佳树数量和深度
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42), # probability=True 为了后续做未知检测
        'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']} # 寻找最佳核函数
    }
}

best_models = {}
results = []

for name, config in models.items():
    print(f"   -> 正在优化 {name} ...")
    grid = GridSearchCV(config['model'], config['params'], cv=5, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    best_models[name] = grid.best_estimator_
    score = grid.best_score_
    results.append({'Model': name, 'Best Accuracy': score, 'Best Params': grid.best_params_})
    print(f"      最佳准确率: {score:.4f} | 参数: {grid.best_params_}")

# ==========================================
# 4. 模块：评估与可视化 (PCA & 混淆矩阵)
# ==========================================
print("\n📊 [Step 3] 结果评估与可视化...")

# 选择表现最好的模型 (这里默认选 RF 进行详细展示)
final_model = best_models['RandomForest']
y_pred = final_model.predict(X_test_scaled)

# --- 图1: 特征重要性 (Feature Importance) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
feat_importances = pd.Series(final_model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh', color='teal')
plt.title('Top 5 Key Features for IoT Identification')
plt.xlabel('Importance Score')

# --- 图2: PCA 降维可视化 (将5维数据压扁成2维看分布) ---
# 这能直观展示为什么模型能区分开它们
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_test, palette='deep', s=60)
plt.title('IoT Device Clusters (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()

# --- 打印混淆矩阵 ---
print("\n--- 最终测试集报告 (Random Forest) ---")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. 模块：未知设备检测 (Security Logic)
# ==========================================
print("\n🛡️ [Step 4] 未知设备入侵检测模拟")

# 模拟一个完全未知的设备 (比如一个黑客的攻击工具，行为模式不在我们库里)
# 它的特征：包非常大，发包非常快 (High Throughput)
unknown_device = np.array([[2000, 500, 0.001, 0.001, 1]]) 
unknown_device_scaled = scaler.transform(unknown_device)

# 获取模型预测的“概率分布”
probs = final_model.predict_proba(unknown_device_scaled)
max_prob = np.max(probs)
pred_label = final_model.predict(unknown_device_scaled)[0]

print(f"输入特征: {unknown_device}")
print(f"模型初步归类: {pred_label} (置信度: {max_prob:.4f})")

# 设定阈值：如果最高置信度低于 0.6，则认为是未知设备
THRESHOLD = 0.6
if max_prob < THRESHOLD:
    print("🚨 结果: 【未知设备/潜在威胁】 (置信度低，触发防御警报)")
else:
    print(f"✅ 结果: 已识别为 {pred_label}")

print("\n🎉 项目运行结束！")