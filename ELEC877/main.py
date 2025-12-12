# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class IoTTrafficSimulator:

    def __init__(self, n_samples=2000):
        self.n_samples = n_samples
        # 定义设备配置 (Paper Table I)
        # size_mu: 包大小均值, size_std: 包大小标准差
        # iat_mu: 到达间隔均值, iat_std: 到达间隔标准差
        # proto: 协议 (0=TCP, 1=UDP)
        self.devices = {
            '智能摄像头 (Camera)': {'size_mu': 1200, 'size_std': 200, 'iat_mu': 0.05, 'iat_std': 0.02, 'proto': 1},
            '智能插座 (Plug)':     {'size_mu': 64,   'size_std': 10,  'iat_mu': 5.0,  'iat_std': 2.0,  'proto': 0},
            '语音助手 (Voice)':    {'size_mu': 300,  'size_std': 150, 'iat_mu': 0.5,  'iat_std': 0.3,  'proto': 0},
            '温控器 (Thermostat)': {'size_mu': 128,  'size_std': 20,  'iat_mu': 60.0, 'iat_std': 5.0,  'proto': 0}
        }

    def generate(self):
        print("🔄根据统计模型生成仿真流量数据...")
        data = []
        labels = []
        
        samples_per_device = self.n_samples // len(self.devices)

        for device_name, stats in self.devices.items():
            for _ in range(samples_per_device):
                # 实现公式 (1) 和 (2): S ~ N(mu, sigma), I ~ N(mu, sigma)
                pkt_size_mean = np.random.normal(stats['size_mu'], stats['size_std'])
                pkt_size_std  = abs(np.random.normal(stats['size_std'], 5))
                iat_mean      = abs(np.random.normal(stats['iat_mu'], stats['iat_std']))
                iat_std       = abs(np.random.normal(stats['iat_std'], 0.01))
                proto         = stats['proto']
                
                # 噪声注入
                # 模拟 5% 的概率出现网络抖动导致的异常包
                if np.random.random() < 0.05:
                    pkt_size_mean += np.random.randint(100, 500)
                
                # 确保数据非负
                pkt_size_mean = max(0, pkt_size_mean)
                iat_mean = max(0.001, iat_mean)

                data.append([pkt_size_mean, pkt_size_std, iat_mean, iat_std, proto])
                labels.append(device_name)

        columns = ['包大小均值', '包大小方差', '间隔均值', '间隔方差', '协议类型']
        df = pd.DataFrame(data, columns=columns)
        df['Label'] = labels
        return df

class ModelTrainer:
    """
   
    """
    def __init__(self, df):
        self.df = df
        self.X = df.drop('Label', axis=1)
        self.y = df['Label']
        self.scaler = StandardScaler()
        self.best_model = None
        self.X_test_scaled = None
        self.y_test = None
        self.classes = None

    def preprocess_and_split(self):
        # 数据划分 (Paper Section V-A)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        
        # 标准化 (Paper Section IV-A: Z-score normalization)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_test = y_test
        
        return X_train_scaled, y_train

    def train_with_gridsearch(self, X_train, y_train):
        print("\n⚙️网格搜索 (Grid Search) 训练模型...")
        
        # 定义 Random Forest 和 SVM 的参数网格
        model_params = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {'C': [1, 10], 'kernel': ['rbf']}
            }
        }

        best_score = 0
        
        for name, config in model_params.items():
            print(f"   -> 正在优化 {name} ...")
            grid = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            
            print(f"      {name} 最佳准确率: {grid.best_score_:.4f} | 参数: {grid.best_params_}")
            
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                self.best_model = grid.best_estimator_
        
        self.classes = self.best_model.classes_
        print(f"✅ 最终选择模型: {type(self.best_model).__name__}")

class OpenSetDetector:

    def __init__(self, model, scaler, threshold=0.6):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def detect(self, features):
        print("\n🛡️ 开放集异常检测 (Open-Set Detection)")
        # 标准化输入特征
        features_scaled = self.scaler.transform(features)
        
        # 获取预测概率 (Posterior Probability)
        probs = self.model.predict_proba(features_scaled)
        pred_indices = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        
        results = []
        for i, prob in enumerate(max_probs):
            pred_class = self.model.classes_[pred_indices[i]]
            
    
            if prob < self.threshold:
                status = "🚨 警报: 未知设备/潜在攻击 (Unknown)"
            else:
                status = f"✅ 认证通过: {pred_class}"
            
            print(f"   输入向量: {features[i]}")
            print(f"   预测类: {pred_class} | 置信度: {prob:.4f} -> {status}")
            results.append(status)
        return results

class Visualizer:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)

    def plot_results(self):
        print("\n📊 [模块 4] 生成可视化图表...")
        plt.figure(figsize=(14, 6))

        # 1. 混淆矩阵 (Fig. 1 in Paper)
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.title('Fig. 1. Confusion Matrix (Test Data)')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        # 2. PCA 降维可视化
        plt.subplot(1, 2, 2)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_test)
        # 将 y_test 转换为颜色索引（简化处理）
        unique_labels = list(set(self.y_test))
        # 使用 seaborn 绘制散点图
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=self.y_test, palette='deep', s=60)
        plt.title('IoT Device Clusters (PCA Projection)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.tight_layout()
        plt.show()
        print("✅ 图表已生成。")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 生成数据
    simulator = IoTTrafficSimulator(n_samples=2000)
    df = simulator.generate()

    # 2. 训练模型
    trainer = ModelTrainer(df)
    X_train, y_train = trainer.preprocess_and_split()
    trainer.train_with_gridsearch(X_train, y_train)

    # 3. 可视化评估
    viz = Visualizer(trainer.best_model, trainer.X_test_scaled, trainer.y_test)
    viz.plot_results()

    # 构造一个高频 UDP 攻击流 (包很大，间隔极短)
    # [包大小均值, 包大小方差, 间隔均值, 间隔方差, 协议]
    attack_vector = np.array([
        [2000, 500, 0.001, 0.001, 1],  # 模拟 DDoS 攻击
        [64, 5, 5.0, 0.1, 0]           # 模拟正常的智能插座 (用于对比)
    ])
    
    detector = OpenSetDetector(trainer.best_model, trainer.scaler, threshold=0.6)
    detector.detect(attack_vector)
    
