import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 1. 读取数据 (注意：有些csv编码格式不同，这里用latin-1以防报错)
# 假设你的文件名为 spam.csv
df = pd.read_csv('spam.csv', encoding='latin-1')

# 只保留我们要的两列：v1是标签(ham/spam)，v2是短信内容
df = df[['v1', 'v2']]
df.columns = ['label', 'message'] # 重命名方便后续操作

# 2. 划分数据：一部分用来“训练/学习”，一部分用来“考试/测试”
# X 是短信内容，y 是标签
X = df['message']
y = df['label']

# 80%的数据用来训练，20%用来测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建模型管道 (Pipeline)
# 这一步是魔法所在：
# CountVectorizer: 把文字变成数字（比如统计"Free"出现了几次）
# MultinomialNB: 朴素贝叶斯分类器，最适合处理文本分类
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 4. 训练模型 (让AI开始学习)
print("正在训练模型...")
model.fit(X_train, y_train)

# 5. 测试模型 (看看它学得怎么样)
predictions = model.predict(X_test)
print(f"模型准确率: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. 自己动手试一试！ (实战演示)
print("\n--- 实战测试 ---")
my_messages = [
    "Hey, are we still going for lunch today?",    # 正常短信
    "Congratulations! You've won a $1000 gift card. Click here to claim now!", # 垃圾短信
    "Urgent! Your mobile number has been awarded a free iPhone.", # 垃圾短信
    "Mom called, call her back when you are free." # 正常短信
]

my_predictions = model.predict(my_messages)

for msg, label in zip(my_messages, my_predictions):
    print(f"短信内容: {msg}\n预测结果: [{label}]\n")