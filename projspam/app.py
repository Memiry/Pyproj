import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- 训练部分 (和上面一样，每次启动重新训练一下，很快) ---
df = pd.read_csv('spam.csv', encoding='latin-1')
X = df['v2']
y = df['v1']
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)

# --- 网页界面部分 ---
st.title("🛡️ 垃圾短信检测器")
st.write("这是一个基于机器学习(朴素贝叶斯)的网络安全防御小工具。")

user_input = st.text_area("请输入一条英文短信内容：", "Congratulations! You won a prize.")

if st.button("检测"):
    result = model.predict([user_input])[0]
    if result == 'spam':
        st.error(f"🚨 警告：这是一条垃圾短信 (Spam)！")
    else:
        st.success(f"✅ 安全：这是一条正常短信 (Ham)。")