# mnist_ui.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=5, verbose=0)

# Streamlit UI 구성
st.set_page_config(page_title="MNIST 숫자 예측기", layout="centered")
st.title("🧠 MNIST 숫자 예측기")
st.markdown("랜덤 손글씨 숫자 이미지를 보고 AI가 맞춰봅니다.")

if st.button("새로운 숫자 예측하기"):
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    label = y_test[idx]

    # 예측
    pred = model.predict(img.reshape(1, 28, 28))
    pred_label = np.argmax(pred)

    # 이미지 출력
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader(f"✅ 실제 숫자: {label}")
    st.subheader(f"🔮 AI의 예측: {pred_label}")
