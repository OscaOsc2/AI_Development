# 딥러닝에 필요한 기본 라이브러리 불러오기

import tensorflow as tf  # 텐서플로우: 딥러닝 모델 구축 및 학습에 사용하는 주요 프레임워크
import matplotlib.pyplot as plt  # matplotlib: 이미지나 그래프 시각화를 위한 라이브러리
import numpy as np  # numpy: 수치 계산 및 배열 처리 라이브러리

# 케라스에서 MNIST 데이터셋과 모델 구성 요소 불러오기
from tensorflow.keras.datasets import mnist  # MNIST: 손글씨 숫자 이미지 데이터셋
from tensorflow.keras.models import Sequential  # Sequential: 층을 순차적으로 쌓는 모델
from tensorflow.keras.layers import Dense, Flatten  # Dense: 완전연결층, Flatten: 다차원 -> 1차원 변환
from tensorflow.keras.utils import to_categorical  # to_categorical: 정수 라벨을 원-핫 인코딩으로 변환

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 학습용/테스트용 이미지 및 라벨 로드

# 2. 전처리
x_train = x_train / 255.0  # 픽셀 값을 0~1로 정규화하여 학습 안정성 향상
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)  # 정수 라벨을 10개 클래스의 원-핫 벡터로 변환
y_test = to_categorical(y_test, 10)

# 3. 모델 만들기
model = Sequential([  # Sequential: 층을 순차적으로 구성하는 모델 정의
    Flatten(input_shape=(28, 28)),  # Flatten: 28x28 이미지를 784개의 1차원 배열로 변환
    Dense(128, activation='relu'),  # Dense: 뉴런 128개, 활성화 함수 ReLU (은닉층)
    Dense(10, activation='softmax')  # Dense: 출력층, 클래스별 확률 출력 (0~9까지 총 10개)
])

# 4. 모델 컴파일
model.compile(optimizer='adam',  # Adam: 최적화 알고리즘. 학습률을 자동으로 조절
              loss='categorical_crossentropy',  # 손실 함수: 다중 클래스 분류에 적합
              metrics=['accuracy'])  # 모델 평가 지표로 정확도 사용

# 5. 훈련
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)  
# epochs=5: 전체 데이터셋을 5번 반복 학습
# batch_size=32: 한번에 32개 샘플로 나눠서 학습
# validation_split=0.1: 학습 데이터 중 10%를 검증용으로 사용

# 6. 평가
loss, acc = model.evaluate(x_test, y_test)  # 테스트 데이터로 모델 성능 평가
print(f"\nTest Accuracy: {acc:.4f}")  # 테스트 정확도 출력

# 랜덤 이미지 한 장 보기
index = np.random.randint(0, len(x_test))  # 테스트 데이터 중 랜덤한 인덱스 선택
plt.imshow(x_test[index], cmap='gray')  # 흑백으로 이미지 출력
plt.title("실제 라벨: " + str(np.argmax(y_test[index])))  # 실제 정답 라벨 출력
plt.axis('off')  # 축 정보 제거
plt.show()

# 예측
pred = model.predict(x_test[index].reshape(1, 28, 28))  # 모델에 이미지 한 장을 넣고 예측 수행
print("AI 예측:", np.argmax(pred))  # 예측 결과 출력 (확률이 가장 높은 클래스)
