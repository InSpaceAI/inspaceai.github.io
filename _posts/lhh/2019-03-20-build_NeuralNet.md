---
layout: post
title: "NeuralNet 만들기"
author: "이현호"
date: 2019-03-20 16:00:00
categories: [Tutorials, XAI]
tags: [XAI, Multi-Classification]
---

본 포스트에서는 XAI 중 Model-Agnostic Method를 적용하기 위해 간단한 Neural Network를 만들어보도록 하겠습니다.

---

## 필요 라이브러리 import

 ```python
 import tensorflow as tf
 import numpy as np
 from sklearn.datasets import load_iris
 ```
 iris 데이터를 학습할 모델을 생성하기 위해 필요한 라이브러리들을 import 해줍니다.

## 데이터 전처리

 ```python
 iris_data = load_iris()
 ```
 iris 데이터를 불러옵니다.
 
 ```python
 total = np.c_[iris_data['data'], iris_data['target']]
 np.random.shuffle(total)
 ```
 iris 데이터 분석 챕터에서 확인 했듯이 target이 섞여있지 않았습니다.  
 훈련셋과 시험셋을 랜덤하게 나눠주기 위해 data와 target을 합쳐서 shuffle 해줍니다.

 ```python
 train, test = total[:len(total)*8//10], totla[len(total)*8//10:]
 x_train, y_train = train[:,:4], train[:, 4]
 x_test, y_test = test[:,:4], test[:, 4]
 ```
 shuffle한 데이터를 학습셋과 시험셋으로 나눠줍니다.  
 그리고 모델에 들어가기 위해 feature와 target으로 나눠줍니다.

## 모델 생성

 ```python
 model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(4,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(32, activation=tf.nn.tanh),
    tf.keras.layers.Dense(32, activation=tf.nn.tanh),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax),
 ])

 model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
 ```
 iris 데이터를 학습할 간단한 모델을 만들어 줍니다.
 분류문제를 해결할 것이므로 categorical_crossentropy 손실함수를 사용합니다.

## 모델 학습

 ```python
 model.fit(x_train, y_train, epochs=10)
 ```
 __Epoch 1/10
 120/120 [==============================] - 0s 1ms/sample - loss: 1.1548 - acc:  0.3583
 Epoch 2/10
 120/120 [==============================] - 0s 83us/sample - loss: 1.0667 - acc: 0.4083
 Epoch 3/10
 120/120 [==============================] - 0s 91us/sample - loss: 0.9996 - acc: 0.5833
 Epoch 4/10
 120/120 [==============================] - 0s 91us/sample - loss: 0.9528 - acc: 0.6500
 Epoch 5/10
 120/120 [==============================] - 0s 91us/sample - loss: 0.8987 - acc: 0.7000
 Epoch 6/10
 120/120 [==============================] - 0s 83us/sample - loss: 0.8430 - acc: 0.7333
 Epoch 7/10
 120/120 [==============================] - 0s 91us/sample - loss: 0.7873 - acc: 0.7667
 Epoch 8/10
 120/120 [==============================] - 0s 91us/sample - loss: 0.7366 - acc: 0.9333
 Epoch 9/10
 120/120 [==============================] - 0s 83us/sample - loss: 0.6843 - acc: 0.9833
 Epoch 10/10
 120/120 [==============================] - 0s 91us/sample - loss: 0.6368 - acc: 0.9917__

 10 epochs로 학습셋을 학습합니다.

## 모델 평가

 ```python
 model.evaluate(x_test, y_test)
 ```
 __30/30 [==============================] - 0s 2ms/sample - loss: 0.6846 - acc: 0.9333__

 90% 넘는 정확도를 가지고 있는 모델을 만들었습니다.

## 모델 저장
 
 ```python
 model.save('iris_model.h5')
 ```
 모델을 저장해 줍니다.



