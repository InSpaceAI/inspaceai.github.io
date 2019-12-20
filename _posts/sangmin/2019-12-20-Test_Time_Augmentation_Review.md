---
layout: post
title: "Test Time Augmentation"
author: "박상민"
date: 2019-12-20 12:00:00
categories: [Tutorials, TTA]
tags: [TTA, Augmentation]
---

본 포스트에서는 Test Time Augmentation에 대해서 소개하고, 관련 코드도 리뷰하도록 하겠습니다.

---

Data Augmentation에 대해 자료조사를 하던 중, TTA(Test Time Augmentation)에 대해 알게되어, 관련 소개를 하고 테스트를 할 수 있는 코드로 리뷰하도록 하겠습니다.

모든 그림과, 소스코드 등은 아래의 게시글에서 모두 참조하였습니다.  
(https://www.kaggle.com/andrewkh/test-time-augmentation-tta-worth-it)  
(https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d)

<br>

# What is Test Time Augmentation?

딥러닝 모델을 학습시킬 때, 데이터가 부족하는 등의 문제가 생기면 Data Augmentation을 사용합니다. 기존에 있는 데이터 셋을 회전/반전/뒤집기/늘이기/줄이기/노이즈 등 다양항 방법을 사용하여 부풀립니다.  

Test Time Augmentation(TTA)도 Augmentation하는 방법은 같습니다. 하지만 학습할 때 augmentation하는게 아닌, 테스트 셋으로 모델을 테스트하거나, 실제 운영할 때 augmentation을 수행하는 것 입니다. 

![TTA_1]({{"/images/sangmin/TTA_1.png" | prepend: site.baseurl }})  

위 사진을 예로 들자면, 원본 입력 이미지 한장을 augmentation해서 2장 더 만들었습니다. 총 3장의 이미지를 학습된 모델에 각각 입력해주고, 모델에서 나온 결과는 모두 평균화하여 최종 결과를 도출하게 됩니다. TTA의 가장 중요한 컨셉은 원본 이미지 한장만 입력받아 결과를 예측하는 것이 아니라, 원본이미지를 augmentation한 다양한 관점의 이미지들을 입력받아 최종 결과를 도출하는 것 입니다.  

좀 더 자세한 예를 들어보겠습니다.  

![The test image]({{"/images/sangmin/TTA_2.png" | prepend: site.baseurl }})  

학습된 분류 모델에 위 테스트 이미지를 입력하여 분류 결과를 에측하려고 합니다. 테스트 이미지를 모델에 입력하면 아래와 같이 각 클래스에 대한 예측 확률이 나옵니다. 가장 높은 확률을 최종 도출하므로 2번째 클래스로 예측을 합니다.  

![Class Socre]({{"/images/sangmin/TTA_3.png" | prepend: site.baseurl }})  

그리고 테스트 이미지의 정답은 아래와 같다고 할 때,  

![Class Label]({{"/images/sangmin/TTA_4.png" | prepend: site.baseurl }})   

모델에 예측한 분류 결과는 틀렸다는 걸 알 수 있습니다. 모델은 2번째 클래스의 확률이 가장 높다고 예측하였지만, 정답은 9번째 클래스였습니다.  

만약 위 상황에서 Test Time Augmentation을 적용한다면 어떻게 될까요? 같은 이미지를 약간만 변형시킨 5장의 이미지를 모델에 입력해보고, 분류 결과를 예측해 보겠습니다.  

![modify test image]]({{"/images/sangmin/TTA_5.png" | prepend: site.baseurl }})  

모델에서 출력된 각 이미지들의 클래스 확률은 다음과 같습니다.  

![modify test image score]]({{"/images/sangmin/TTA_6.png" | prepend: site.baseurl }})  

보시다시피, 1번과 4번 이미지는 9번째 클래스 예측 확률(결과)이 가장 합리적이고, 높게 나왔습니다. 2번 이미지는 2번째와 9번째 클래스 예측 확률의 차이가 매우 적고, 3번과 5번 이미지는 분류 결과가 틀렸습니다. 이제 5개의 결과를 모두 평균으로 구하면 어떻게 될까요?  

![avg all score]({{"/images/sangmin/TTA_7.png" | prepend: site.baseurl }})  
