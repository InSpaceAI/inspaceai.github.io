---
layout: post
title: "화재 감지 모델 테스트"
author: "박상민"
date: 2019-06-17 12:00:00
categories: [Review, Fire Detection]
tags: [FireNet, Test Review]
---

본 포스트에서는 딥러닝 기반의 화재 감지 모델에 대해 소개하고, 모델 테스트 결과에 대해 리뷰하도록 하겠습니다.

---


딥러닝 기반의 화재 감지 모델을 찾던 중 우연히, 화재 이미지를 분류 *(논문에서 binary fire detection 이라고 표현을 하였으나, 이진 분류라고 표현하는게 더 자연스러운 것 같습니다.)* 및 감지(Localization)하는 모델을 찾게 되었습니다.

논문 제목은 'EXPERIMENTALLY DEFINED CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE VARIANTS FOR NON-TEMPORAL REAL-TIME FIRE DETECTION' 이고, [논문](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)과 함께 [소스코드](https://github.com/tobybreckon/fire-detection-cnn)도 구현되어 깃헙에 공유되어 있으니 참고하시면 좋을 것 같습니다.

모델 3개 중 화재 이미지를 분류하는 모델 2개의 코드를 테스트 해봤습니다. 실제 테스트 환경에서 모델의 정상 작동 여부와 이미지 한장 당 처리 시간(fps)을 측정하여 정리해보았습니다.

보통 딥러닝 모델의 평가는 성능지표를 이용해 정확도 등을 측정하지만, 테스트 데이터 셋이 없고, 본 목적이 동작 여부 확인 및 처리 속도 측정이기 때문에 정확도는 측정하지 않았습니다.
<br>

# 모델 소개

논문에서는 총 3개의 모델을 제안합니다. 

1. FireNet 
    * 화재인지 아닌지 분류하는 이진 분류 모델입니다.
    * AlexNet 구조 기반이고, AlexNet에서 3번, 4번 레이어를 제거하여 3개의 레이어로 구성되어 있는 구조입니다. 
    * 논문에서는 정확도는 InceptionV1-OnFire 모델에 비해서는 낮지만, 처리속도는 더 빠르다고 합니다. 아래 사진은 FireNet 모델 구조입니다.  
        ![FireNet]({{ "/images/sangmin/FireNet.PNG" | prepend: site.baseurl }}) [출처](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)
 
2. InceptionV1-OnFire 
    * 화재인지 아닌지 분류하는 이진 분류 모델입니다.
    * InceptioinV1 구조 기반이고, 3개의 consecutive inception 모듈을 사용하였습니다. 아래 사진은 InceptionV1-OnFire 모델 구조 사진입니다.
    * 논문에서는 처리속도는 FireNet 모델에 비해서는 느리지만, 정확도는 더 높다고 합니다. 아래 사진은 FireNet 모델 구조입니다.  
        ![InceptionV1-OnFire]({{ "/images/sangmin/InceptionV1-OnFire.PNG" | prepend: site.baseurl }})
        [출처](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)


3. Superpixel Localization
    * InceptionV1-OnFire 모델에 Superpixel regions을 사용하여 화재 유무에 대해 Localization 하는 모델입니다.
    * SLIC(simple linear iteracitve clustering) Superpixel 기법을 사용하였고, Superpixel 기법은 이미지를 색상과 질감이 비슷한 지역을 over-segment 한다고 합니다. 아래 사진은 Superpixel 모델의 결과 사진입니다.  
        ![Superpixel]({{ "/images/sangmin/Superpixel.PNG" | prepend: site.baseurl }})
        [출처](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)

# 모델 테스트

1. 테스트 목표
    * 실제 환경에서 소스코드의 정상 작동 여부 확인과 모델의 이미지당 처리 속도(fps)를 측정하기 위함입니다.
    * 드론 촬영 영상에도 모델이 좋은 결과가 나오는지 확인하기 위함입니다.(모델 학습시에는 드론 영상이 아닌, 사람의 캠코더 위치 같은 낮은 위치에서 찍은 데이터를 사용하였습니다.)

2. 테스트 환경
    * 모델 테스트는 Linux Ubuntu GPU 서버에서 진행하였습니다. 상세 스펙은 다음과 같습니다.  
        ![GPU_Server_Spec.]({{ "/images/sangmin/GPU_server_spec.PNG" | prepend: site.baseurl }})

3. 테스트 데이터
    * 총 7개의 동영상 데이터를 이용해 테스트를 진행하였습니다. Github에 공유된 동영상과, 유튜브에 업로드된 드론 촬영 영상을 사용했습니다.
    * 7개 동영상 중 4개 동영상('barbecue/case2_car/test/rover')은 모델 학습, 평가에 사용했던 데이터 셋 이고, Github에 공유되어 있습니다.
    * 그 외에 3개 동영상('
    dalma[[1]](https://www.kaggle.com/csjcsj7477/firedetectionmodelkeras-for-video#dalma_400240.mp4), 
    drone_fire_1[[2]](https://www.youtube.com/watch?v=MyaHvSwiFuo&t=1222s), 
    drone_fire_2[[2]](https://www.youtube.com/watch?v=MyaHvSwiFuo&t=1222s)')은 유튜브에서 업로드 되어있는 동영상을 사용하였습니다.

* 테스트 모델
    * 테스트 모델은 FireNet과 InceptionV1-OnFire 2개 입니다.
    * 모델은 모두 학습이 된 상태로 github에 업로드 되었고, 학습된 모델을 불러와 테스트를 진행하였습니다.
    * 테스트에 필요한 코드(시간 측정, 이미지 저장 등) 수정 외에 튜닝, 추가 학습 등의 코드 수정은 하지 않았습니다.

4. 소스코드의 정상 작동 여부 확인 
    * 소스코드는 문제 없이 잘 작동했습니다. 소스코드가 있는 [Github](https://github.com/tobybreckon/fire-detection-cnn)을 참고하시면 좋을 것 같습니다.  
        ![FireNet_Github]({{ "/images/sangmin/FireNet_github.PNG" | prepend: site.baseurl }}) [출처](https://github.com/tobybreckon/fire-detection-cnn)
    

5. 모델별 결과 추출 속도(FPS) 측정
    * 논문에서 공개한 모델 성능지표와 처리속도는 다음과 같습니다.  
        ![FireNet_Evaluate]({{ "/images/sangmin/FireNet_Evaluate.PNG" | prepend: site.baseurl }}) [출처](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)
    * 논문에서는 GPU가 없는 환경(CPU : INtel Core i5, RAM 8GB)에서 측정을 하였으므로, 제가 측정한 지표들과 차이가 있을 수 있습니다.

    * 아래는 테스트 결과 비교 표 입니다.  
        ![FireNet_Github]({{ "/images/sangmin/Fire_Detection_model_compare.png" | prepend: site.baseurl }})  [출처](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)
    * 모델에서 측정한 지표보다 굉장히 좋게 나왔습니다. 

<br>

# 모델 출력 결과

영상 1~4는 사람의 눈높이 위치에서 촬영한 영상입니다. 
영상 5는 사람이 찍었지만 불 이 만힝 보이지 않는(전형적인 산불 영상 연기가 많이 나는 영상)입니다. 
영상 6~7은 드론을 이용해 고도가 높은 시점에서 촬영한 영상입니다.

* 영상 1
    * FireNet  
        ![FireNet]({{ "/images/sangmin/FireNet-barbecue.gif" | prepend: site.baseurl }}) 
    * InceptionV1-OnFire  
        ![InceptionV1-OnFire]({{ "/images/sangmin/Inception_barbecue.gif" | prepend: site.baseurl }}) 

* 영상 2
    * FireNet  
        ![FireNet]({{ "/images/sangmin/FireNet-case2_car.gif" | prepend: site.baseurl }}) 
    * InceptionV1-OnFire  
        ![InceptionV1-OnFire]({{ "/images/sangmin/Inception_case2_car.gif" | prepend: site.baseurl }}) 

* 영상 3
    * FireNet  
        ![FireNet]({{ "/images/sangmin/FireNet-test.gif" | prepend: site.baseurl }}) 
    * InceptionV1-OnFire   
        ![InceptionV1-OnFire]({{ "/images/sangmin/Inception-test.gif" | prepend: site.baseurl }}) 

* 영상 4
    * FireNet  
        ![FireNet]({{ "/images/sangmin/FireNet-rover.gif" | prepend: site.baseurl }}) 
    * InceptionV1-OnFire  
        ![InceptionV1-OnFire]({{ "/images/sangmin/Inception-rover.gif" | prepend: site.baseurl }}) 


* 영상 5
    * FireNet  
        ![FireNet]({{ "/images/sangmin/FireNet-dalma.gif" | prepend: site.baseurl }}) 
    * InceptionV1-OnFire  
        ![InceptionV1-OnFire]({{ "/images/sangmin/inception-dalma.gif" | prepend: site.baseurl }}) 

* 영상 6
    * FireNet  
        ![FireNet]({{ "/images/sangmin/FireNet-drone_fire_1.gif" | prepend: site.baseurl }}) 

    * InceptionV1-OnFire  
        ![InceptionV1-OnFire]({{ "/images/sangmin/inception-drone_fire_1.gif" | prepend: site.baseurl }}) 

* 영상 7
    * FireNet  
        ![FireNet]({{ "/images/sangmin/FireNet_drone_fire_2.gif" | prepend: site.baseurl }}) 

    * InceptionV1-OnFire  
        ![InceptionV1-OnFire]({{ "/images/sangmin/inception-drone_fire_2.gif" | prepend: site.baseurl }}) 




