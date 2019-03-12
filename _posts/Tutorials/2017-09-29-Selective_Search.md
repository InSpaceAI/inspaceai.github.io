---
layout: post
title: "선택적 탐색"
author: "정한솔"
date: 2017-09-29 17:00:00
categories: Tutorials
comments: true
---

이 포스트에서는 객체 검출에서 객체의 영역 후보를 구하는 방법 중의 하나인 선택적 탐색(selective search) 알고리즘에 대해서 설명하겠습니다.

---

### 개요

선택적 탐색은 2012년 Uijlings 등의 논문에서 제안된, 이미지에서 객체가 존재할 후보 영역을 찾는 알고리즘입니다. 이 알고리즘은 입력 이미지를 분할(segmentation)한 뒤, 해당 정보를 기반으로 객체가 존재할 후보 영역이 높은 영역을 찾습니다. 이 논문에서는 기계학습 알고리즘의 일종인 서포트 벡터 머신(support vector machine)과 결합해서 객체 검출하는 방법을 함께 제안하고 있습니다.

이후에는 서포트 벡터 머신 대신 딥 러닝 기반의 R-CNN, Fast R-CNN, SPPNet 등의 객체 검출 알고리즘에서도 선택적 탐색 알고리즘을 사용하여 객체가 존재할 후보 영역을 추출하고 있습니다. 하지만 Faster R-CNN, YOLO, FCN 등의 최신 알고리즘에서는 후보 영역 선정 방법이 CNN과 통합되거나 구조 자체가 변화되어 더이상 선택적 탐색을 사용하지 않습니다.

---

### 영상 분할

이미지 전체에서 완전 탐색(exhaustive search)을 하는 경우엔 가능한 후보 영역 갯수가 지나치게 많아지고 정확도도 떨어지기 때문에 객체 검출에는 사용할 수 없습니다. 따라서 물체가 위치할 가능성이 높은 영역만을 제안하기 위한 특별한 알고리즘이 필요합니다. 선택적 탐색은 이미지를 분할하여 한 분할 영역 안에는 하나의 객체만이 존재할 가능성이 높다고 간주하고 해당 영역 내에서만 완전 탐색을 실행하기 때문에 정확도와 속도가 개선되었습니다.

선택적 탐색에서는 영상 분할 알고리즘은 선택이 매우 중요합니다. 해당 논문에서는 2004년 Felzenszwalb 등의 논문에서 제안된 그래프 기반의 영상 분할 알고리즘을 선택합니다. 각 픽셀을 그래프의 노드로 간주하고, 각 노드끼리 선분(edge)으로 연결하여 그래프를 형성합니다. 각 선분에는 가중치(weight)가 지정됩니다. 그리고 모든 노드를 순회하면서 두 그룹의 최대 가중치 중 작은 쪽과 두 그룹을 연결하는 선분의 최소 가중치를 비교하여, 전자가 작으면 그룹을 분할된 상태로 그대로 두고, 후자가 작으면 두 그룹을 통합합니다. 즉, 그룹 내의 가중치보다 그룹 간의 가중치가 더 커지도록 그룹을 병합해갑니다.

이 과정에서 사용되는 가중치를 구하기 위해서, 모든 픽셀을 x, y, r, g, b 5개 축을 갖는 5차원 유클리드 공간으로 투영합니다. 여기서 x, y는 해당 픽셀의 위치이고, r, g, b는 해당 픽셀의 색상 값입니다. 각 픽셀 간의 유클리드 거리를 선분의 가중치로 두고 그래프 병합을 실시합니다.

![Felzenszwalb image segmentation]({{ "/images/2017-09-29-Felzenszwalb.png" | prepend: site.baseurl }})

이 영상 분할 방식은 완벽한 것은 아니지만, 연산 속도가 매우 빠르고 그에 비해서 결과가 좋은 편이기에 단순히 영역 제안의 보조역할로 사용되는 경우에는 적합하다고 판단되어 이 알고리즘을 선택한 것으로 보입니다.

---

### 영역 병합

대부분의 경우에는 Felzennszwalb 알고리즘을 사용하더라도 영역 제안에 사용하기에는 많은 영역이 얻어집니다. 따라서 이 영역을 적절히 병합할 필요성이 있습니다. 병합을 위해 이 논문에서는 유사도(similarity)라는 척도를 도입하여, 두 영역 간의 유사도를 평가한 다음에 유사도가 높은 영역끼리 차례대로, 계층 트리 형태로 병합해 갑니다. 유사도는 총 4가지 항목으로 평가됩니다.

 1. 색상 유사도: 두 영역 내의 픽셀들의 색상을 비교하여 유사도를 평가합니다.
 2. 텍스처 유사도: 두 영역 내의 특징(feature)의 분포를 비교하여 유사도를 평가합니다. SIFT 알고리즘을 사용합니다.
 3. 크기 유사도: 각 영역의 크기가 작을수록 유사도가 높습니다.
 4. 채움(fill) 유사도: 두 영역의 합 영역의 크기와 외접하는 직사각형의 넓이의 차가 작을수록 유사도가 높습니다.

3번 항목의 크기 유사도는 영역의 크기가 작을수록 우선적으로 병합되도록 유도하는 역할을 합니다. 또한 4번 항목의 채움 유사도는 병합 결과가 직사각형에 가까울수록 우선적으로 병합되도록 유도하는 역할을 합니다.

![union segmentation by level]({{ "/images/2017-09-29-union_segmentation.png" | prepend: site.baseurl }})

이 과정을 통해 여러 규모(scale)로 합병된 후보를 골라낼 수 있습니다.