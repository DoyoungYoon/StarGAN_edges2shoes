# StarGAN_edges2shoes
스케치 이미지로부터 다양한 제품 디자인을 자동 생성하는 기법

* 이 코드는 저자의 코드를 수정하여 구현하였습니다. https://github.com/yunjey/stargan




## 1. Abstracts
디자이너가 제품을 디자인할 때, 구상한 스케치 도면을 그리고 그것을 바탕으로 제품 이미지를 렌더링하게 된다. 단순히 스케치 이미지만으로는 실제 제품의 느낌을 얻기 어려우므로 여러 번의 시행착오를 겪게 된다. 이러한 문제 때문에 렌더링 작업은 많은 자원을 소모하게 된다. 이러한 문제를 해결하고자 컴퓨터가 먼저 렌더링 이미지를 생성하여 제품 이미지를 보여준다면 시행착오를 줄일 수 있을 것이다. 현재 이와 관련한 기계학습 기반 제품 이미지를 생성해주는 방법들이 있으나 한 가지 스타일만 생성하거나 원하는 스타일의 이미지를 입력 이미지로 넣어줘야 하는 등의 제한점이 있다. 본 연구는 이런 기존 연구들의 제한점을 보완하고자 StarGAN 모델을 변형하여 디자이너의 스케치 이미지와 원하는 도메인을 선택하여 원하는 제품 이미지를 생성할 수 있는 기법을 제안하고자 한다.


## 2. Result
<img src="https://user-images.githubusercontent.com/30281089/48409693-e85be700-e77f-11e8-93f6-553a7e1fa525.png" width="60%"></img>
<img src="https://user-images.githubusercontent.com/30281089/48409742-0d505a00-e780-11e8-9e79-efa09ea60f70.png" width="60%"></img>

## 3. Train
* edges2shoes
<pre><code>>python main.py --mode train --dataset edges2shoes --image_size 136 --c_dim 5</code></pre>

