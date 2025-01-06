# DeepSort_C++
deepsort 객체 추적 알고리즘을 C++언어로 실행시키는 코드입니다. 
```
https://github.com/RichardoMrMu/deepsort-tensorrt.git
```
위의 레포지토리를 기반으로 하고있으며, TensorRT 버전 변경에 따른 코드 수정 및 원하는 영상에 객체 추적 알고리즘을 적용할 수 있도록 수정을 진행하였습니다. 
## Compatibility
본 코드는 JetPack6.1에서 사용할 수 있도록 작성하였으며, 환경 설정과 관련된 부분은 아래와 같습니다.  
<div align=center>
  <a href="s">
    <img align="center" src="https://github.com/user-attachments/assets/ba7e45e6-09d5-48d7-a56f-af2c53c40984"/>
  </a>
</div>  

## Build & Run
아래의 레포지토리를 클론합니다. 첫 번째 레포지토리는 deepsort를 pytorch로 구현한 레포지토리입니다.  
이 레포지토리에서 deepsort 모델의 가중치를 가져올 것입니다.
```
git clone git@github.com:ZQPei/deep_sort_pytorch.git
git clone git@github.com:miiiingi/deep_sort_cxx_jetpack6.git
```
먼저, 본 레포지토리의 exportOnnx.py 파일을 복사하여 클론한 deep_sort_pytorch 레포지토리에 복사합니다.   
deep_sort_pytorch 레포지토리에서 exportOnnx.py을 실행하여 deepsort.onnx 파일을 얻습니다.  
그리고 deepsort.onnx 파일을 본 레포지토리의 resources 폴더에 놓습니다. 그 후, 본 레포지토리에 대한 빌드를 진행합니다.   
```
mkdir build
cd build
bash ../build.sh
bash ../convert.sh
```
객체 탐지 모델의 사용을 위해 ultralytics 라이브러리를 사용합니다. 먼저 아래의 명령어를 통해 ultralytics 라이브러리를 다운로드 받습니다.
```
pip install ultralytics
```
아래의 명령어를 통해서 yolov8s.onnx 파일을 생성합니다.
```
yolo export model=yolov8s.pt imgsz=1080,1920 format=onnx
```
생성된 yolov8s.onnx 파일을 resources 폴더에 옮깁니다.
아래의 bash shell을 통해서 demo를 실행시킵니다.

```
bash ../run.sh
```

## 초기 결과 기록

DeepSort를 사용하지 않은 경우  

<div align=center>
  <video width="640" height="360" controls>
    <source src="https://github.com/miiiingi/deep_sort_cxx_jetpack6/blob/dev/5/result/NotusingDS.mp4" type="video/mp4">
  </video>
</div>

DeepSort를 사용한 경우  

<div align=center>
  <video width="640" height="360" controls>
    <source src="https://github.com/miiiingi/deep_sort_cxx_jetpack6/blob/dev/5/result/usingDS.mp4" type="video/mp4">
  </video>
</div>