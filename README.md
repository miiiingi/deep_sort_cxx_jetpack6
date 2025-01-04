# DeepSort_TensorRT
deepsort 객체 추적 알고리즘을 TensorRT를 통해서 가속화하는 코드입니다.
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
git clone git@github.com:miiiingi/deepsort-tensorrt-JetPack6.git
```
먼저, 본 레포지토리의 exportOnnx.py 파일을 복사하여 클론한 deep_sort_pytorch 레포지토리에 복사합니다.   
deep_sort_pytorch 레포지토리에서 exportOnnx.py을 실행하여 deepsort.onnx 파일을 얻습니다.  
그리고 deepsort.onnx 파일을 본 레포지토리의 resources 폴더에 놓습니다. 그 후, 본 레포지토리에 대한 빌드를 진행합니다.   
```
mkdir build
cd build
cmake ..
make
./onnx2engine ../resources/deepsort.onnx ../resources/deepsort.engine
```
아래의 코드로 데모를 실행할 수 있습니다. 본 데모는 검은 바탕위에 객체 추적 결과의 바운딩 박스가 표현됩니다.
```
./demo ../resources/deepsort.engine ../resources/track.txt
```
