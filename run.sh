#!/bin/bash

# 명령행 인자가 1개인지 확인 (NDS만 받기 때문)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <NDS>"
    exit 1
fi

# 하드코딩된 경로
engine_path="../resources/deepsort.engine"
video_path="../resources/demo2.mp4"
onnx_path="../resources/yolov8s.onnx"

# 명령행 인자에서 NDS 값 추출
nds=$1  # NDS 값

# 추출한 값을 출력 (실제로는 이 값을 사용하여 로직을 수행)
echo "Engine Path: $engine_path"
echo "Video Path: $video_path"
echo "ONNX Path: $onnx_path"
echo "NDS: $nds"

 ./demo $engine_path $video_path $onnx_path $nds