#!/bin/bash

# 명령행 인자가 1개인지 확인 (NDS만 받기 때문)
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <video_path> <onnx_path> <NDS> <width> <height>"
    exit 1
fi

engine_path="../resources/deepsort.engine"

video_path=$1
onnx_path=$2
nds=$3
width=$4
height=$5

# 추출한 값을 출력 (실제로는 이 값을 사용하여 로직을 수행)
echo "Engine Path: $engine_path"
echo "Video Path: $video_path"
echo "ONNX Path: $onnx_path"
echo "NDS: $nds"
echo "Width: $width"
echo "Height: $height"

 ./demo $engine_path $video_path $onnx_path $nds $width $height