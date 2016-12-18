# FastMask

FastMask: Segment Multi-scale Object Candidates in One Shot

Hexiang Hu\*, Shiyi Lan\*, Yuning Jiang, Zhimin Cao, Fei Sha  
(\*Equal contribution. Work was done when Hexiang Hu was interns at Megvii)

(Here comes the information for citation)

# Requirements and Dependencies
- MAC OS X or Linux
- NVIDIA GPU with compute capability 3.5+
- [COCOApi](https://github.com/pdollar/coco), redis, python-cjson, opencv2, numpy
- [Alchemy](https://github.com/voidrank/alchemy), [caffe-fm](https://github.com/voidrank/caffe-fm)

# Quick Start

## Step in common


For ubuntu 14.04
```
sudo apt-get install python-opencv
```

```
git clone --recursive https://github.com/voidrank/FastMask
cd FastMask
mkdir params results
pip install requirements.txt
cd caffe-fm
make pycaffe -j 4
cd ..
```

## Demonstrate

### Download parameters of pretrained models

(here comes the download link of pretrained model)

### Image Demo
```
python image_demo.py [gpu_id] [model] [input_image_path] [--init_weights=weights] [--threshold=0.9]
```

This instruction will segment the image at `[input_image_path]` with `models/[model].test.prototxt` and `params/[weights].caffemodel`

There is an example: (suppose that the input image named `input.jpg` is at `./`)

```
python image_demo.py 0 fm-res39 input.jpg --init_weights=fm-res39_final_params.caffemodel
```

### Video Demo

```
python video_demo.py [gpu_id] [model] [input_video_path] [output_video_path] [--init_weights=weights] [--threshold=0.9]
```

## Training

### Set Up Redis Server for Multiprocess Communication
```
nohup redis-server redis.conf
```

### Download pretrained model on imagenet

(here comes the download link of pretrained model)

### Training
```
python train.py [gpu_id] [model] [--init_weights=ResNet-50-model.caffemodel] [--process=4]
```
