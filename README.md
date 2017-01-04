# FastMask

FastMask: Segment Multi-scale Object Candidates in One Shot

[Hexiang Hu\*](http://hexianghu.com), Shiyi Lan\*, Yuning Jiang, Zhimin Cao, Fei Sha
(\*Equal contribution. Work was done during their internships at Megvii Inc.)

 If you are using code or other related resources from this repository, please cite the following paper:
```
@article{hu2016fastmask,
  title={FastMask: Segment Multi-scale Object Candidates in One Shot},
  author={Hu, Hexiang and Lan, Shiyi and Jiang, Yuning and Cao, Zhimin and Sha, Fei},
  journal={arXiv preprint arXiv:1612.08843},
  year={2016}
}
```

# Requirements and Dependencies
- MAC OS X or Linux
- NVIDIA GPU with compute capability 3.5+
- Python 2.7+
- [COCOApi](https://github.com/pdollar/coco), redis, python-cjson, opencv2, numpy
- [Alchemy](https://github.com/voidrank/alchemy), [caffe-fm](https://github.com/voidrank/caffe-fm)

# Quick Start

## Step in common

We highly recommend you to use [anaconda2](https://www.continuum.io/downloads) on ubuntu 14.04,
which is our main experimental environment.

For ubuntu 14.04
```
sudo apt-get update
sudo apt-get install python-opencv python-pip
```

and then install cocoapi see [COCOApi](https://github.com/pdollar/coco)

```
git clone --recursive https://github.com/voidrank/FastMask
cd FastMask
mkdir params results
pip install -r requirements.txt
cd caffe-fm
make pycaffe -j 4
cd ..
```

## Demonstrate

### Download parameters of pretrained models

Download [final model](https://drive.google.com/file/d/0B91BSyN61NHRS3Y3UEl1LVE5MjQ/view?usp=sharing) and save it in ./params

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

### Download COCO

```
cd data
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip train.zip instances_train_val2014.zip val2014.zip
cd ..
```


### Download pretrained model on imagenet

Download [ResNet-50-model.caffemodel](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777) and save it in ./params

### Training
```
python train.py [gpu_id] [model] [--init_weights=ResNet-50-model.caffemodel] [--process=4]
```

For examples,

```
python train.py 0 fm-res39 --init_weights=ResNet-50-model.caffemodel
```


## Evaluation

```
python test.py [gpu_id] [model] [--init_weights=xxx.caffemodel]
```
```
python evalCOCO.py [model]
```
