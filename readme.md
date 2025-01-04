# Out-Of-Distribution Detection with Diversification (Provably) (NeurIPS'24)
This repo contains the code for our NeurIPS 2024 paper [Out-Of-Distribution Detection with Diversification (Provably)](https://arxiv.org/abs/2411.14049).

# Setup
In a conda env with pytorch / cuda available, run:
```bash
pip install -r requirements.txt
```
# Data Preparation
We follow the ATOM and OpenOOD to prepare the datasets. We provide links and instructions to download each dataset:
## ID datasets
CIFAR10 and CIFAR100: https://www.cs.toronto.edu/~kriz/cifar.html .
## Auxiliary OOD Datasets
- [Downsampled ImageNet Datasets](https://patrykchrabaszcz.github.io/Imagenet32/): we use the ImageNet64x64, which could be downloaded from [ImageNet Website](http://image-net.org/download-images). After downloading it, place it in this directory: `datasets/ImagenetRC`. 
## OOD Test Datasets
- [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`.
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
- [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365`.
- [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN`.
- [LSUN-resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN_resize`.
- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.

These datasets are used to evaluate OOD detection performance.
# Pipeline
## Training
```bash
python train_diverseMix.py --in-dataset CIFAR-10 --name diverseMix --seed 0
python train_diverseMix.py --in-dataset CIFAR-100 --name diverseMix --seed 0
```
## Evaluation
```bash
python eval_ood_detection.py --in_dataset CIFAR-10 --method energy --name diverseMix/0
python eval_ood_detection.py --in_dataset CIFAR-100 --method energy --name diverseMix/0
```
## Compute and Print Metrics
```bash
python compute_metrics.py --in-dataset CIFAR-10 --method energy --name diverseMix/0
python compute_metrics.py --in-dataset CIFAR-100 --method energy --name diverseMix/0
```

# Citation
Please cite our paper if you find the repo helpful in your work:
```
@inproceedings{yaoout,
  title={Out-Of-Distribution Detection with Diversification (Provably)},
  author={Yao, Haiyun and Han, Zongbo and Fu, Huazhu and Peng, Xi and Hu, Qinghua and Zhang, Changqing},
   booktitle={Advances in Neural Information Processing Systems},
   year={2024}
}
```

# Acknowledgements
The code is developed based on [ATOM](https://github.com/jfc43/informative-outlier-mining) and [OpenOOD](https://github.com/Jingkang50/OpenOOD/).