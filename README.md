# IJCV 2024: Winning Prize Comes from Losing Tickets: Improve Invariant Learning by Exploring Variant Parameters for Out-of-Distribution Generalization

Zhuo Huang<sup>1</sup>, Muyang Li<sup>1</sup>, Li Shen<sup>2</sup>, Jun Yu<sup>3</sup>, Chen Gong<sup>4</sup>, Bo Han<sup>5</sup>, Tongliang Liu<sup>1</sup>

<sup>1</sup>The University of Sydney, <sup>2</sup>JD Explore Academy, <sup>3</sup>University of Science and Technology of China, <sup>4</sup>Nanjing University of Science and Technology, <sup>5</sup>Hong Kong Baptist University


# Overview
Out-of-Distribution (OOD) Generalization aims to learn robust models that generalize well to various environments without fitting to distribution-specific features. Recent studies based on Lottery Ticket Hypothesis (LTH) address this problem by minimizing the learning target to find some of the parameters that are critical to the task. However, in OOD problems, such solutions are suboptimal as the learning task contains severe distribution noises, which can mislead the optimization process. Therefore, apart from finding the task-related parameters (\textit{i.e.}, invariant parameters), we propose \textbf{Exploring Variant parameters for Invariant Learning (EVIL)} which also leverages the distribution knowledge to find the parameters that are sensitive to distribution shift (\textit{i.e.}, variant parameters). Once the variant parameters are left out of invariant learning, a robust subnetwork that is resistant to distribution shift can be found. Additionally, the parameters that are relatively stable across distributions can be considered invariant ones to improve invariant learning. By fully exploring both variant and invariant parameters, our EVIL can effectively identify a robust subnetwork to improve OOD generalization. In extensive experiments on integrated testbed: DomainBed, EVIL can effectively and efficiently enhance many popular methods, such as ERM, IRM, SAM, etc.


![Overview](imgs/framework.png )



## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for our study.

```
Python: 3.8.6
PyTorch: 1.7.0+cu92
Torchvision: 0.8.1+cu92
CUDA: 9.2
CUDNN: 7603
NumPy: 1.19.4
PIL: 8.0.1
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /my/datasets/path
```

## License

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.


## Reference

<br> **📑 If you find our paper and code helpful for your research, please consider citing:** <br>
```
@article{huang2023winning,
  title={Winning Prize Comes from Losing Tickets: Improve Invariant Learning by Exploring Variant Parameters for Out-of-Distribution Generalization},
  author={Huang, Zhuo and Li, Muyang and Shen, Li and Yu, Jun and Gong, Chen and Han, Bo and Liu, Tongliang},
  journal={arXiv preprint arXiv:2310.16391},
  year={2023}
}
```

If you have any problems, please feel free to raise an issue or directly contact [zhuohuang.ai@gmail.com](zhuohuang.ai@gmail.com).
