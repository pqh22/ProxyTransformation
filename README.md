<br>
<p align="center">
<h1 align="center"><strong>ProxyTransformation: Preshaping Point Cloud Manifold With Proxy Attention For 3D Visual Grounding</strong></h1>
  <p align="center">
    <a href='https://pqh22.github.io/' target='_blank'>Qihang Peng</a>&emsp;
    <a href='https://scholar.google.com/citations?view_op=list_works&hl=en&user=gZCggycAAAAJ' target='_blank'>Henry Zheng</a>&emsp;
    <a href='https://www.gaohuang.net/' target='_blank'>Gao Huang</a>&emsp;
    <br>
    Tsinghua University
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2502.19247-blue)](https://arxiv.org/abs/2502.19247)
[![](https://img.shields.io/badge/Paper-%F0%9F%93%96-blue)](https://arxiv.org/pdf/2502.19247)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-blue)](https://pqh22.github.io/projects/ProxyTransformation/index.html)

</div>

## 🔥 News
- \[2025-02\] We release the [paper](https://arxiv.org/pdf/2502.19247) of ProxyTransformation. Please check the [webpage](https://pqh22.github.io/projects/ProxyTransformation/index.html) for brief introduction!
- \[2025-02\] Our paper was accept by CVPR2025 ! 🥳



## ⭐ Motivation
<div style="text-align: center;">
    <img src="asset/illustrate_only_intutive.drawio.png" alt="Dialogue_Teaser" width=100% >
</div>

After reconstructing the scene point cloud from multi-view images in ego-centric 3D visual grounding, the noise in the reconstruction process and large-scale downsampling will cause the scene point cloud to lose a large amount of geometric and semantic information. Previous point cloud enhancement work was only based on a single point cloud modality. By enhancing the geometric structure through point cloud features, it did not make full use of the multi-modal information in this context. Moreover, these methods often require preprocessing, which does not meet our online requirements. Therefore, we hope to make full use of multi-modal information for point cloud enhancement, use text prompt and multi-view image to generate corresponding transformations, and conduct partition optimization on the scene point cloud structure.  

## 📖 Framework
<div style="text-align: center;">
    <img src="asset/pt.png" alt="Dialogue_Teaser" width=100% >
</div>

In ego-centric 3D visual grounding, we first generate a uniform grid prior in space and perform an initial clustering. Each cluster is then processed by an offset network to obtain deformable offsets for the cluster centers, allowing the initial grid prior to be shifted toward more important regions and enabling clustering to capture the sub-manifold of the target region. We utilize a proxy block based on proxy attention to process multi-modal information, obtaining a transformation matrix and translation vector for each sub-manifold. This optimizes the relative positions and internal structures of the sub-manifolds, which are subsequently fed into downstream structures for feature learning and fusion, ultimately achieving precise localization of the target object in the scene.

## 📝 TODO List


- \[ \] Clean up the codebase and release our code.
- \[ \] Upload our model weights.
- \[ \] Full release and further updates.


## 📚 Getting Started

### Installation

1. Clone this repository.

```bash
git clone https://github.com/pqh22/ProxyTransformation.git
cd ProxyTransformation
```

2. Create environment and install packages

```bash
conda create -n pt python=3.8 -y
conda activate pt
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
python install.py all
```

By the way, it's easy to encounter problems when installing Mink Engine and PyTorch3D. You can follow the official document carefully.

### Data Preparation
Please refer to [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) for downloading and organization.

And the enhanced detection model and augmented text for training are proposed by [DenseG](https://opendrivelab.github.io/Challenge%202024/multiview_THU-LenovoAI.pdf). If you are interested in obtaining the data for your own research or experiments, please get in touch with the authors of the relevant work.

### Training and evaluation

We provide some scripts for your reference.

To train our grounding model with pytorch, you can run:

```bash
torchrun --nproc_per_node={NUM_NODES} tools/train.py {CONFIG} --work-dir={WORK_DIR} --launcher="pytorch"
```

To inference and evaluate the model, you can run:

``` bash
torchrun --nproc_per_node={NUM_NODES} tools/eval.py {CONFIG} --work-dir={WORK_DIR}  --launcher="pytorch" --resume {CKPT}
```


## 📦 Model & Weights
You can download our model from my [Google Drive](https://drive.google.com/drive/folders/1dPiaKuldbIn-OX9obhOlem7SCZeGnkt-?hl=zh_CN).

## 📬 Bugs or questions?

If you have any questions related to the codes or the paper, please feel free to contact Qihang Peng (`pqh22@mails.tsinghua.edu.cn`) or open an issue.

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{peng2025proxytransformationpreshapingpointcloud,
      title={ProxyTransformation: Preshaping Point Cloud Manifold With Proxy Attention For 3D Visual Grounding}, 
      author={Qihang Peng and Henry Zheng and Gao Huang},
      year={2025},
      eprint={2502.19247},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.19247}, 
}
```

## 👏 Acknowledgements
The development of ProxyTransformation is based on [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) and [DenseG](https://opendrivelab.github.io/Challenge%202024/multiview_THU-LenovoAI.pdf). We deeply appreciate their contribution to the community.