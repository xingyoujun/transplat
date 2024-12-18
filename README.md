<p align="center">
  <h1 align="center">TranSplat: Generalizable 3D Gaussian Splatting <br> from Sparse Multi-View Images with Transformers</h1>
  <p align="center">
    <a href="https://xingyoujun.github.io/">Chuanrui Zhang *</a>
    &nbsp;·&nbsp;
    <a href="https://heiheishuang.xyz/">Yingshuang Zou *</a>
    &nbsp;·&nbsp;
    <a href="https://lizhuoling.github.io/">Zhuoling Li</a>
    &nbsp;·&nbsp;
    <a>Minmin Yi</a>
    &nbsp;·&nbsp;
    <a>Haoqian Wang †</a>
  </p>
  <h3 align="center">AAAI 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2408.13770">Paper</a> | <a href="https://xingyoujun.github.io/transplat">Project Page</a> | <a href="https://huggingface.co/xingyoujun/transplat">Pretrained Models</a> </h3>
<!--   <div align="center">
    <a href="https://news.ycombinator.com/item?id=41222655">
      <img
        alt="Featured on Hacker News"
        src="https://hackerbadge.vercel.app/api?id=41222655&type=dark"
      />
    </a>
  </div> -->

<!-- <ul>
<li><b>08/11/24 Update:</b> Explore our <a href="https://github.com/donydchen/mvsplat360">MVSplat360 [NeurIPS '24]</a>, an upgraded MVSplat that combines video diffusion to achieve 360° NVS for large-scale scenes from just 5 input views! </li>  
<li><b>21/10/24 Update:</b> Check out Haofei's <a href="https://github.com/cvg/depthsplat">DepthSplat</a> if you are interested in feed-forward 3DGS on more complex scenes (DL3DV-10K) and more input views (up to 12 views)!</li>
</ul>
<br>
</p> -->

## Installation

**a. Create a conda virtual environment and activate it.**

```bash
conda create --name transplat -y python=3.10.14
conda activate transplat
conda install -y pip
```

**b. Install PyTorch and torchvision.**

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# Recommended torch==2.1.2
```

**c. Install mmcv.**

```bash
pip install openmim
mim install mmcv==2.1.0
```

**d. Install other requirements.**
```bash
pip install -r requirements.txt
```

## Acquiring Datasets

### RealEstate10K and ACID

We use the same training datasets as pixelSplat and MVSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

### DTU (For Testing Only)

We use the same testing datasets as MVSplat. Below we quote MVSplat's [detailed instructions](https://github.com/donydchen/mvsplat/tree/main?tab=readme-ov-file#dtu-for-testing-only) on getting datasets.

> * Download the preprocessed DTU data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).
> * Convert DTU to chunks by running `python src/scripts/convert_dtu.py --input_dir PATH_TO_DTU --output_dir datasets/dtu`
> * [Optional] Generate the evaluation index by running `python src/scripts/generate_dtu_evaluation_index.py --n_contexts=N`, where N is the number of context views. (For N=2 and N=3, we have already provided our tested version under `/assets`.)

## Running the Code

### Evaluation

For inference, first prepare pretrained models.

* get the [pretrained models](https://huggingface.co/xingyoujun/transplat) of transplat, and save them to `/checkpoints`

* get the [pretrained models](https://github.com/DepthAnything/Depth-Anything-V2/tree/main?tab=readme-ov-file#pre-trained-models) of Depth-Anything-V2-Base, and save them to `/checkpoints`

* run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=./checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true 

# acid
python -m src.main +experiment=acid \
checkpointing.load=./checkpoints/acid.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true 
```

* the rendered novel views will be stored under `outputs/test`

You can find more running commands (eg. Cross-Dataset Generalization) in [run.sh](run.sh).

### Training

Run the following:

```bash
# download the backbone pretrained weight from unimatch and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
# train mvsplat
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 python -m src.main +experiment=re10k data_loader.train.batch_size=2 wandb.mode=run wandb.name=transplat-re10k 2>&1 | tee transplat-re10k.log
```

Our models are trained with 7 RTX3090 (24GB) GPU.

## BibTeX

```bibtex
@article{zhang2024transplat,
  title={Transplat: Generalizable 3d gaussian splatting from sparse multi-view images with transformers},
  author={Zhang, Chuanrui and Zou, Yingshuang and Li, Zhuoling and Yi, Minmin and Wang, Haoqian},
  journal={arXiv preprint arXiv:2408.13770},
  year={2024}
}
```

## Acknowledgements

The project is largely based on [MVSplat](https://github.com/donydchen/mvsplat) and has incorporated numerous code snippets from [UniMatch](https://github.com/autonomousvision/unimatch), Depth-Anything-V2 from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) and transformer architecture from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Many thanks to these four projects for their excellent contributions!
