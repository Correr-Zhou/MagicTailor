# MagicTailor: Component-Controllable Personalization in Text-to-Image Diffusion Models

[![Page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoColor=white)](https://correr-zhou.github.io/MagicTailor/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2410.13370)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-%233_Paper_of_the_Day-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/papers?date=2024-10-21)
[![News](https://img.shields.io/badge/Neuronad-News-980e5a?logo=googlechrome&logoColor=white)](https://neuronad.com/ai-news/tech/magictailor-personalization-in-text-to-image-generation/)

[Donghao Zhou](https://scholar.google.com/citations?hl=en&user=RsLS11MAAAAJ)<sup>1*</span></sup>,
[Jiancheng Huang](https://huangjch526.github.io/)<sup>2*</span></sup>,
[Jinbin Bai](https://noyii.github.io/)<sup>3</sup>,
[Jiaze Wang](https://jiazewang.com/)<sup>1</sup>,
[Hao Chen](https://scholar.google.com.hk/citations?user=tT03tysAAAAJ&hl=zh-CN)<sup>1</sup>,
[Guangyong Chen](https://guangyongchen.github.io/)<sup>4</sup>,<br>
[Xiaowei Hu](https://xw-hu.github.io/)<sup>5&dagger;</sup>,
[Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/)<sup>1</sup>

<span class="author-block"><sup>1</sup>CUHK &nbsp;</span>
<span class="author-block"><sup>2</sup>SIAT, CAS &nbsp;</span>
<span class="author-block"><sup>3</sup>NUS &nbsp;</span>
<span class="author-block"><sup>4</sup>Zhejiang Lab &nbsp;</span>
<span class="author-block"><sup>5</sup>Shanghai AI Lab</span>

<br>

![teaser](assets/teaser.gif)

We present **MagicTailor** to enable **component-controllable personalization**, a newly formulated task aiming to reconfigure specific components of concepts during personalization.

<details>
  <summary>Abstract</summary>
  <p>Recent advancements in text-to-image (T2I) diffusion models have enabled the creation of high-quality images from text prompts, but they still struggle to generate images with precise control over specific visual concepts. Existing approaches can replicate a given concept by learning from reference images, yet they lack the flexibility for fine-grained customization of the individual component within the concept. In this paper, we introduce component-controllable personalization, a novel task that pushes the boundaries of T2I models by allowing users to reconfigure and personalize specific components of concepts. This task is particularly challenging due to two primary obstacles: semantic pollution, where unwanted visual elements corrupt the personalized concept, and semantic imbalance, which causes disproportionate learning of visual semantics. To overcome these challenges, we design MagicTailor, an innovative framework that leverages Dynamic Masked Degradation (DM-Deg) to dynamically perturb undesired visual semantics and Dual-Stream Balancing (DS-Bal) to establish a balanced learning paradigm for visual semantics. Extensive comparisons, ablations, and analyses demonstrate that MagicTailor not only excels in this challenging task but also holds significant promise for practical applications, paving the way for more nuanced and creative image generation.</p>
</details>


## 🔥 Updates
- 2024.10: Our code is released! Feel free to [contact me](mailto:dhzhou@link.cuhk.edu.hk) if anything is unclear.
- 2024.10: [Our paper](https://arxiv.org/pdf/2410.13370) is available. The code is coming soon!


## 🛠️ Installation
1. Install the conda environment:
```
conda env create -f environment.yml
```
2. Install other dependencies (here we take CUDA 11.6 as an example):
```
conda activate magictailor
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
3. Clone the Grounded-SAM repository:
```
cd {PATH-TO-THIS-CODE}
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
```
4. Follow the section of ["Install without Docker"](https://github.com/IDEA-Research/Grounded-Segment-Anything) to setup Grounded-SAM (please make sure that the CUDA version used for the installation here is the same as that of PyTorch).

> ❗You can skip Step 3 and 4 if you just want to have a quick try using the example images we provide.

## 🔬 Training and Inference

### Preparing Data
Directly use the example images in `./examples`, or you can prepare your own data:
1. Create a folder named `{CONCEPT}_{ID}+{COMPONENT}_{ID}`, where `{CONCEPT}` and `{COMPONENT}` are the category names for the concept and component respectively, and `{ID}` is the customized index (you can set it to whatever you want) that helps you distinguish.
2. Put the reference images into this folder, and rename them as `0_{CONCEPT}_{ID}.jpg` and `1_{COMPONENT}_{ID}.jpg` for the images of the concept and component respectively.
3. Finally, the data will be organized like:
```
person_a+hair_a/
├── 0_person_a0.jpg
├── 0_person_a1.jpg
├── 0_person_a2.jpg
├── 1_hair_a0.jpg
├── 1_hair_a1.jpg
└── 1_hair_a2.jpg
```

### Training
You can train MagicTailor with default hyperparameters:
```
python train.py --instance_data_dir {PATH-TO-PREPARED-DATA}
```
For example:
```
python train.py --instance_data_dir examples/person_k+hair_c
```
> ❗Please check the quality of the masks output by Grounded-SAM to ensure that the model runs correctly.

Alternatively, you can also train it with customized hyperparameters, just like:
```
python train.py 
    --instance_data_dir examples/person_k+hair_c \
    --phase1_train_steps 200 \
    --phase2_train_steps 300 \
    --phase1_learning_rate 1e-4 \
    --phase2_learning_rate 1e-5 \
    --lora_rank 32 \
    --alpha 0.5 \
    --gamma 32 \
    --lambda_preservation 0.2
```
You can refer to [our paper](https://arxiv.org/pdf/2410.13370) or `train.py` to understand the meaning of the arguments.
Adjusting these hyperparameters helps yield better results.

Moreover, we also provide a detailed training script in `scripts/train.sh` for reserach or development purpose, supporting further modfication.

### Inference
After training, a model will be saved in `outputs/magictailor`. Placeholder tokens `<v0>` and `<v1>` will be assigned to the concept and componet respectively for text-to-image geneartion.

Then, you can genearte images with the saved model, just like:
```
python inference.py \
  --model_path "outputs/magictailor" \
  --prompt "<v0> with <v1>, on the beach" \
  --output_path "outputs/inference/result.jpg"
```


## 📑 Citation
If you find that our work is helpful in your research, please consider citing our paper:
```latex
@article{zhou2024magictailor,
  title={MagicTailor: Component-Controllable Personalization in Text-to-Image Diffusion Models},
  author={Zhou, Donghao and Huang, Jiancheng and Bai, Jinbin and Wang, Jiaze and Chen, Hao and Chen, Guangyong and Hu, Xiaowei and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2410.13370},
  year={2024}
} 
```

## 🤝 Acknowledgement
Our code is built upon the repositories of [diffusers](https://github.com/huggingface/diffusers) and [Break-A-Scene](https://github.com/google/break-a-scene/). Thank their authors for their excellent work.

<p align="center">
  <a href="https://star-history.com/#Correr-Zhou/MagicTailor&Date">
    <img src="https://api.star-history.com/svg?repos=Correr-Zhou/MagicTailor&type=Date" alt="Star History Chart">
  </a>
</p>