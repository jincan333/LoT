# Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate
[![LICENSE](https://img.shields.io/badge/LICENSE-MIT-4caf50.svg)](https://github.com/jincan333/LoT)
[![arXiv](https://img.shields.io/badge/arXiv-2402.02769-b31b1b.svg)](https://arxiv.org/abs/2402.02769)


## Table of Contents

[**Overview**](#overview) | [**Requirements**](#install-requirements) | [**Datasets**](#prepare-datasets) | [**WANDB**](#configure-wandb) | [**Implementation**](#run-lot) | [**Contributor**](#contributors) | [**Citation**](#citation)


## Overview

Generalization remains a central challenge in machine learning. In this work, we propose *Learning from Teaching* (**LoT**), a novel regularization technique for deep neural networks to enhance generalization. Inspired by the human ability to capture concise and abstract patterns, we hypothesize that generalizable correlations are expected to be easier to teach. LoT operationalizes this concept to improve the generalization of the main model with auxiliary student learners. The student learners are trained by the main model and improve the main model to capture more generalizable and teachable correlations by providing feedback. Our experimental results across several domains, including Computer Vision, Natural Language Processing, and Reinforcement Learning, demonstrate that the introduction of LoT brings significant benefits compared to merely training models on the original training data. It suggests the effectiveness of LoT in identifying generalizable information without falling into the swamp of complex patterns in data, making LoT a valuable addition to the current machine learning frameworks.

Code for the paper [Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate](https://arxiv.org/pdf/2402.02769.pdf).

Authors: [Can Jin](https://jincan333.github.io/), Tong Che, [Hongwu Peng](https://harveyp123.github.io/), Yiyuan Li, [Marco Pavone](https://web.stanford.edu/~pavone/index.html).

## Install Requirements: 
```
conda create -n LoT python=3.9
conda activate LoT
pip install -r requirements.txt
```

## Prepare Datasets:

To run the language modeling tasks, you can run the following code to download the WikiText-103 and the Penn Tree Bank (PTB) datasets. For other tasks, the datasets will be downloaded automatically.
```
bash getdata.sh
```

## Configure WANDB

Configure WANDB USER_NAME and API_KEY in the key.config file.

## Run LoT

### Reinforcement Learning
For Reinforcement Learning tasks, run the following command to implement experiments on BeamRider.
```
bash run/run_atari_games_LoT.sh
```
By changing env_id in the `run_atari_games_LoT.sh` file, you can run other games.

### Language Modeling
Run the following command for Transformer-XL on WikiText-103.
```
bash run/run_transformer_wikitext103_LoT.sh
```
Run the following command for Transformer-XL on PTB.
```
bash run/run_transformer_ptb_LoT.sh
```
Run the following command for LSTM on WikiText-103.
```
bash run/run_lstm_wikitext103_LoT.sh
```
Run the following command for LSTM on PTB.
```
bash run/run_lstm_ptb_LoT.sh
```

### Image Classification
Run the following command for ResNet-20 on CIFAR100.
```
bash run/run_image_classification_LoT.sh
```
Change values for depth_list in the `run_image_classification_LoT.sh` file to alter models and change and change values for dataset to choose a different dataset.

## Contributors
Some of the code in this repository is based on the following amazing works.

* [CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms](https://github.com/vwxyzjn/cleanrl) (Huang et al., 2022)
* [Recurrent Neural Network Regularization](https://github.com/hjc18/language_modeling_lstm) (Zaremba et al., 2014)
* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://github.com/kimiyoung/transformer-xl) (Dai et al., 2018)
* [Does Knowledge Distillation Really Work?](https://github.com/samuelstanton/gnosis) (Stanton et al., 2021)


## Citation
We encourage citing our paper if our findings are used in your research.
```bibtex
@misc{jin2024learning,
      title={Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate}, 
      author={Can Jin and Tong Che and Hongwu Peng and Yiyuan Li and Marco Pavone},
      year={2024},
      eprint={2402.02769},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}