# Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate
<p align="left">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg" alt="License"></a>
</p>

## Overview

Generalization remains a central challenge in machine learning. In this work, we propose *Learning from Teaching* (**LoT**), a novel regularization technique for deep neural networks to enhance generalization. Inspired by the human ability to capture concise and abstract patterns, we hypothesize that generalizable correlations are expected to be easier to teach. LoT operationalizes this concept to improve the generalization of the main model with auxiliary student learners. The student learners are trained by the main model and improve the main model to capture more generalizable and teachable correlations by providing feedback. Our experimental results across several domains, including Computer Vision, Natural Language Processing, and Reinforcement Learning, demonstrate that the introduction of LoT brings significant benefits compared to merely training models on the original training data. It suggests the effectiveness of LoT in identifying generalizable information without falling into the swamp of complex patterns in data, making LoT a valuable addition to the current machine learning frameworks.

Code for the paper [Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate](https://arxiv.org/pdf/2402.02769.pdf).

Authors: [Can Jin](https://jincan333.github.io/), Tong Che, [Hongwu Peng](https://harveyp123.github.io/), Yiyuan Li, [Marco Pavone](https://web.stanford.edu/~pavone/index.html).

## 1. Install Requirements: 
```
conda create -n LoT python=3.9
conda activate LoT
pip install -r requirements.txt
```

## 2. Prepare Datasets:

To run the language modeling tasks, you can run the following code to download the WikiText-103 and the Penn Tree Bank (PTB) datasets. For other tasks, the datasets will be downloaded automatically.
```
bash getdata.sh
```

## 3. Configure WANDB

Configure WANDB USER_NAME and API_KEY in the key.config file.

## 4. Run LoT

### Reinforcement Learning
For Reinforcement Learning tasks, run the following command to implement experiments on BeamRider.
```
bash run/run_atari_games_LoT.sh
```
By changing env_id in the `run_atari_games_LoT.sh` file, you can run other games.

### Language Modeling
Run the following command for Transformer-XL on WikiText-103.
```
bash run/run_wikitext103_transformer_LoT.sh
```
Run the following command for Transformer-XL on PTB.
```
bash run/run_ptb_transformer_LoT.sh
```

## Contributors
Some of the code in this repository is based on the following amazing works.

* https://github.com/vwxyzjn/cleanrl
* https://github.com/hjc18/language_modeling_lstm
* https://github.com/kimiyoung/transformer-xl
* https://github.com/samuelstanton/gnosis


# Citation
If you find this work helpful, please consider citing our paper.
```bibtex
@misc{jin2024learning,
      title={Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate}, 
      author={Can Jin and Tong Che and Hongwu Peng and Yiyuan Li and Marco Pavone},
      year={2024},
      eprint={2402.02769},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}