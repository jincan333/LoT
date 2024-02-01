# Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate

# Overview

Generalization remains a central challenge in machine learning. In this work, we propose *Learning from Teaching* (**LoT**), a novel regularization technique for deep neural networks to enhance generalization. Inspired by the human ability to capture concise and abstract patterns. we hypothesize that generalizable correlations are expected to be easier to teach. LoT operationalize this concept to improve the generalization of the main model with auxiliary student learners. The student learners are trained by the main model and provide feedback to the main model by capturing more generalizable and teachable correlations. Our experimental results across several domains, including Computer Vision, Natural Language Processing, and Reinforcement Learning, demonstrate that the introduction of LoT brings significant benefits compared to merely training models on the original training data. It suggests the effectiveness of LoT in identifying generalizable information without falling into the swamp of complex patterns in data, making LoT a valuable addition to the current machine learning frameworks.

# 1. Install Requirements: 
```
conda create -n LoT python=3.9
conda activate LoT
pip install -r requirements.txt
```

# 2. Prepare Datasets:

To run the language modeling tasks, you can run the following code to download the WikiText-103 and the Penn Tree Bank (PTB) datasets. For other tasks, the datasets will be downloaded automatically.
```
bash getdata.sh
```

# 3. Configure WANDB

Configure WANDB USER_NAME and API_KEY in the key.config file.

# 4. Run LoT

## Reinforcement Learning
For Reinforcement Learning tasks, run the following command to implement experiments on BeamRider.
```
bash run/run_atari_games_LoT.sh
```
By changing env_id in the `run_atari_games_LoT.sh` file, you can run other games.

## Language Modeling
Run the following command for Transformer-XL on WikiText-103.
```
bash run/run_wikitext103_transformer_LoT.sh
```
Run the following command for Transformer-XL on PTB.
```
bash run/run_ptb_transformer_LoT.sh
```