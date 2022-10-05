# 2D Temporal Action Recognition

## About
First place (after video presenation) solution for IndabaX South Africa 2021 hackathon
["Autonomous Shopper Prediction by Cape AI"](https://zindi.africa/competitions/indabax-south-africa-2021).

Hosted by an autonmous shopping venture, the challenge focused on using CV to determine if an item had been taken off a shelf given a sequence skeletal keypoints from a customer’s shopping session. 

We were inspired by SOTA skeleton-based action recognition techniques that rely on the inherent graph structure of human skeletons, whereby each joint is represented as a node, and bones as edges. For this competition, we implement variants of the [CTR-GCN architecture](https://arxiv.org/abs/2107.12213). This network relies on learnt and fixed graph topology (represented as adjacency matrices) to condition an aggregation of standard temporal convolutional layers. Unlike the base implementation which simply normalises and flattens the last activation map (to be fed into a linear classification head), we leverage the temporal nature of these output features by using:

1) A multi-head attention layer to weight features from specific instances in time more relevant to the action and; 
2) A transformer encoder network to produce a single context-rich feature vector.

Model 1's public test AUC was 0.878 (private 0.719)
Model 2's public test AUC was 0.853 (private 0.704)

The ensemble of these two models (and our final solution) resulted in a public test AUC of 0.881 and a private AUC of 0.712. Expected notebook runtime (training on a system with a GTX 1070ti, Ryzen 5 5600G and 16GB RAM) is 1-1.5 hours.

## Getting Started

Setup your environment.

```bash
conda env create -f environment.yml
conda activate masters
```

Place `Train.csv`, `Train_Targets.csv` and `Test.csv` in your root (download [here](https://zindi.africa/competitions/indabax-south-africa-2021/data)) and `main.ipynb` from start to finish.

The following files and directories will be created:
- ctr-gcn-skip-mini-att.csv
- ctr-gcn-skip-mini-trans.csv
- ensamble.csv
- dataloader_test.pt
- dataloader_train.pt
- dataloader_val.pt
- test_dataset_ex.pkl
- train_dataset_ex.pkl
- val_dataset_ex.pkl
- lightning_logs/

`lightning_logs/` will contain tensorboard logs for model 1 and model 2 (in directories `../version_0` and `../version_1` respectively) as well as checkpointed models. We use the last model for our preds (`last.ckpt`), but we also save the model which achieved the highest val AUC.

That's all!

## Refrences

> Yan, Sijie, Yuanjun Xiong, and Dahua Lin. "Spatial temporal graph convolutional networks for skeleton-based action recognition." Proceedings of AAAI conference on artificial intelligence. 2018.

> Chen, Yuxin, et al. "Channel-wise topology refinement graph convolution for skeleton-based action recognition." Proceedings of the IEEE/CVF international conference on computer vision. 2021.



Authors: Geoffrey Frost, [Kevin Eloff](https://github.com/KevinEloff), [Matthew Baas](https://github.com/RF5)
