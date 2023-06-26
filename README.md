# Rethinking Federated Learning with Domain Shift: A Prototype View

> Rethinking Federated Learning with Domain Shift: A Prototype View,            
> Wenke Huang, Mang Ye, Zekun Shi, He Li, Bo Du
> *CVPR, 2023*
> [Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)

## News
* [2022-03-16] Code has released. Digits [Link](https://drive.google.com/drive/folders/1SSv9dqQPBGyHS3rSwoFKmpBIeF4GX-i6?usp=sharing)
* [2022-03-06] Repo created. Paper and code will come soon.

## Abstract
Federated learning shows a bright promise as a privacy-preserving collaborative learning technique. However, prevalent solutions mainly focus on all private data sampled from the same domain. An important challenge is that when distributed data are derived from diverse domains. The private model presents degenerative performance on other domains (with domain shift). Therefore, we expect that the global model optimized after the federated learning process stably provides generalizability performance on multiple domains. In this paper, we propose Federated Prototypes Learning (FPL) for federated learning under domain shift. The core idea is to construct cluster prototypes and unbiased prototypes, providing fruitful domain knowledge and a fair convergent target. On the one hand, we pull the sample embedding closer to cluster prototypes belonging to the same semantics than cluster prototypes from distinct classes. On the other hand, we introduce consistency regularization to align the local instance with the respective unbiased prototype. Empirical results on Digits and Office Caltech tasks demonstrate the effectiveness of the proposed solution and the efficiency of crucial modules.

## Citation
```
@inproceedings{HuangFPL_CVPR2023,
    author    = {Huang, Wenke and Mang, Ye and Shi, Zekun and Li, He and Bo, Du},
    title     = {Rethinking Federated Learning with Domain Shift: A Prototype View},
    booktitle = {CVPR},
    year      = {2023}
}
```

## Relevant Projects
[1] Learn from Others and Be Yourself in Heterogeneous Federated Learning - CVPR 2022 [[Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)][[Code](https://github.com/WenkeHuang/FCCL)]

[2] Federated Graph Semantic and Structural Learning - IJCAI 2023 [[Link](https://marswhu.github.io/publications/files/FGSSL.pdf)][[Code](https://github.com/wgc-research/fgssl)]

