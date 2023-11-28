This is the repository for the paper, ["Revisiting pre-trained remote sensing model
benchmarks: resizing and normalization matters"](https://arxiv.org/abs/2305.13456)

### TL;DR

In our experiments evaluating numerous pretrained models downstream remote sensing tasks, we find that to achieve the best performance and perform fair comparisons it is important to do the following:

- **Resize**: Small satellite and/or aerial image patches (32x32, 64x64) have a negative performance impact for ResNet models. Simply resizing to the original training or pretraining image size, e.g. 224x224, achieves significant improvements on downstream performance.
- **Normalize**: When comparing to models pretrained using standard normalization, e.g. ImageNet pretrained models, standard normalization
- **Compare to a Strong Baseline**: We find that a simple unsupervised baseline of computing channelwise statistics as features outperforms several methods pretrained on large-scale remote sensing datasets.
- **Prefer K-Nearest Neighbors over Linear Probing & Fine-tuning**: Linear Probing and Fine-tuning have the potential to overstate a pretrained model's representation ability.
