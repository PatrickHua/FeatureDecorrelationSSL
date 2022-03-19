# On feature decorrelation in self-supervised learning

This is a PyTorch implementation of the paper [On Feature Decorrelation in Self-Supervised Learning](https://arxiv.org/pdf/2105.00470.pdf):

```
@inproceedings{hua2021feature,
  title={On feature decorrelation in self-supervised learning},
  author={Hua, Tianyu and Wang, Wenxiao and Xue, Zihui and Ren, Sucheng and Wang, Yue and Zhao, Hang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9598--9608},
  year={2021}
}
```
The code and instructions follow [SimSiam](https://github.com/PatrickHua/SimSiam).


# CIFAR10 test run
python main.py --data_dir ~/Data --log_dir ~/.cache --ckpt_dir ./output/ -c 'configs/exp_small_200ep/cifar10/lin_vs_bn_vs_dbn_vs_sdbn/sdbn.yaml' --debug
### OUTPUT:
```python
creating file /Users/tiany/.cache/in-progress_0319103643_covnorm-cifar10-resnet18_cifar_variant1
To view: "watch -n 1 tail /Users/tiany/.cache/in-progress_0319103643_covnorm-cifar10-resnet18_cifar_variant1/train.log"
Epoch 0/1: 100% 1/1 [00:25<00:00, 25.59s/it, loss=231, corr=0.162, rank=31, std=1.02, bias=0, lr=0]
Using k = 3200% 1/1 [00:20<00:00, 20.58s/it, loss=231, corr=0.162, rank=31, std=1.02, bias=0, lr=0]
kNN: 100% 1/1 [00:15<00:00, 15.90s/it, Accuracy=15.6]
Training: 100% 1/1 [01:01<00:00, 61.41s/it, epoch=0, accuracy=15.6, loss=231, F1Unif=0, F1Corr=0, F2Unif=0, F2Corr=0, f1align=0, f2align=0]
Model saved to /Users/tiany/.cache/covnorm-cifar10-resnet18_cifar_variant1_0319103747.pth
Evaluating: 100% 1/1 [00:20<00:00, 20.07s/it]
Accuracy = 6.25
Log file has been saved to ./output/debug_0319103643_covnorm-cifar10-resnet18_cifar_variant1
```




