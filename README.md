# Virtual Immunohistochemistry Staining for Histological Images Assisted by Weakly-supervised Learning
The official implementation of "Virtual Immunohistochemistry Staining for Histological Images Assisted by Weakly-supervised Learning".

## How to use: 
```shell
python3 train.py --data_train_A ./dataset/trainA --data_train_B ./dataset/trainB \
    --load_size 256 --crop_size 256 --preprocess none --model confusion_gan --pretrained_IHC_Classifier ./pretrain_IHC_classifier.pth \
    --netG unet_256 --netD basic --netE basic_3d --A_labels ./trainA_labels.pt \
    --dataset_mode unaligned --direction AtoB 
```

## Reference
If you found our work useful in your research, please consider citing our works(s) at:
```
@inproceedings{li2024virtual,
  title={Virtual immunohistochemistry staining for histological images assisted by weakly-supervised learning},
  author={Li, Jiahan and Dong, Jiuyang and Huang, Shenjin and Li, Xi and Jiang, Junjun and Fan, Xiaopeng and Zhang, Yongbing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11259--11268},
  year={2024}
}
```

🧱 **Built Upon**

Parts of this codebase are adapted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).  
We thank the original authors for their contributions.

© This code is released under the GPLv3 license and is intended for non-commercial academic research only.
