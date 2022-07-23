# UHD-Low-light-image-enhancement
Convolutional neural networks (CNNs) have achieved unparalleled success in the single Low-light Image Enhancement (LIE) task. Existing CNN-based LIE models over-focus on pixel-level reconstruction effects, hence ignoring the theoretical guidance for sustainable optimization, which hinders their application to Ultra-High Definition (UHD) images. To address the above problems, we propose a new interpretable network, which capable of performing LIE on UHD images in real time on a single GPU. The proposed network consists of two CNNs: the first part is to use the first-order unfolding Taylor’s formula to build an interpretable network, and combine two UNets in the form of first-order Taylor’s polynomials. Then we use this constructed network to extract the feature maps of the low-resolution input image, and finally process the feature maps to form a multi-dimensional tensor termed a bilateral grid that acts on the original image to yield an enhanced result. The second part is the image enhancement using the bilateral grid. In addition, we propose a polynomial channel enhancement method to enhance UHD images. Experimental results show that the proposed method significantly outperforms state-of-the-art methods for UHD LIE on a single GPU with 24G RAM (100 fps).
 

## Cite
```
@article{lin2022uhd,
  title={UHD Low-light image enhancement via interpretable bilateral learning},
  author={Lin, Qiaowanni and Zheng, Zhuoran and Jia, Xiuyi},
  journal={Information Sciences},
  year={2022},
  publisher={Elsevier}
}
```

## Pretrain model

链接：https://pan.baidu.com/s/1xI2uhv0Dzej692DkN-wY1g?pwd=e3np 
提取码：e3np 


