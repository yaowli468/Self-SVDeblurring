<!-- Title -->
## Deep Self-supervised Spatial-Variant Image Deblurring


 
## Dependencies
* Linux(Tested on Ubuntu 18.04) 
* Python 3.7 (Recomend to use [Anaconda](https://www.anaconda.com/products/individual#linux))
* Pytorch 1.8
* visdom
* pytorch-msssim (Please refer [HERE](https://github.com/jorge-pessoa/pytorch-msssim))

## Get Started

### Download
* The related generator model can be downloaded from [HERE](https://pan.baidu.com/s/1YQ0z6pS4_vWlY1cyl-OagQ)(ydee), please put them to './pretrainModel/'

### Runing
1. Run the following commands. Loading different generator models based on the input image. For example, input a blurry face image, please load the model `stylegan2-ffhq-config-f.pt`
    ```sh
    python run.py --pretrainStyleGanModel './pretrainModel/stylegan2-ffhq-config-f.pt'
    ```

## Citation
	@inproceedings{Li2021DREDSD,
      title={Deep Self-supervised Spatial-Variant Image Deblurring},
      author={Li, Yaowei and Jiang Bo and Shi Zhenghao and Chen Xiaoxuan and Pan, Jinshan},
      booktitle={Neural Networks},
      year={2024}
    }

 



