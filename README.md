# StylizedNeRF

This project aims to perform neural style transfer ([A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)) 
on a [NeRF](https://www.matthewtancik.com/nerf) generated scene.

|<img src="imgs/fern.gif" alt="fern" width="200"/>|<img src="imgs/water.jpg" alt="style" width="50"/><img src="imgs/fern_stylized_2.gif" alt="fern_stylized" width="200"/>|<img src="imgs/picasso.jpg" alt="style" width="80"/><img src="imgs/fern_stylized_1.gif" alt="fern_stylized" width="200"/>|
|---|---|---|

## Data Preparation
Please follow the instructions [here](https://github.com/yenchenlin/nerf-pytorch#more-datasets) to prepare the necessary data
for NeRF training.

## How to Run
Please make sure to run
```bash
bash download_model.sh
```
beforehand to download the pre-trained VGG-19 weights for the later style-transfer algorithm.

The style transfer here requires the model to be pre-trained with the original NeRF algorithm.
One may consider training such a model using the codes from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), 
or directly download the pre-trained models provided in the same repo.
The pre-trained model weights should be placed under the folder `logs/[exp_name]`, where `[exp_name]` is the folder in which the 
logs of the later stylization experiment will be saved.

## Acknowledgements
This code repo is heavily based on [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). 
Thanks the author for his great job!
