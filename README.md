# LLM4DSR: Leveraging Large Language Model for Denoising Sequential Recommendation

![Static Badge](https://img.shields.io/badge/Journal-TOIS2025-FF8C00)

This is the PyTorch implementation for our TOIS 2025 paper. 
> Bohao Wang, Feng Liu, Changwang Zhang, Jiawei Chen, Yudi Wu, Sheng Zhou, Xingyu Lou, Jun Wang, Yan Feng, Chun Chen, Can Wang 2025. LLM4DSR: Leveraging Large Language Model for Denoising Sequential Recommendation. [arXiv link](https://arxiv.org/abs/2408.08208)


## Train LLMs for Denoising
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_denoise_model.py --dataset_name "Movie_noise_user0.0" --sample 5000
```

## Generate Denoised Data
```
./denoise.sh $DENOISE_MODEL_PATH $NOISE_DATASET_PATH
```
`DENOISE_MODEL_PATH`: Path of the denoising model, e.g.: ./save_denoise_model/Movie_noise_user0.0/batch128_sample5000_epoch50/checkpoint-2000

`NOISE_DATASET_PATH`: Path of the dataset to be denoised, e.g.: ./data/Movie_noise_user0.0

## Train SASRec
```
cd ./SASRec_code
```
**Train SASRec on the original noisy dataset:**
```
python main.py --dataset Toy_denoise_user0.1 --denoise_category 'none' 
```
**Train SASRec on the denoised dataset:**
```
python main.py --dataset Toy_denoise_user0.1_5000_2000 --denoise_category 'all' --train_noise_filter_threshold 0.6 --test_noise_filter_threshold 0.9 --train_max_denoise_num 2 --test_max_denoise_num 1
```
`dataset` represents the name of the generated denoising dataset. `train_noise_filter_threshold` indicates the noise probability threshold for the training set, while `train_max_denoise_num` specifies the maximum number of items that can be modified in a sequence within the training set. Similarly, `test_noise_filter_threshold` and `test_max_denoise_num` represent the corresponding parameters for the test set.



## Citation
If you find the paper useful in your research, please consider citing:
```
@article{10.1145/3762182,
author = {Wang, Bohao and Liu, Feng and Zhang, Changwang and Chen, Jiawei and Wu, Yudi and Zhou, Sheng and Lou, Xingyu and Wang, Jun and Feng, Yan and Chen, Chun and Wang, Can},
title = {LLM4DSR: Leveraging Large Language Model for Denoising Sequential Recommendation},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3762182},
doi = {10.1145/3762182},
journal = {ACM Trans. Inf. Syst.},
keywords = {Sequential Recommendation, Denoise, Large Language Model}
}
```
