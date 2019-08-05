# Unsupervised Domain-Specific Deblurring via Disentangled Representations

Pytorch implementation of the paper [Unsupervised Domain-Specific Deblurring via Disentangled Representations](https://arxiv.org/pdf/1903.01594.pdf). The proposed method is unsupervised and can takes blurred domain-specific image (faces or text) as an input and procude the corresponding sharp estimate.

## Sample Results


## Dataset

## Usage

### Data Preperation



### Train

To train the model, run the following command line in the source code directory. You may set other parameters based on your experiment setting.

```bash
python train.py --dataroot ../datasets/DatasetName/ --name job_name --batch_size 2 --lambdaB 0.1 --lr 0.0002
```

### Test
Our pre-trained model for face and text can be downloaded [here](https://drive.google.com/drive/folders/1P0mP8JjfdV55tDK7a3fIU4yghVmaUJyF?usp=sharing). To test the model, run the following command line in the source code directory. You may set other parameters based on your experiment setting. To choose the perceptual loss type as face, you need to manually set the VGG face model path in network.py. VGG_face pretrained model can be found [here](https://drive.google.com/drive/folders/1P0mP8JjfdV55tDK7a3fIU4yghVmaUJyF?usp=sharing).


```bash
python test.py --dataroot ../datasets/dataset_name/test_blur/ --num 1 --resume ../results/model/locations --name job_name --orig_dir ../datasets/dataset_name/test_orig --percep face
```

## Citation

If you find the code helpful in your research or work, please kindly cite our paper.

```
@inproceedings{lu2019unsupervised,
  title={Unsupervised domain-specific deblurring via disentangled representations},
  author={Lu, Boyu and Chen, Jun-Cheng and Chellappa, Rama},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={10225--10234},
  year={2019}
}
```
## Acknowledgments

The code borrows heavily from [DRIT](). 
