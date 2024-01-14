# Disentangled Generation with Information Bottleneck for Enhanced Few-Shot Learning
**Apologize for the dirty code. We are working on organizing the code.**

## Dependencies

* Python 3.8.16
* [PyTorch 1.9.0](http://pytorch.org)

## Prepare Datasets
To be updated
## Training

``` sh
python main.py --phase pretrain_encoder --gpu 2 --save-path "path to save" --train-shot 5 --val-shot 5 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet12 --dataset miniImageNet --z_disentangle --zd_beta 6.0 --zd_beta_annealing --add_noise 0.2 --temperature 500 --feature_size 640 --generative_model vae --latent_size 64 --attSize 171 --distill --use_feature
```

### Testing

```sh
python main.py --phase generative_test --gpu 1 --save-path "path to save" --train-shot 1 --val-shot 1 --train-query 15 --val-query 15 --head FuseCosNet  --network ResNet12 --dataset miniImageNet --feature_size 640 --generative_model vae --latent_size 64 --attSize 171 --use_feature
```

## License

To be updated
