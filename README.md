# MODA: Improving DNN Modularization via Activation-Driven Training

### Experiment Setup:
- Ubuntu 20.04
- CUDA 10.2 & cuDNN 7.6.5
- Python 3.8.10
- Pip3 dependencies from `./requirements.txt`

### Run steps:


**1. Training target model:**

1.1. Train standard model:
```sh
$ python3 model_trainer.py --model vgg16 --dataset cifar10 --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1 --wf_affinity 0.0 --wf_dispersion 0.0 --wf_compactness 0.0
```

1.2. Train modular model:
```sh
$ python3 model_trainer.py --model vgg16 --dataset cifar10 --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1 --wf_affinity 1.0 --wf_dispersion 1.0 --wf_compactness 0.3
```

Arguments: 
- `model`={vgg16, resnet18, mobilenet}
- `dataset`={svhn, cifar10, cifar100}
- `wf_affinity` (alpha), `wf_dispersion` (beta), `wf_compactness` (gamma)

**2.Modularizing and Reuse for Sub-task:**

> Edit model_checkpoint_dir in model_modularizer.py to point to the directory containing modular model trained in Step 1

```sh
$ python3 model_modularizer.py --model vgg16 --dataset cifar10 --wf_affinity 1.0 --wf_dispersion 1.0 --wf_compactness 0.3 --activation_rate_threshold 0.9
```

Arguments: 
- `model`={vgg16, resnet18, mobilenet}
- `dataset`={svhn, cifar10, cifar100}
- `activation_rate_threshold`=[0, 1]

**3.Replacing module to improve accuracy:**

3.1. Change base directory:
```sh
$ cd exp_repair
```
3.2. Train weak model:
```sh
$ python3 weak_model_trainer.py --model lenet5 --dataset mixed_cifar10_for_repair --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1
```

Arguments: 
- `model`={lenet5}
- `dataset`={mixed_svhn_for_repair, mixed_cifar10_for_repair}


3.2. Train strong model:
```sh
$ python3 strong_model_trainer.py --model vgg16 --dataset mixed_cifar10_for_repair --batch_size 128 --learning_rate 0.05 --n_epochs 200 --checkpoint_every_n_epochs -1
```

Arguments: 
- `model`={vgg16, resnet18}
- `dataset`={mixed_svhn_for_repair, mixed_cifar10_for_repair}

3.3. Replace module:
```sh
$ python3 weak_model_repair.py --weak_model lenet5 --strong_model vgg16 --dataset mixed_cifar10_for_repair --mixed_class 0 --repair_strategy moda --batch_size 128 --target_epoch 200
```

Arguments: 
- `weak_model`={lenet5}
- `strong_model`={vgg16, resnet18}
- `dataset`={mixed_svhn_for_repair, mixed_cifar10_for_repair}
- `mixed_class`={0,1,2,3,4}
- `repair_strategy`={moda, cnnsplitter}
- `target_epoch`=[1, 200]


> To compare MODA with MwT, find the MwT's source code here https://github.com/icse2025-dnn-modularization/forked_MwT