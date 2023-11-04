# FedPN: Privacy-enhanced Federated Prototype Networks #

## Making mirror
1. Write Dockerfile
```dockerfile
FROM python:3.10-slim

ADD ./ /FL
WORKDIR /FL

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
```

2. Excute
```bash
docker build -t 18580629860/fedpn:1.1 .
```

3. Run
```bash
docker run -it --name fedpn 18580629860/fedpn:1.1 bash
```

4. Excute
```bash
python main.py --rounds=50 --num_users=5 --alg=fedpn
```

## Pulling mirror
1. Install [Docker](https://www.docker.com/).

2. Download [automated build](https://hub.docker.com/r/18580629860/fedpn) from public [Docker Hub](https://hub.docker.com/):
```bash
docker pull 18580629860/fedpn:1.1
```

3. Run
```bash
docker run -it --name fedpn 18580629860/fedpn:1.1 bash
```

4. Excute
```bash
python main.py --rounds=50 --num_users=5 --alg=fedpn
```

## Environment

**If you wish to conduct experiments at a faster pace, it is recommended to avoid using docker, as it may not be GPU-friendly.**



| Category                     | Parameters                                 |
| ----------------------------- | ---------------------------------------- |
| `CPU`                     | Intel Xeon 4210R|
| `RAM` | 32GB|
| `GPU`      | Nvidia RTX 3090|
| `Operation system` | Ubuntu 18.04 LTS|
| `CUDA version` | 11.3|
| `Machine learning Framework` | torch 1.13.0|



## Experiments

**It is worth noting that our proposed algorithm is FedPN.**



1. The experiment takes approximately **20 hours** to run.  The section 5.2 of the experiment described in this article was determined by executing the following shell command:

```bash
python main.py --rounds=50 --num_users=5 --alg=local
```



```bash
python main.py --rounds=50 --num_users=5 --alg=fedpn
```



```bash
python main.py --rounds=50 --num_users=5 --alg=fedproto
```



```bash
python main.py --rounds=50 --num_users=5 --alg=fedavg
```



```bash
python main.py --rounds=50 --num_users=5 --alg=fedprox
```



2. To enable **differential privacy mode** or **multi-key semi-homomorphic encryption mode**, it is advised not to use docker. Instead, please create Python virtual environments on your computer and install the necessary libraries as outlined in the requirements file.



```bash
python main.py --rounds=50 --num_users=5 --alg=fedprox --is_not_the=1 --add_noise_proto=1
```



## Tiny-VehicleDataset

You can download VehicleDataset [here](https://www.kaggle.com/datasets/shamate2b/vehicledataset). After the download is complete, please create a file named **data** in the main program directory and place the decompressed file inside it.

## Parameters

| Parameter                      | Description                                                                              |
| ----------------------------- |------------------------------------------------------------------------------------------|
| `rounds`                     | number of rounds of training.                                                            |
| `num_users` | number of users.                                                                         |
| `alg`      | algorithms. Options: `fedpn`, `fedavg`, `fedprox`.                                       |
| `train_ep` | the number of local episodes.                                                            |
| `local_bs` | local batch size.                                                                        |
| `lr` | learning rate.                                                                           |
| `momentum` | SGD momentum.                                                                            |
| `weight_decay` | Adam weight decay.                                                                       |
| `optimizer`    | optimizer. Options: `SGD`, `Adam`.                                                       |
| `num_bb` | number of backbone.                                                                      |
| `train_size` | proportion of training dataset.                                                          |
| `num_classes` | number of classes.                                                                       |
| `alpha` | parameters of probability distribution.                                                  |
| `non_iid` | non-iid. Options:`0(feature shift)`,`1(label shift)`,`2(feature shift and label shift)`. |
| `ld` | hyperparameter of fedproto and fedpn.                                                    |
| `mu` | hyperparameter of fedprox.                                                               |
| `is_not_the` | multi-key encryption scheme.`0 (is not enabled)`, `1 (is enabled)`.                      |
| `add_noise_proto` | differential privacy. `0 (is not enabled)`, `1 (is enabled)`.                            |
| `scale` | noise distribution std.                                                                  |
| `noise_type` | noise type.                                                                              |


## Usage

Here is an example to run FedPN on VehicleDataset:
```
python main.py --rounds=25 \
    --num_users=5 \
    --alg=fedpn \
    --local_bs=32 \
    --train_size=0.9 \
    --non_iid=2 \
    --is_not_the=1 \
    --add_noise_proto=1 \
```



## Hyperparameters

If you try a setting different from our paper, please tune the hyperparameters of FedPH. You may tune mu and ld from {0.001, 0.01, 0.1, 1, 5, 10}. If you have sufficient computing resources, you may also tune temperature from {0.1, 0.5, 1.0}. 

