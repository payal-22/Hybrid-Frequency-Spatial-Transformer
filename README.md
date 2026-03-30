1. Create a new conda environment
```
conda create -n Main python=3.7
conda activate Main 
```

2. Install dependencies
```
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Data Preparation
Put your input images into folder `Image` of folder `MMLE`

Run `main.m` to will result in a prior image.

Finally, the structure of  `data`  are aligned as follows:

```
data
├── UIEB
    │   ├─ train
    │   │   ├─ DC
    │   │   │   └─ ... (image filename)
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ INPUT
    │   │       └─ ... (image filename)       
    │   └─ test
    │       └─ ...
    └─ ... (dataset name)
```


## Training and Evaluation


### Train

You can modify the training settings for each experiment in the `configs` folder.
Then run the following script to train the model:

```sh
python train.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example:

```sh
python train.py --model Main-m --dataset UIEB --exp data
```

[TensorBoard](https://pytorch.org/docs/1.10/tensorboard.html) will record the loss and evaluation performance during training.

### Test

Run the following script to test the trained model:

```sh
python test.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example:

```sh
python test.py --model Main-m --dataset UIEB --exp data
```
or

```sh
python test-un.py --model Main-m --dataset UIEB-V60 --exp data
```