# TAD

Main code of "Disentangle Irrelevant and Critical Representations for Face Anti-Spoofing"

## Main Dependencies

```
python=3.8
pytorch=1.10.0+cu113
```

The hardware environment is Nvidia A40 GPU.

## Dataset

 we first randomly select 20 frames and 10 frames from each video on all datasets for training and testing respectively. Take OULU-NPU dataset as an example, the directory of dataset OULU-NPU is showing as follows:

```
|--oulu-npu
|  |--Protocol_1
|  |  |--testA      # live face in testset
|  |  |--testAD     # depth map of  live face in testset
|  |  |--testB      # spoof face in testset
|  |  |--trainA     # live face in trainset
|  |  |--trainAD    # depth map of live face in trainset
|  |  |--trainB     # spoof face in trainset
|  |  |--trainBD    # depth map of spoof face in trainset
|  |--Protocol_2
|  |  |--...
|  |--Protocol_3
|  |  |--test1A
|  |  |--test1B
|  |  |--...
|  |  |--test6A
|  |  |--train1A
|  |  |--train1B
|  |  |--train1D
|  |  |--...
|  |  |--train6A
|  |  |--train6B
|  |  |--train6D
|  |--Protocol_4
|  |  |--...
```

Image naming Rules：

live images exsample: 000000.jpg, 000001.jpg ...

spoof image exsample: 000000_print1_.jpg, 000020_print2_.jpg, 000040_replay1_.jpg, 000060_replay2_.jpg, 

## Training

Before training TAD, the following things need to be done.

- please modify the configuration file, mainly including batch_size, data_root, and protocol, etc.

- Train a pre-trained depth model (Section 3.1) to assist TAD in learning face depth 

  ```
  python train_depthnet.py
  ```

after getting the pre-trained depth model, you can start the training of TAD

```
python train.py
```

## Testing

you can modify  the main function of test.py to choose the item you need to test, then run the testing code:

```
python test.py
```

