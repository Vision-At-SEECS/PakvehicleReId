## Pak-Vehicle-ReID: A Multi-perspective Benchmark for Vehicle Re-Identification in Unconstrained Urban Road Environment
This repository is for the "**PakVehicle-ReID**" dataset which can be requested for download from the authors of the paper by filling in the form given at the following link: 

[Dataset Acquisition Form](https://docs.google.com/forms/d/e/1FAIpQLSdSwCp2GEXFudaGpDzRpwcBQyAioBFTXAbzKZRfq-XaWNppPg/viewform)

We will respond to your request at the earliest with the link to the dataset.

## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Results](#results)
6. [Authors](#authors)
7. [Citation](#citation)

## Introduction
The challenge of re-identifying vehicles in urban city surveillance systems and major traffic arteries such as highways and roads is an important area of research. The advent of large-scale benchmarks such as VeRI-776 and Vehicle-ID has propelled efforts to enhance search operations from large databases for re-identification. However, several unresolved challenges associated with vehicle re-identification in unconstrained environments remain to be explored. In order to foster research in this field, we have compiled a new multi-perspective dataset, PAKVehicle-ReId, captured by real-world surveillance cameras in urban cities in Pakistan and India. To the best of our knowledge, this is the first such dataset collected under unconstrained conditions in the Asian region. The dataset comprises 80,000 images of 20,000 unique vehicles. Additionally, a deep learning-based technique for extracting multi-dimensional robust features for vehicle re-identification using convolutional neural networks has been proposed. The results show the effectiveness of the proposed method on the PAKVehicle-ReId dataset as well as on two other existing datasets, VeRi-776 and VehicleID. 

## Network 
![The Method](docs/architecure.jpg)


## Setup
The following libraries/packages are required for successful execution of this project.

  1. PyTorch
  2. Shapely
  3. Opencv
  4. TensorboardX

3. For convience, you may execute `requirements.txt` in cmd prompt.

## Dataset

Please download the PakVehicle-ReID dataset and put the training and test folders in `datasets` directory as follows.
```
├── Vehiclereid/
│   ├── datasets/                  
│   ├── PakVehicleID/                   
│       ├── Images 
│       ├── image_query
        ├── Train_test_splits/
            ├── train_list 
            ├── test_list

``` 

After successful completion of setup, please verify the directory paths and other parameters in config.py.

You may execute the following command given in the dataset folder to view a summary of the dataset.
```
pakvehicleid.py
```


## Usage
### train
To train an image-reid model, please execute the command given below.  
```
python train_xent_tri.py \
-s Pakvehiclereid \    #source dataset for training
-t Pakvehiclereid \    # target dataset for test
--height 128 \ # image height
--width 256 \ # image width
--optim amsgrad \ # optimizer
--lr 0.0003 \ # learning rate
--max-epoch 60 \ # maximum epoch to run
--stepsize 20 40 \ # stepsize for learning rate decay
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \ # network architecture
--save-dir log/resnet50-Pakvehiclereid \ # where to save the log and models
--gpu-devices 0 \ # gpu device index
```
### test
1. To test the model, please verify the paths of test directory and also specify the pretrained model in checkpoint in the config.py file. 
2. Use --evaluate to switch to the evaluation mode. In doing so, no model training is performed.
3. Multiple test size are given in the dataset folder which can be evaluated by changing test size command (-test--size) given in the script below.
```
python test_imgreid.py \
-s Pakvehiclereid \ # this does not matter any more
-t Pakvehiclereid \ # you can add more datasets here for the test list
--height 128 \
--width 256 \
--test-size 800 \
--test-batch-size 100 \
--evaluate \
-a resnet50 \
--load-weights path_to_model.pth.tar \
--save-dir log/Pakvehiclereid \
--gpu-devices 0 \
```
## Results

### Quantative Results
![Quantative Results](docs/table4.PNG)
![Quantative Results](docs/table5.PNG)
### Visual Results of Query Matching againts Vehicles Gallery 
![Qualitative Examples of SOTA Methods on PakVehicle-ReID](docs/result.png)

# Authors

Hasan Ali Asghar <hasghar.mscs19seecs@seecs.edu.pk>

Bostan Khan <bkhan.mscs19seecs@seecs.edu.pk>

Zuhair Zafar <zuhair.zafar@seecs.edu.pk>

Muhammad Moazam Fraz <mfraz@turing.ac.uk>

<!-- # Citation
- If you follow the code or ideas from our paper, please cite our paper:
```
``` -->
