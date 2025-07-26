# UniCR: A Unimodel with Contribution Learning Network and Residual Frame-wise Attention for Multi-aspect-ratio Audio-Visual Event Localization


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green.svg)


## 📦 Installation

1. **Clone the repository**

```bash
git git@github.com:talent0325/UniCR.git
cd UniCR

conda env create -f environment.yaml
conda activate unicr
```

## 📁 Dataset

AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK. And then unzip the video files into the `./dataset/data/ave/videos/` folder.

AVE-PM can be downloaded from [Baidu Cloud Link](https://pan.baidu.com/s/1ErDp1zVEe0mugVMmQFbqow?pwd=2979). And then unzip the video files into the `./dataset/data/ave-pm/videos/` folder.

### ⚙️ Data Preparation
Prior to training and evaluation, you'll need to preprocess the data by extracting audio and visual features.

Or you can download AVE [audio_features](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing) and [visual_features](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing) (7.7GB). And AVE-PM features is not provided, you need to extract features by yourself according to the [README.md](https://github.com/dzdydx/ave-pm/tree/main) of AVE-PM.


This will extract audio and visual features from the videos and store them in `dataset/data/feature/<dataset_name>/`.

### 📂 Directory Structure

```graphql
UniCR/
├── dataset/
|	├── csv/			   # csv files for training,validating and testing
|	|	├── ave-pm/
|	|	|   |── select/
|	|	|── ave/
|	|	|   |── select/
|	├── data/
|	|	├── ave-pm/
|   |	|	|── videos/
|   |	|	|── frames/
|	|	|── ave/
|   |	|	|── videos/
|   |	|	|── frames/
|	|── feature/
|	|	├── ave-pm/
|	|	|── ave/
```

## 🚀 Training & Evaluation
Before training and evaluation, you need to modify the `.sh` script to specify the dataset to use and the training/evaluation mode.

- if is_select is true, you choose the S-xxx dataset to train and test
- if ave is true, you use the AVE dataset to train and test
- if avepm is true, you use the AVE-PM dataset to train and test

Note: you should set the correct data_root, meta_root, v_feature_root, a_feature_root, category_num, and preprocess according to your own dataset
To train or evaluate the model, run the following command:

```bash
bash train.sh 
# or
bash test.sh
```

## 📈 Checkpoints
You can download the checkpoints from [coming soon]() and put them into the `./checkpoints/` folder.



## 📌 Citation
If you find this project helpful, please consider citing:

```

```



## 🙏 Acknowledgements
We acknowledge the following works: 
- AVE (ECCV'18): https://github.com/YapengTian/AVE-ECCV18
- AVE-PM: https://github.com/dzdydx/ave-pm/tree/main


## 📄 License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).