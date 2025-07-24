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

You can download AVE-PM dataset from [Baidu Cloud Link](https://pan.baidu.com/s/1ErDp1zVEe0mugVMmQFbqow?pwd=2979). And then unzip the video files into the `dataset/data/ave-pm/videos/` folder.

### ⚙️ Data Preparation
Prior to training and evaluation, you'll need to preprocess the data by extracting audio and visual features.

🔉 **Feature Extraction**

Use the provided script to extract features from raw audio and video data. The script is located at `./utils/encode.py`:


This will extract audio and visual features from the videos and store them in `dataset/data/features`.

🧠 **Event Template and Preprocessing**
First generate the `event_templates.pkl` file for audio preprocessing, then process the audio/video data:

``` bash
# Step 1: get the event_templates.pkl file
bash scripts_helper/get_template.sh

# Step 2: process audio/video and extract audio/visual features
bash scripts_helper/run_preprocess.sh
```

Customize the parameters in the `.sh` files as needed. Alternative feature extraction methods may be used.


🎞️ **Video Frames & Audio Extraction for LAVISH**

For LAVISH model training, extract video frames and raw audio into respective directories:

```python
python /LAVISH/scripts/extract_frames.py --out_dir /dataset/data/video_frames/ --video_dir dataset/data/videos/
python /LAVISH/scripts/extract_audio.py --video_pth dataset/data/videos/ --save_pth dataset/data/raw_audios
```

Output locations:
- Frames → `/dataset/data/video_frames/`
- Audios → `/dataset/data/raw_audios/`
  


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
|	|	|── ave/
|	|── feature/
```

## 🚀 Training & Evaluation

We provide scripts for four baseline models. Run evaluations using:

```bash
bash run.sh # Customize model and mode inside the script
```

Modify `run.sh` to select models and specify training/evaluation modes.

**Note**: The `run.sh` script serves as a configuration wrapper. Adjust parameters in the corresponding scripts as needed.


## 📌 Citation




## 🙏 Acknowledgements



## 📄 License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).