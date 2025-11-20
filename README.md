# 📘 Age & Gender Prediction using CNN (UTKFace Dataset)

This project builds a multi-task deep learning model that predicts:

Age (regression)

Gender (classification: male/female)

using the UTKFace dataset, a widely used dataset for facial attribute prediction tasks.

A full training pipeline, a modular PyTorch codebase, and a Streamlit web app for real-time predictions are included.

🚀 Project Structure
Age-Gender-Prediction-using-CNN/
│
├── app/
│   └── streamlit_app.py          # Streamlit UI for model inference
│
├── checkpoints/
│   └── best_checkpoint.pth       # Saved model (ignored in Git due to size)
│
├── data/
│   └── UTKFace/                   # UTKFace dataset (age_gender_race_date.jpg)
│
├── src/
│   ├── train.py                  # Main training script
│   ├── data/
│   │   └── dataset.py            # UTKFace dataset loader
│   ├── models/
│   │   └── multitask_resnet.py   # Model definition (ResNet backbone)
│   ├── utils.py                  # Checkpoint save/load utilities
│   └── __init__.py
│
├── .gitignore
└── README.md (this file)

🧠 1. Problem Statement

Predict a person’s age and gender from a single input image.

This project uses:

A multi-task learning approach

A shared backbone (ResNet-18/34/50)

Two output heads:

Age Regression Head

Gender Classification Head

🖼️ 2. Dataset — UTKFace

UTKFace contains over 20,000+ face images, spanning:

Attribute	Description
Age	0–116 years
Gender	0 = Male, 1 = Female
Race	Not used in this project
Image Size	Varies (mostly 200×200)
Filename format
age_gender_race_dateandtime.jpg
Example: 21_0_2_20170116174512345.jpg

🔧 3. Model Architecture — MultiTaskResNet

Your model is defined in:

src/models/multitask_resnet.py

Architecture Summary:

Backbone: ResNet (pretrained or untrained)

Shared CNN Features

Two Heads:

age_head → outputs a single regression value

gender_head → outputs a 2-class logits vector

Loss Function
Total Loss = AgeLoss(MSE or L1) + 0.5 * GenderLoss(CrossEntropy)


You can tune the gender weight using:

--gender_loss_weight 0.5

🛠️ 4. Training Pipeline (src/train.py)

Your training file performs the following steps:

✔ Load Dataset

UTKFaceDataset parses filename to extract age & gender.

✔ Train/Val/Test Split
--val_frac 0.1
--test_frac 0.1

✔ Dataloaders created with:

Augmentations via torchvision.transforms

Configurable batch_size

Safe num_workers for Windows

✔ Model Creation
model = MultiTaskResNet(pretrained=True/False)

✔ Mixed Precision Training (optional)
--use_amp

✔ Checkpointing

Best validation loss model stored at:

checkpoints/best_checkpoint.pth

✔ Final Evaluation on Test Set
📊 5. Training Metrics (Your Results)

These are the metrics you provided last night:

Epoch 5 Results
Metric	Value
Train Loss	184.9176
Validation Loss	201.5752
Validation MAE (Age)	10.3883 years
Validation Gender Accuracy	57.50%
Final Test Metrics
Metric	Value
Test Loss	255.7194
Test MAE (Age)	12.1538 years
Test Gender Accuracy	64.50%

These are good starting results for a lightweight model trained on a reduced dataset.

⚡ 6. Streamlit Inference App

Your app lives at:

app/streamlit_app.py

Features:

Upload a face image (jpg/png)

Preprocess → Resize → Normalize

Auto-load model with @st.cache_resource

Predict:

Age (float)

Gender (Male/Female with probability)

Beautiful UI with use_container_width

Run Streamlit
cd "E:\Age_Gender Prediction using CNN"
streamlit run app/streamlit_app.py

🧩 7. Detailed Explanation of Important Files
📁 src/data/dataset.py

Loads UTKFace images

Parses filename: age_gender_*.jpg

Returns:

{
  'image': image_tensor,
  'age': age,
  'gender': gender
}


Handles:

Transformations

Corrupt file filtering

Iterable Dataset

📁 src/models/multitask_resnet.py

Defines the multitask model:

backbone = resnet18(pretrained)
age_head = Linear(512, 1)
gender_head = Linear(512, 2)


Outputs:

age_pred, gender_logits

📁 src/train.py

Responsible for:

CLI arguments

Deterministic seed setup

Dataset creation

Subsampling (max_samples, balanced_by_gender)

Training loop with progress bar

Validation and test evaluation

Checkpoint saving

📁 src/utils.py

Contains:

save_checkpoint()

load_checkpoint()

Utility wrappers for safe model reloads

📁 checkpoints/

Default folder for .pth files
(ignored in .gitignore because of GitHub's 100MB limit)

📁 app/streamlit_app.py

The core UI.

Handles:

Safe path injection for imports

Cached model loading

Image uploader with unique key

Prediction with progress spinner

⚙️ 8. Installation & Setup
Clone Repository
git clone https://github.com/Ranjan83711/Age-Gender-Prediction-using-CNN.git
cd Age-Gender-Prediction-using-CNN

Create Virtual Environment
conda create -n agegender python=3.10
conda activate agegender

Install Requirements
pip install torch torchvision streamlit numpy pillow scikit-learn tqdm

🏋️ 9. Run Training
Basic Training
python -m src.train --data_dir data/UTKFace --epochs 5 --batch_size 32

With Pretrained Backbone
python -m src.train --pretrained

Train with Fewer Images (for slow PCs)
python -m src.train --max_samples 2000

Balanced Subset:
python -m src.train --balanced_by_gender --max_per_gender 1000

🖥️ 10. Run Inference App
streamlit run app/streamlit_app.py


Upload an image → Model predicts age & gender.

🚧 11. Known Limitations / Future Work

Age MAE ~10 years on small subset — can be improved by:

Using L1 loss

Training longer on full dataset

Using ResNet34/50 or EfficientNet

Gender accuracy ~65% on limited data

Add face detection (MTCNN/RetinaFace)

Add augmentation (ColorJitter, Cutout)

Host model in Hugging Face Spaces

🌟 12. Conclusion

This project demonstrates a full end-to-end deep learning pipeline:

✔ UTKFace dataset parsing
✔ Custom multi-task model
✔ Training + validation + testing
✔ Model checkpointing
✔ Streamlit inference dashboard
✔ GitHub-ready modular code structure

You now have a complete production-style age & gender prediction system.
