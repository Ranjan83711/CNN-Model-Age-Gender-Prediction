# 🖼️ Age & Gender Prediction using CNN  
![Banner](/mnt/data/7040395b-6451-4003-9cc1-4114207862f9.png)

## 📌 Overview  
This project builds a **multi-task deep learning model** to predict:

- **Age** (regression)  
- **Gender** (classification: Male / Female)  

using the **UTKFace** dataset.  
It includes a **PyTorch training pipeline**, a **modular code structure**, and a **Streamlit web app** for real-time inference.

---

## 📂 Project Structure

Age-Gender-Prediction-using-CNN/
│
├── app/
│ └── streamlit_app.py # Streamlit UI for predictions
│
├── checkpoints/
│ └── best_checkpoint.pth # Saved model (ignored in GitHub: >100MB)
│
├── data/
│ └── UTKFace/ # UTKFace dataset images (age_gender_*.jpg)
│
├── src/
│ ├── train.py # Training script
│ ├── data/
│ │ └── dataset.py # Custom PyTorch Dataset loader
│ ├── models/
│ │ └── multitask_resnet.py # CNN model (ResNet backbone)
│ ├── utils.py # Checkpoint utilities
│ └── init.py
│
└── README.md

yaml
Copy code

---

## 📁 Dataset — UTKFace

Each image filename follows the pattern:

age_gender_race_date.jpg
Example: 21_0_2_20170116174512345.jpg

yaml
Copy code

Used attributes:

| Field | Description |
|-------|-------------|
| age | 0–116 years |
| gender | 0 = Male, 1 = Female |

Dataset includes **20k+ images** with variations in:

- Pose  
- Lighting  
- Background  
- Ethnicity  

---

## 🧠 Model Architecture — MultiTaskResNet

The model shares a **ResNet** backbone and branches into two heads:

java
Copy code
         Shared CNN (ResNet)
                 │
  ┌──────────────┴──────────────┐
  │                             │
Age Regression Head Gender Classification Head
(1 output neuron) (2 output neurons)

shell
Copy code

### Loss Function
Total Loss = L_age + 0.5 × L_gender

markdown
Copy code

Where:
- **L_age** = MSE or L1 Loss  
- **L_gender** = CrossEntropyLoss  

---

## 🛠️ Training Pipeline (src/train.py)

### ✔ Features
- Custom CLI arguments  
- Train/Val/Test split  
- Dataset subsampling (`--max_samples`, `--balanced_by_gender`)  
- GPU/CPU auto-detection  
- Mixed precision training (`--use_amp`)  
- Checkpoint saving  
- Final evaluation  

### ✔ Sample Command

```bash
python -m src.train --data_dir data/UTKFace --epochs 5 --batch_size 32
📊 Model Performance (Your Results)
These metrics are from your run on a reduced dataset:

Validation Metrics
Metric	Value
Train Loss	184.92
Val Loss	201.58
Val MAE (Age)	10.39 years
Val Gender Accuracy	57.50%

Test Metrics
Metric	Value
Test Loss	255.72
Test MAE	12.15 years
Test Gender Accuracy	64.50%

These results are good for a lightweight model trained on a small sample.
Performance improves significantly with:

Larger dataset

Pretrained backbone

Augmentations

More epochs

🖥️ Streamlit App (app/streamlit_app.py)
⭐ Features
Upload image (jpg/png)

Image preview

Auto-loaded model

Predictions with probability

Clean UI layout

Run the app:
bash
Copy code
streamlit run app/streamlit_app.py
The app displays:

Predicted Age

Predicted Gender + Probability

📁 Code Explanation (Important Files)
🔹 src/data/dataset.py
Loads image

Parses age & gender from filename

Applies transforms

Returns dict {image, age, gender}

🔹 src/models/multitask_resnet.py
Defines the model:

ResNet backbone

Age regression head

Gender classification head

🔹 src/train.py
Full training loop

Subset sampling

Progress bars

Evaluation + checkpointing

🔹 app/streamlit_app.py
Frontend interface

Uses st.cache_resource to cache model

Preprocessing & inference

🔹 checkpoints/
Stores .pth trained weights

Added to .gitignore because GitHub blocks files >100MB

⚙️ Setup & Installation
Create environment:
bash
Copy code
conda create -n agegender python=3.10
conda activate agegender
Install requirements:
bash
Copy code
pip install torch torchvision streamlit pillow numpy scikit-learn tqdm
Download UTKFace dataset:
Place images in:

kotlin
Copy code
data/UTKFace/
🚀 Training Examples
Use whole dataset:
bash
Copy code
python -m src.train --data_dir data/UTKFace --pretrained
Train on fewer samples (good for low-end PCs):
bash
Copy code
python -m src.train --max_samples 2000
Balanced gender dataset:
bash
Copy code
python -m src.train --balanced_by_gender --max_per_gender 1000
🚧 Future Improvements
Add face detection (MTCNN / RetinaFace)

Switch age regression → age classification bins

Use ResNet50 or EfficientNet

Deploy on Hugging Face Spaces

Add Grad-CAM visualization

Optimize age regression using L1 loss

🙌 Conclusion
This repository provides a complete, modular, well-structured pipeline for:

✔ Data loading
✔ Multi-task CNN modeling
✔ Training + validation + testing
✔ Real-time prediction via Streamlit
