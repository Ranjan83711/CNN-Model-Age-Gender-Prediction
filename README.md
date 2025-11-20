# Age & Gender Prediction using CNN


A lightweight project that predicts **Age** and **Gender** from face images using a **CNN + ResNet backbone** trained on the **UTKFace dataset**.

---

## Overview

* Predicts **Age (regression)** and **Gender (Male/Female)** from an image.
* Built using **PyTorch** and **Streamlit**.
* Uses **UTKFace** dataset where filenames encode `age_gender_race_date.jpg`.

---

## Project Structure

```
src/
 ├── train.py               # Training script
 ├── data/dataset.py        # Dataset loader
 ├── models/multitask_resnet.py   # CNN model (ResNet backbone)
 └── utils.py               # Checkpoint helpers

app/
 └── streamlit_app.py       # Web interface for predictions

checkpoints/                # Model weights (ignored in GitHub)
data/UTKFace/               # Dataset images
```

---

## Training Results (Your Run)

* **Train Loss:** 184.92
* **Val Loss:** 201.58
* **Val MAE:** 10.38 years
* **Val Gender Accuracy:** 57.5%
* **Test MAE:** 12.15 years
* **Test Gender Accuracy:** 64.5%

---

## How to Train

```
python -m src.train --data_dir data/UTKFace --epochs 5
```

For faster training on low-end PCs:

```
python -m src.train --max_samples 2000
```

---

## How to Run the App

```
streamlit run app/streamlit_app.py
```

Upload a face image → Get age & gender prediction.

---

## Notes

* UTKFace images must be placed in `data/UTKFace/`.
* Large model checkpoint files are ignored by GitHub due to size limits.

---
