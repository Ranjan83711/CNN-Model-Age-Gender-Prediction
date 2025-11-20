# src/data/dataset.py (recursive discovery version)
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, age_mode="regression"):
        self.root_dir = root_dir
        # recursively find images (full paths)
        p = Path(root_dir)
        if not p.exists():
            self.files = []
        else:
            self.files = sorted([str(f) for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.age_mode = age_mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.files):
            raise IndexError("Index out of range")

        # fname is full path now
        path = self.files[idx]
        fname = os.path.basename(path)
        img = Image.open(path).convert('RGB')

        # parse labels from filename: age_gender_race_date.jpg
        parts = fname.split('_')
        try:
            age = float(parts[0])
            gender = int(parts[1])
        except Exception:
            age, gender = 0.0, 0

        if self.age_mode == 'binned':
            age_label = int(age // 10)
        else:
            age_label = age

        img = self.transform(img)
        return {
            'image': img,
            'age': age_label,
            'gender': gender,
            'filename': fname,
            'path': path
        }
