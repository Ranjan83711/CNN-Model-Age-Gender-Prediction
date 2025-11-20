# app/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.models.multitask_resnet import MultiTaskResNet

st.set_page_config(page_title='UTKFace Age & Gender Demo', layout='centered')
st.title('Age & Gender Prediction (UTKFace)')

# Use cache_resource for model objects (replaces deprecated st.cache)
@st.cache_resource
def load_model(path: str = 'checkpoints/best_checkpoint.pth', pretrained: bool = False):
    """
    Load model and keep it cached as a resource so Streamlit doesn't reload on every rerun.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskResNet(pretrained=pretrained)
    # Load checkpoint (expects dict with 'model_state')
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# UI: allow user to request model load or let it load automatically when needed
col1, col2 = st.columns([2, 1])
with col2:
    model_path = st.text_input("Checkpoint path", value="checkpoints/best_checkpoint.pth", key="checkpoint_path")
    preload_button = st.button("Load model", key="load_model_btn")

# Single file_uploader with a unique key
uploaded = st.file_uploader(
    "Upload a face image (.jpg/.png). Use UTKFace-style images for best results.",
    type=['jpg', 'jpeg', 'png'],
    key="input_image_uploader"
)

# Informational message when no image
if uploaded is None:
    st.write('Upload an image to predict age and gender.')
else:
    # Display uploaded image
    try:
        img = Image.open(uploaded).convert('RGB')
    except Exception as e:
        st.error(f"Could not open uploaded image: {e}")
        st.stop()

    st.image(img, caption='Uploaded image', use_container_width=True)

    # Preprocessing transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    tensor = transform(img).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(device)

    # Load model: either when user pressed button or automatically when image is uploaded
    model = None
    try:
        if preload_button:
            # If user asked to load explicit path from text_input
            model = load_model(path=model_path)
            st.success(f"Model loaded from {model_path}")
        else:
            # Try to load the default checkpoint path (or the one set in text_input)
            model = load_model(path=model_path)
    except Exception as e:
        st.error(f"Could not load model from '{model_path}': {e}")
        st.stop()

    # Run inference
    if model is not None:
        with st.spinner('Predicting...'):
            try:
                # model expected to return (age_out, gender_out_logits)
                age_out, gender_out = model(tensor)
                age_pred = age_out.detach().cpu().numpy().item()
                gender_probs = torch.softmax(gender_out, dim=1).detach().cpu().numpy()[0]
                # `gender_probs` indexing depends on your model mapping.
                # Here we display the argmax label and its probability.
                arg = int(gender_probs.argmax())
                gender_label = 'Female' if arg == 1 else 'Male'
                prob = float(gender_probs[arg])
                st.success(f'Predicted age: {age_pred:.1f} years')
                st.info(f'Predicted gender: {gender_label} (prob: {prob:.2f})')
            except Exception as e:
                st.error(f"Prediction failed: {e}")
