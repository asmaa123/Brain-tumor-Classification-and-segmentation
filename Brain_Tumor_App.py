# -----------------------------
# 📚 Import Libraries
# -----------------------------
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 🧠 Load Models
# -----------------------------
clf_model = load_model("efficient_tumor_model.keras")     # Classification
seg_model = load_model("D:/tumor_segmentation/unet_brain_tumor.h5", compile=False)  # Segmentation

# -----------------------------
# 🖼️ Helper Functions
# -----------------------------
def prepare_image(image, target_size=(224, 224)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_classification(model, image):
    img = prepare_image(image)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    classes = {0: "No Tumor ❌", 1: "Tumor ✅"}
    return classes[class_idx], pred

def predict_segmentation(model, image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128,128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=(0,-1))
    pred_mask = model.predict(img_input)[0,:,:,0]
    return img_resized, pred_mask

# -----------------------------
# 🌟 Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor App 🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("⚙️ Navigation")
app_mode = st.sidebar.radio("Choose a task:", ["🧾 Classification", "🧩 Segmentation"])

# -----------------------------
# 🧾 Classification Page
# -----------------------------
if app_mode == "🧾 Classification":
    st.title("🧠 Brain Tumor Classification")
    uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, caption='Uploaded Image', width=200)
        st.write("---")

        # Prediction
        result, prob = predict_classification(clf_model, image)
        st.markdown(f"### Prediction: **{result}**")

        # Probability bars
        classes = ["No Tumor ❌", "Tumor ✅"]
        probs = prob[0]
        st.markdown("### Probability Scores")
        for c, p in zip(classes, probs):
            st.progress(int(p*100))
            st.write(f"{c}: {p*100:.2f}%")

        # Probability bar chart
        fig, ax = plt.subplots()
        ax.bar(classes, probs, color=["green", "red"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

# -----------------------------
# 🧩 Segmentation Page
# -----------------------------
elif app_mode == "🧩 Segmentation":
    st.title("🧩 Brain Tumor Segmentation")
    uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Predict segmentation
        img_resized, pred_mask = predict_segmentation(seg_model, image)
        threshold = st.slider("🎚️ Segmentation Threshold", 0.0, 1.0, 0.1, 0.05)
        pred_mask_bin = (pred_mask > threshold).astype(np.uint8) * 255

        # Display results side by side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_resized, caption="🖼️ Original MRI", use_container_width=True, clamp=True, channels="GRAY")
        with col2:
            st.image(pred_mask_bin, caption="🔮 Predicted Mask", use_container_width=True, clamp=True, channels="GRAY")
        with col3:
            fig, ax = plt.subplots(figsize=(3.5,3.5))
            sns.heatmap(pred_mask, cmap="viridis", cbar=True, ax=ax)
            ax.set_title("🌈 Probability Heatmap")
            ax.axis("off")
            st.pyplot(fig, clear_figure=True)
