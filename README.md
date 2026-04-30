# 🧠 DDPM Face Generator (Streamlit App)

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** using PyTorch and deploys it with Streamlit.

## 🚀 Features

- Generate realistic face images from random noise
- Visualize intermediate denoising steps
- Lightweight Streamlit web app
- Trained on FFHQ dataset

---

## 📂 Project Structure
├── app.py # Streamlit UI
├── model.py # UNet architecture
├── utils.py # Diffusion + sampling
├── config.py # Hyperparameters
├── ddpm_weights_only.pth # Trained model
├── requirements.txt


---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
🖼️ Output
Starts from random noise
Gradually denoises image
Shows intermediate steps
⚙️ Model Details
Image Size: 128x128
Timesteps: 300
Architecture: U-Net
Loss: MSE
📌 Note

Make sure the model architecture and config match the training setup.

👨‍💻 Author

Mubashir Siddique


---

# ✅ 2. SMALL FIX IN `requirements.txt`

Add this line (important for Streamlit rendering images):

```txt
matplotlib

Final:

streamlit
torch
torchvision
numpy
Pillow
matplotlib
✅ 3. IMPORTANT CHECK (MOST STUDENTS MISS THIS)
In app.py, make sure:
model.load_state_dict(torch.load("ddpm_weights_only.pth", map_location="cpu"))

✔ Not GPU
✔ Not different filename

✅ 4. FINAL DEPLOYMENT OPTIONS
🔹 Option 1: Run Locally
streamlit run app.py
🔹 Option 2: Deploy Online (FREE)
👉 Use Streamlit Cloud
Go to: https://streamlit.io/cloud
Connect GitHub

Select repo:

diffusion-face-generator-streamlit

Main file:

app.py
Deploy 🚀
⚠️ 5. Possible Issue (IMPORTANT)

Your file:

ddpm_weights_only.pth

👉 If it is >100MB, GitHub may block it.

Fix:
Use Git LFS OR
Upload to:
Google Drive
Kaggle
HuggingFace
🚀 6. OPTIONAL (to impress teacher)

Add this to app.py:

st.sidebar.title("Controls")

OR add a button:

if st.button("Generate New Face"):
