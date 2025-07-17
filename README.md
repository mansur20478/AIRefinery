# AIRefinery

**AIRefinery** is a mobile app that lets users enhance and refine their text **locally on their phones** — no data leaves the device.  
It runs advanced language models entirely offline, combining the power of **LG’s EXAONE 3.5** model with the efficiency of **llama.cpp** for on‑device inference.

---

## ✨ Features

- 🔒 **Privacy‑first** – all processing happens on your phone, no cloud required.
- ⚡ **Fast & Lightweight** – powered by optimized `llama.cpp` inference.
- 🧠 **State‑of‑the‑art models** – uses LG’s **EXAONE 3.5** to refine, polish, and enhance text.
- 📱 **Runs entirely on-device** – works without an internet connection once installed.

---

## 🚀 How It Works

1. User inputs raw text in the app.
2. The app runs the text through the EXAONE 3.5 model, using llama.cpp as the backend.
3. Enhanced, polished, or simplified text is returned — instantly and securely.

---

## 📦 Installation

> ⚠️ **Important:**  
> You must **manually add the model file** to your project before building the app.  
> Download or obtain the following model file:
>
> ```
> EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf
> ```
>
> Then place it in:
>
> ```
> app/src/main/assets/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf
> ```

### Steps to build:
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/AIRefinery.git
2. Place the model file as described above.
3. Open the project in Android Studio.
4. Sync Gradle and build the app.
5. Deploy to your Android device and run.
