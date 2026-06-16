# Denim Jeans Defect Analyzer - Premium Edition

Welcome to the **Denim Jeans Defect Analyzer**! This is a complete end-to-end Machine Learning pipeline and premium desktop application designed to automatically detect manufacturing defects in denim jeans using AI.

## 🧠 How I Made It

This project was built from the ground up to solve real-world quality control issues in apparel manufacturing. The core architecture uses **PyTorch** to train a lightweight, highly efficient convolutional neural network.

1. **Dataset Pipeline:** We created an automated script (`build_dataset.py`) to scrape through Excel files provided by quality control inspectors. It dynamically extracts image IDs and maps them to their respective defect labels (like "Spot", "Broken Stitch", "Uncut", etc.), automatically separating clean jeans from damaged ones.
2. **Transfer Learning:** We took a pre-trained **MobileNetV2** model and fine-tuned it specifically on our dataset of denim textures and defects. This allows the model to leverage edge-detection and feature-extraction capabilities learned from millions of images to achieve an incredible **~96% accuracy** on validation data!
3. **Desktop Application:** We didn't stop at a python script. We packaged the entire inference engine into a sleek, premium, dark-mode desktop application using `CustomTkinter`.

## ⚙️ How It Works

The core of the system relies on **Grad-CAM (Gradient-weighted Class Activation Mapping)** technology. 

When you upload an image of a pair of jeans:
1. The app passes the image through the MobileNetV2 neural network.
2. The network outputs a probability score indicating if the jeans are CLEAN or DAMAGED.
3. Behind the scenes, the Grad-CAM algorithm calculates the gradients of the defect probability with respect to the final convolutional layer of the network. 
4. It uses these gradients to figure out *exactly which pixels* caused the AI to flag the jeans as damaged!
5. This localization is rendered as a colorful heat map overlaid directly onto your original image, highlighting the exact location of the defect.

## 🚀 How to Use It

We have optimized this app so that anyone can use it instantly, without needing to install Python, PyTorch, or write a single line of code!

### Running the App
1. Download the `dist/gui_app` folder (the packaged application).
2. Inside the `gui_app` folder, you will see a folder named `ProcessedData`.
3. Make sure your trained model file (`defect_model.pth`) is placed inside that `ProcessedData` folder.
4. Double click `gui_app.exe` to launch the application instantly!

### Using the Interface
* Click **"Select Image"** on the left sidebar to upload an image of denim jeans.
* The application will instantly analyze the image and display the original image side-by-side with the AI-generated Heatmap.
* The bottom text will glow green for **CLEAN** jeans and bright red for **DAMAGED** jeans, alongside the exact confidence percentage from the AI model.
* You can toggle between Dark Mode and Light Mode using the dropdown in the bottom left corner.

---

*Developed by Abdul-Rahman258*
