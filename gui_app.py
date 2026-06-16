import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import cv2

def get_model_path():
    """ Get absolute path to the model, finding the directory of the executable """
    if getattr(sys, 'frozen', False):
        # Running as compiled PyInstaller executable (.exe)
        base_dir = os.path.dirname(sys.executable)
    else:
        # Running as normal python script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "ProcessedData", "defect_model.pth")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        output = self.model(x)
        self.model.zero_grad()
        output.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        return cam.cpu().detach().numpy(), torch.sigmoid(output).item()

def generate_cam_overlay(img_tensor, cam):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).detach().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(cam_img[:, :, ::-1])

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Denim Jeans Defect Analyzer - Premium Edition")
        self.geometry("900x650")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Denim Analyzer", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.select_btn = ctk.CTkButton(self.sidebar_frame, text="Select Image", command=self.select_image)
        self.select_btn.grid(row=1, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Model Status: Loading...", text_color="orange")
        self.status_label.grid(row=2, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set("Dark")

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1), weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.original_label = ctk.CTkLabel(self.main_frame, text="Original Image", font=ctk.CTkFont(size=16, weight="bold"))
        self.original_label.grid(row=0, column=0, padx=10, pady=(20, 0))
        
        self.heatmap_label = ctk.CTkLabel(self.main_frame, text="AI Defect Heatmap", font=ctk.CTkFont(size=16, weight="bold"))
        self.heatmap_label.grid(row=0, column=1, padx=10, pady=(20, 0))

        self.img_disp_orig = ctk.CTkLabel(self.main_frame, text="")
        self.img_disp_orig.grid(row=1, column=0, padx=10, pady=10)

        self.img_disp_heatmap = ctk.CTkLabel(self.main_frame, text="")
        self.img_disp_heatmap.grid(row=1, column=1, padx=10, pady=10)

        self.result_label = ctk.CTkLabel(self.main_frame, text="Awaiting Image...", font=ctk.CTkFont(size=24, weight="bold"))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=20)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.after(100, self.load_model)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def load_model(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, 1)
            
            model_path = get_model_path()

            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model = self.model.to(self.device)
                self.model.eval()

                target_layer = self.model.features[-1] 
                self.grad_cam = GradCAM(self.model, target_layer)

                self.status_label.configure(text="Model Status: Ready", text_color="green")
            else:
                self.status_label.configure(text="Model Status: Not Found", text_color="red")
                self.result_label.configure(text=f"Error: Could not find {model_path}", text_color="red")
        except Exception as e:
            self.status_label.configure(text="Model Status: Error", text_color="red")
            self.result_label.configure(text=f"Error: {e}", text_color="red")
            print(f"Error loading model: {e}")

    def select_image(self):
        if self.status_label.cget("text") != "Model Status: Ready":
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, img_path):
        self.result_label.configure(text="Analyzing...", text_color="white")
        self.update()
        
        try:
            image = Image.open(img_path).convert('RGB')
            # Resize image to fit nicely in UI but keep original ratio
            orig_disp = ctk.CTkImage(light_image=image, dark_image=image, size=(320, 320))
            self.img_disp_orig.configure(image=orig_disp, text="")

            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True
            cam, prob = self.grad_cam(img_tensor)

            is_damaged = prob > 0.5

            if is_damaged:
                self.result_label.configure(text=f"DAMAGED ({prob:.1%} confidence)", text_color="#FF4C4C") # Bright Red
            else:
                self.result_label.configure(text=f"CLEAN ({(1-prob):.1%} confidence)", text_color="#00E676") # Bright Green

            heatmap_img = generate_cam_overlay(img_tensor.squeeze(0).cpu(), cam)
            heatmap_disp = ctk.CTkImage(light_image=heatmap_img, dark_image=heatmap_img, size=(320, 320))
            self.img_disp_heatmap.configure(image=heatmap_disp, text="")

        except Exception as e:
            self.result_label.configure(text="Error processing image", text_color="red")
            print(f"Error: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
