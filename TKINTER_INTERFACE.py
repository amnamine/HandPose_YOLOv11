import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("hand.pt")

# Global image reference
current_image = None
current_image_path = None

# Functions
def load_image():
    global current_image, current_image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return
    try:
        img = Image.open(file_path)
        img.thumbnail((600, 400))  # Resize for display
        current_image = ImageTk.PhotoImage(img)
        current_image_path = file_path
        canvas.create_image(300, 200, image=current_image)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")


def predict_image():
    global current_image, current_image_path
    if not current_image_path:
        messagebox.showwarning("No Image", "Please load an image first.")
        return

    try:
        results = model(current_image_path)  # Run YOLO prediction
        result_img = results[0].plot()  # Render bounding boxes

        # Convert to PIL Image
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_img)
        pil_img.thumbnail((600, 400))
        current_image = ImageTk.PhotoImage(pil_img)
        canvas.create_image(300, 200, image=current_image)
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")


def reset_canvas():
    global current_image, current_image_path
    current_image = None
    current_image_path = None
    canvas.delete("all")
    canvas.create_text(300, 200, text="", font=("Arial", 16), fill="gray")


# --- UI Setup ---
root = tk.Tk()
root.title("YOLO Image Predictor")
root.geometry("1366x768")
root.resizable(False, False)
root.configure(bg="#f4f4f4")

# Canvas to display images
canvas = tk.Canvas(root, width=600, height=400, bg="white", relief="ridge", bd=2)
canvas.pack(pady=20)
canvas.create_text(300, 200, text="", font=("Arial", 16), fill="gray")

# Buttons frame
button_frame = tk.Frame(root, bg="#f4f4f4")
button_frame.pack(pady=10)

btn_load = tk.Button(button_frame, text="Load Image", command=load_image, width=15, bg="#3498db", fg="white", font=("Arial", 12, "bold"), relief="ridge")
btn_load.grid(row=0, column=0, padx=10)

btn_predict = tk.Button(button_frame, text="Predict", command=predict_image, width=15, bg="#2ecc71", fg="white", font=("Arial", 12, "bold"), relief="ridge")
btn_predict.grid(row=0, column=1, padx=10)

btn_reset = tk.Button(button_frame, text="Reset", command=reset_canvas, width=15, bg="#e74c3c", fg="white", font=("Arial", 12, "bold"), relief="ridge")
btn_reset.grid(row=0, column=2, padx=10)

# Run Tkinter loop
root.mainloop()
