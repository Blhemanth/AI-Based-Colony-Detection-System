from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

# Load trained model
model = YOLO("runs/detect/train-3/weights/best.pt")

current_image = None  # for saving processed image


# ================= PROCESS IMAGE ================= #
def process_image(path):
    results = model(path, conf=0.5)

    for r in results:
        count = 0

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            if float(box.conf[0]) > 0.5 and area > 80:
                count += 1

        img = r.plot()

        cv2.putText(img, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img, count


# ================= DISPLAY IMAGES ================= #
def display_images(original, processed):
    # Original Image
    orig = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    orig = Image.fromarray(orig)
    orig.thumbnail((300, 300))
    orig = ImageTk.PhotoImage(orig)

    original_panel.config(image=orig)
    original_panel.image = orig

    # Processed Image
    proc = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    proc = Image.fromarray(proc)
    proc.thumbnail((300, 300))
    proc = ImageTk.PhotoImage(proc)

    processed_panel.config(image=proc)
    processed_panel.image = proc


# ================= UPLOAD FUNCTION ================= #
def upload_image():
    global current_image

    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not path:
        return

    result_label.config(text="Processing...")
    root.update()

    original = cv2.imread(path)

    processed_img, count = process_image(path)
    current_image = processed_img

    display_images(original, processed_img)

    result_label.config(text=f"Detected Colonies: {count}")


# ================= SAVE FUNCTION ================= #
def save_image():
    global current_image

    if current_image is None:
        result_label.config(text="No image to save!")
        return

    path = filedialog.asksaveasfilename(defaultextension=".jpg")

    if path:
        cv2.imwrite(path, current_image)
        result_label.config(text="Image saved successfully!")


# ================= GUI DESIGN ================= #
root = tk.Tk()
root.title("AI Bacterial Colony Counter")
root.geometry("750x650")
root.configure(bg="#1e1e1e")

# Title
title = tk.Label(root, text="Bacterial Colony Counter",
                 font=("Arial", 20, "bold"),
                 bg="#1e1e1e", fg="white")
title.pack(pady=10)

# Buttons
btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame, text="Upload Image",
                       command=upload_image,
                       font=("Arial", 12),
                       bg="#4CAF50", fg="white",
                       width=15)
upload_btn.grid(row=0, column=0, padx=10)

save_btn = tk.Button(btn_frame, text="Save Result",
                     command=save_image,
                     font=("Arial", 12),
                     bg="#2196F3", fg="white",
                     width=15)
save_btn.grid(row=0, column=1, padx=10)

# Labels for images
label_frame = tk.Frame(root, bg="#1e1e1e")
label_frame.pack()

tk.Label(label_frame, text="Original Image",
         bg="#1e1e1e", fg="white", font=("Arial", 12)).grid(row=0, column=0)

tk.Label(label_frame, text="Detected Image",
         bg="#1e1e1e", fg="white", font=("Arial", 12)).grid(row=0, column=1)

# Image display frame
frame = tk.Frame(root, bg="#1e1e1e")
frame.pack()

original_panel = tk.Label(frame, bg="#1e1e1e")
original_panel.grid(row=0, column=0, padx=10, pady=10)

processed_panel = tk.Label(frame, bg="#1e1e1e")
processed_panel.grid(row=0, column=1, padx=10, pady=10)

# Result label
result_label = tk.Label(root, text="Detected Colonies: 0",
                        font=("Arial", 14),
                        bg="#1e1e1e", fg="#00FF00")
result_label.pack(pady=10)

# Footer
footer = tk.Label(root, text="AI-Based Colony Detection System",
                  font=("Arial", 10),
                  bg="#1e1e1e", fg="gray")
footer.pack(side="bottom", pady=10)

root.mainloop()