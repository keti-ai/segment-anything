import os
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import cv2
from segment_anything import SamPredictor, sam_model_registry
import coord_transform as ct
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

class ImageClassifierApp:
    def __init__(self, roi_detector, dir_in, csv_okay, csv_fail):
        self.dir_in = dir_in
        self.csv_okay = csv_okay
        self.csv_fail = csv_fail
        self.image_files = sorted([f for f in os.listdir(dir_in) if f.endswith('_color.png')])
        self.current_index = 0
        self.k = None  # Initialize k
        self.d = None  # Initialize d

        self.root = tk.Tk()
        self.root.title("Image Classifier")

        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT)

        self.ok_button = tk.Button(self.root, text="OK", command=self.mark_okay)
        self.ok_button.pack(side=tk.LEFT)

        self.fail_button = tk.Button(self.root, text="Fail", command=self.mark_fail)
        self.fail_button.pack(side=tk.RIGHT)

        self.roi_detector = roi_detector
        self.show_image()
        self.root.mainloop()

    def update_images(self, image_num):
        error_file = f"{image_num}_color.png"
        error_depth = f"{image_num}_depth.png"

        image_path = os.path.join(self.dir_in, error_file)
        depth_path = os.path.join(self.dir_in, error_depth)

        image = Image.open(image_path)
        image = np.array(image)
        depth = Image.open(depth_path)
        depth = np.array(depth)

        output = self.roi_detector.set_image(image)

        # Get all segmentations for the entire image
        masks, _, _ = self.roi_detector.predict(point_coords=None, point_labels=0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        for mask in masks:
            show_mask(mask, ax)

        plt.axis('off')
        fig.canvas.draw()

        return fig, ax

    def show_image(self):
        if self.current_index < len(self.image_files):
            image_num = self.image_files[self.current_index].split('_')[0]
            fig, ax = self.update_images(image_num)

            canvas = FigureCanvasTkAgg(fig, master=self.left_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        else:
            tk.Label(self.left_frame, text="No more images.").pack()

    def mark_okay(self):
        self.save_result(self.csv_okay)
        self.current_index += 1
        self.show_image()

    def mark_fail(self):
        self.save_result(self.csv_fail)
        self.current_index += 1
        self.show_image()

    def save_result(self, csv_file):
        image_file = self.image_files[self.current_index]
        df = pd.DataFrame([image_file], columns=["filename"])
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["default"](checkpoint="/media/syh/ssd2/data/sam_checkpoint/sam_vit_h_4b8939.pth")
    sam.to(device)
    predictor = SamPredictor(sam)

    dir_in = "/media/syh/ssd2/data/order_240523/20240523_trim"
    csv_okay = "okay_dummy_.csv"
    csv_fail = "fail_dummy_.csv"

    app = ImageClassifierApp(predictor, dir_in, csv_okay, csv_fail)
