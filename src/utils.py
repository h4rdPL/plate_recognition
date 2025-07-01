import os
import cv2
import matplotlib.pyplot as plt

def show_image(img, title=""):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def list_images(folder, exts=('.jpg', '.jpeg', '.png')):
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]
