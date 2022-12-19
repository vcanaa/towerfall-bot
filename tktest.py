import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import time

img2 = None

def handle():
  global img2
  a2 = np.ones((60, 60, 3), dtype=np.uint8)*255
  img2 =  ImageTk.PhotoImage(image=Image.fromarray(a2, mode='RGB'))
  canvas_screen.itemconfig(image_screen, image = img2)

root = tk.Tk()
a1 = np.ones((60, 60, 3), dtype=np.uint8)*124
img1 =  ImageTk.PhotoImage(image=Image.fromarray(a1, mode='RGB'))

canvas_screen = tk.Canvas(root, bg='magenta', width=340, height=340)
canvas_screen.pack()

image_screen = canvas_screen.create_image(10, 20, anchor="nw", image=img1)

btn_pause = tk.Button(root, text="Update", command = handle)
btn_pause.pack()

root.mainloop()

# while True:
#   root.update_idletasks()
#   root.update()
#   time.sleep(0.05) #the actual number doesn't matter

