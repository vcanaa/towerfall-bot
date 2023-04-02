import tkinter as tk

from PIL import Image, ImageTk

from bots import Bot

from typing import Tuple

class ScreenViewer:
  def __init__(self,
      scale: int,
      canvas: tk.Canvas,
      bot: Bot):
    self.scale = scale
    self.canvas = canvas
    self.bot = bot
    self.image_screen = self.canvas.create_image(0, 0, anchor="nw",
        image=None)

  def update(self):
    image: Image.Image
    def get_data(mv):
      nonlocal image
      image = Image.frombytes(data=mv.tobytes(), size=(320, 240),  mode='RGBA')
    self.bot.get_game_screen(get_data)
    self.img = ImageTk.PhotoImage(image=image.resize((640, 480),Image.Resampling.NEAREST))
    self.canvas.itemconfig(self.image_screen, image = self.img)