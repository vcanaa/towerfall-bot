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
    screen_data = self.bot.get_game_screen()

    # a = screen_data.reshape(240, 320, 4)
    # a = np.roll(a, int(160 - self.bot.me.e['pos']['x']), axis=1)
    # a = np.roll(a, int(120 + self.bot.me.e['pos']['y']), axis=0)
    new_shape = (screen_data.shape[1]*self.scale,
        screen_data.shape[0]*self.scale)
    self.img = ImageTk.PhotoImage(
        image=Image.fromarray(screen_data, mode='RGBA').resize(
            new_shape,
            Image.Resampling.NEAREST))
    self.canvas.itemconfig(self.image_screen, image = self.img)