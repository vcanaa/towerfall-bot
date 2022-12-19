import numpy as np
import tkinter as tk
import json

from bots import QuestBot
from threading import Thread

from PIL import Image, ImageTk

from common import log

class BotUI(Thread):
  def __init__(self, bot: QuestBot):
    self.bot: QuestBot = bot
    self.is_paused = False
    Thread.__init__(self)

  def run(self):
    self.root = tk.Tk()

    self.canvas_screen = tk.Canvas(self.root, bg='magenta', width=320*2 -2, height=240*2 - 2)

    self.image_screen = self.canvas_screen.create_image(0, 0, anchor="nw", image=None)
    self.canvas_screen.pack(side=tk.TOP)

    self.btn_pause = tk.Button(self.root, text="Pause", command = self.pause_handle)
    self.btn_pause.pack(side=tk.TOP)

    self.lst_entities = tk.Listbox(self.root, selectmode=tk.SINGLE)
    self.lst_entities.pack(side=tk.LEFT, anchor=tk.NW)

    self.rect_selection_id = self.canvas_screen.create_rectangle(0,0,0,0, outline='magenta')
    self.circle_selection_id = self.canvas_screen.create_oval(0,0,0,0, outline='green2')

    def onselect(evt):
      w: tk.Listbox = evt.widget
      index = int(w.curselection()[0])
      e = self.entities[index]
      self.msg_info.configure(text=json.dumps(e.e, indent=2))
      # self.canvas_screen.coords(self.rect_selection_id,
      #   e.p.x * 2 - e.s.x + 320,
      #   (240 - e.p.y) * 2 - e.s.y - 240,
      #   e.p.x * 2 + e.s.x + 320,
      #   (240 - e.p.y) * 2 + e.s.y - 240)
      self.canvas_screen.coords(self.rect_selection_id,
        e.p.x * 2 - e.s.x,
        (240 - e.p.y) * 2 - e.s.y,
        e.p.x * 2 + e.s.x,
        (240 - e.p.y) * 2 + e.s.y)
      width = max(15, e.s.x*2)
      height = max(15, e.s.y*2)
      # self.canvas_screen.coords(self.circle_selection_id,
      #   e.p.x * 2 - width + 320,
      #   (240 - e.p.y) * 2 - height - 240,
      #   e.p.x * 2 + width + 320,
      #   (240 - e.p.y) * 2 + height - 240)
      self.canvas_screen.coords(self.circle_selection_id,
        e.p.x * 2 - width,
        (240 - e.p.y) * 2 - height,
        e.p.x * 2 + width,
        (240 - e.p.y) * 2 + height)

    self.lst_entities.bind('<<ListboxSelect>>', onselect)

    self.msg_info = tk.Message(self.root)
    self.msg_info.pack(side=tk.LEFT)

    update_th = Thread(target = self.update)
    update_th.daemon = True
    update_th.start()

    self.root.mainloop()


  def pause_handle(self):
    if self.is_paused:
      self.is_paused = False
      update_th = Thread(target = self.update)
      update_th.daemon = True
      update_th.start()
      self.btn_pause.config(text="Pause")
    else:
      self.is_paused = True
      screen_data = self.bot.get_game_screen()
      log(str(screen_data.shape))

      a = screen_data.reshape(240, 320, 4)
      # a = np.roll(a, int(160 - self.bot.me.e['pos']['x']), axis=1)
      # a = np.roll(a, int(120 + self.bot.me.e['pos']['y']), axis=0)
      self.img = ImageTk.PhotoImage(image=Image.fromarray(a, mode='RGBA').resize((320*2, 240*2), Image.Resampling.NEAREST))
      self.canvas_screen.itemconfig(self.image_screen, image = self.img)
      self.btn_pause.config(text="Resume")

      self.entities = self.bot.entities
      self.lst_entities.delete(0, tk.END)
      self.lst_entities.insert(tk.END, *[e.type for e in self.entities])


  def update(self):
    while True:
      if self.is_paused:
        return
      self.bot.update()