import logging
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import json

from bots import QuestBotRL
from threading import Thread

from common import Entity, GPath

from typing import Tuple, List

from .grid import Grid
from .ScreenViewer import ScreenViewer


screen_size = (320*2, 240*2)

class BotUI(Thread):
  def __init__(self, bot: QuestBotRL):
    self.bot: QuestBotRL = bot
    self.is_paused = False
    self.show_grid = False
    Thread.__init__(self)


  def show_entity(self, e: Entity):
    self.msg_info.configure(text=json.dumps(e.e, indent=2))
    # self.canvas_screen.coords(self.rect_selection_id,
    #   e.p.x * 2 - e.s.x + 320,
    #   (240 - e.p.y) * 2 - e.s.y - 240,
    #   e.p.x * 2 + e.s.x + 320,
    #   (240 - e.p.y) * 2 + e.s.y - 240)
    # width = max(15, e.s.x*2)
    # height = max(15, e.s.y*2)
    # self.canvas_screen.coords(self.circle_selection_id,
    #   e.p.x * 2 - width + 320,
    #   (240 - e.p.y) * 2 - height - 240,
    #   e.p.x * 2 + width + 320,
    #   (240 - e.p.y) * 2 + height - 240)
    self.canvas_screen.coords(self.rect_selection_id,
      e.p.x * 2 - e.s.x,
      (240 - e.p.y) * 2 - e.s.y,
      e.p.x * 2 + e.s.x,
      (240 - e.p.y) * 2 + e.s.y)
    width = max(15, e.s.x*2)
    height = max(15, e.s.y*2)
    self.canvas_screen.coords(self.circle_selection_id,
      e.p.x * 2 - width,
      (240 - e.p.y) * 2 - height,
      e.p.x * 2 + width,
      (240 - e.p.y) * 2 + height)


  def clear_path(self):
    if not hasattr(self, 'path_rects'):
      return

    for rect_id in self.path_rects:
      self.canvas_screen.delete(rect_id)


  def show_path(self, path: GPath):
    if not hasattr(self, 'path_rects'):
      self.path_rects: List[int] = []

    i = path.checkpoint.i
    j = path.checkpoint.j
    self.path_rects.append(self.canvas_screen.create_rectangle(
        i*20, (24-j)*20,(i+1)*20,(24 - j -1)*20, fill='yellow', stipple="gray25"))


  def run(self):
    self.root = tk.Tk()

    def on_closing():
      self.bot.stop()
      self.is_paused
      self.root.destroy()
      logging.info('on_closing')

    self.root.protocol("WM_DELETE_WINDOW", on_closing)

    self.canvas_screen = tk.Canvas(self.root, bg='magenta',
        width=screen_size[0] - 2,
        height=screen_size[1] - 2)
    self.canvas_screen.pack(side=tk.TOP)

    self.wall_grid: Grid = Grid(screen_size, self.canvas_screen)
    self.screen_viewer = ScreenViewer(2, self.canvas_screen, self.bot)

    self.btn_pause = tk.Button(self.root, text="Pause",
        command = self.pause_handle)
    self.btn_pause.pack(side=tk.TOP)

    self.btn_show_grid = tk.Button(self.root, text="Show Grid",
        command = self.grid_handle)
    self.btn_show_grid.pack(side=tk.TOP)

    self.btn_reset = tk.Button(self.root, text="Reset",
        command = self.reset_handle)
    self.btn_reset.pack(side=tk.TOP)

    self.lst_entities = tk.Listbox(self.root, selectmode=tk.SINGLE)
    self.lst_entities.pack(side=tk.LEFT, anchor=tk.NW)

    self.rect_selection_id = self.canvas_screen.create_rectangle(0,0,0,0,
        outline='magenta')
    self.circle_selection_id = self.canvas_screen.create_oval(0,0,0,0,
        outline='green2')

    def onselect(evt):
      w: tk.Listbox = evt.widget
      index = int(w.curselection()[0])

      self.clear_path()
      for element in self.elements[index][1:]:
        if isinstance(element, Entity):
          self.show_entity(element)
        elif isinstance(element, GPath):
          self.show_path(element)
        else:
          raise Exception('The type \'{}\' has no show function'.format(
              type(element)))

    self.lst_entities.bind('<<ListboxSelect>>', onselect)

    self.msg_info = tk.Message(self.root)
    self.msg_info.pack(side=tk.LEFT)

    update_th = Thread(target = self.update)
    update_th.daemon = True
    update_th.start()

    self.root.mainloop()


  def grid_handle(self):
    self.wall_grid.toggle()
    if self.wall_grid.is_visible:
      self.btn_show_grid.config(text="Hide Grid")
    else:
      self.btn_show_grid.config(text="Show Grid")


  def pause_handle(self):
    if self.is_paused:
      self.unpause()
    else:
      self.pause()


  def unpause(self):
    self.is_paused = False
    update_th = Thread(target = self.update)
    update_th.daemon = True
    update_th.start()
    self.btn_pause.config(text="Pause")


  def pause(self):
    self.is_paused = True
    self.screen_viewer.update()

    self.btn_pause.config(text="Resume")

    self.elements: List[Tuple[str, object]] = self.bot.get_entities()
    self.lst_entities.delete(0, tk.END)
    self.lst_entities.insert(tk.END, *[e[0] for e in self.elements])
    self.wall_grid.update(self.bot.grid())


  def reset_handle(self):
    self.bot.reset()


  def update(self):
    while True:
      if self.is_paused:
        return
      self.bot.update()
      if self.bot.pause:
        self.pause()

