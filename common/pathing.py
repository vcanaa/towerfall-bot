from .common import *

from typing import Optional, Tuple, List

class PathCell:
  def __init__(self, i, j, depth):
    self.i = i
    self.j = j
    self.pos = Vec2(5+i*10, 5+j*10)
    self.depth = depth
    self.visited = False
    self.prev: Optional[PathCell] = None
    self.next: Optional[PathCell] = None
    self.entities = []


class GPath:
  def __init__(self, start: PathCell, end: PathCell, checkpoint: PathCell):
    self.start = start
    self.end = end
    self.checkpoint = checkpoint


  def dir(self):
    return Vec2(self.checkpoint.i - self.start.i, self.checkpoint.j - self.start.j)


class PathGrid:
  def __init__(self):
    self.grid = [[PathCell(i, j, 0) for j in range(24)] for i in range(32)]


  def cell(self, i, j) -> PathCell:
    return self.grid[i % 32][j % 24]


  def add_to_visit(self, fro: PathCell, to: PathCell, to_visit: List[PathCell]):
    # if fro.next:
    #   raise Exception('({}, {}) already linked to ({}, {})'.format(fro.i, fro.j, fro.next.i, fro.next.j))
    # fro.next = to
    if to.prev:
      raise Exception('({}, {}) already linked to ({}, {})'.format(to.i, to.j, to.prev.i, to.prev.j))
    to.prev = fro
    to_visit.append(to)
    to.visited = True


  def is_blocked(self, cell, wall):
    return wall[cell.i][cell.j] != 0


  def create_path(self, start: PathCell, end: PathCell, wall) -> GPath:
    if start == end:
      return GPath(start, end, start)

    next = end
    cell = end.prev
    while cell:
      cell.next = next
      # logging.info('link ({}, {}) to ({}, {})'.format(cell.i, cell.j, cell.next.i, cell.next.j))
      if cell == start:
        break
      next = cell
      cell = next.prev

    if cell != start:
      raise Exception('A contiguous path between start and end is expected')

    prev = start
    cell = start.next
    while cell:
      # logging.info('create_path: {} {}'.format(cell.i, cell.j))
      if not isCleanPath(start.pos, cell.pos, wall):
        return GPath(start, end, prev)
      if cell == end:
        return GPath(start, end, end)
      prev = cell
      cell = cell.next
    #logging.info("{} {}".format(start.i, start.j))
    #logging.info("{} {}".format(end.i, end.j))
    #logging.info("{} {}".format(prev.i, prev.j))
    raise Exception('A contiguous path between start and end is expected')


  def getClosestEntity(self, startPos: Vec2, entities: List[Entity], wall) -> Tuple[Optional[Entity], Optional[GPath]]:
    for e in entities:
      p = grid_pos(e.p)
      self.cell(p[0], p[1]).entities.append(e)

    p = grid_pos(startPos)
    startCell = self.cell(p[0], p[1])
    to_visit: List[PathCell] = [startCell]
    while to_visit:
      cell = to_visit.pop(0)
      for e in cell.entities:
        return e, self.create_path(startCell, cell, wall)

      def visit_next(i, j):
        next = self.cell(cell.i + i, cell.j + j)
        if not next.visited and not self.is_blocked(next, wall):
          self.add_to_visit(cell, next, to_visit)

      visit_next(0, 1)
      visit_next(0, -1)
      visit_next(1, 0)
      visit_next(-1, 0)

    return None, None