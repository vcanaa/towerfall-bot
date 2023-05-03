import logging


class NoLevelFormatter(logging.Formatter):
  '''Class documentation here'''
  def format(self, record):
    return record.getMessage()


def set_default():
  logging.basicConfig(level=logging.INFO)
  logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())