from ui import BotUI
from bots import QuestBotRL
import logging

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()


logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

BotUI(QuestBotRL()).start()

# BotUI(QuestBot()).start()
# QuestBot().run()
