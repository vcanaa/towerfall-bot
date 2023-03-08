from ui import BotUI
from bots import QuestBotRL
import logging

logging.basicConfig(level=logging.INFO)

BotUI(QuestBotRL()).start()

# BotUI(QuestBot()).start()
# QuestBot().run()
