import sys
import json
import random

# The communication is done over stdin. Every line is a json. Every message read requires a reply.

def reply(msg = None):
  if msg:
    sys.stdout.write(msg)
  sys.stdout.write('\n')
  sys.stdout.flush()

def press(b):
  sys.stdout.write(b)

while True:
  # Read the state of the game in a loop.
  gameState = json.loads(sys.stdin.readline())

  # There are three main types to handle, 'init', 'scenario' and 'update'.
  # Check 'type' to handle each accordingly.
  if gameState['type'] == 'init':
    # 'init' is sent every time a match series starts. It contains information about the players and teams.
    # The seed is based on the bot index so each bots acts differently.
    stateInit = gameState
    random.seed(stateInit['index'])
    reply()
    continue

  if gameState['type'] == 'scenario':
    # 'scenario' informs your bot about the current state of the ground. Store this information
    # to use in all subsequent loops. (This example bot doesn't use the shape of the scenario)
    stateScenario = gameState
    reply()
    continue

  if gameState['type'] == 'update':
    # 'update' informs the state of entities in the map (players, arrows, enemies, etc).
    stateUpdate = gameState

  # After receiving an 'update', your bot is expected to output string with the pressed buttons.
  # Each button is represented by a character:
  # r = right
  # l = left
  # u = up
  # d = down
  # j = jump
  # z = dash
  # s = shoot
  # The order of the characters are irrelevant. Any other character is ignored. Repeated characters are ignored.

  # This bot acts based on the position of the other player only. It
  # has a very random playstyle:
  #  - Runs to the enemy when they are below.
  #  - Runs away from the enemy when they are above.
  #  - Shoots when in the same horizontal line.
  #  - Dashes randomly.
  #  - Jumps randomly.

  myState = None
  enemyState = None

  players = []
  for state in stateUpdate['entities']:
    if state['type'] == 'player':
      players.append(state)
      if state['playerIndex'] == stateInit['index']:
        myState = state

  # Bot only does anything if player is in game.
  if myState == None:
    reply()
    continue

  for state in players:
    if state['team'] != myState['team']:
      enemyState = state

  # Bot only does anything if there is a player.
  if enemyState == None:
    reply()
    continue

  myPos = myState['pos']
  enemyPos = enemyState['pos']
  if enemyPos['y'] >= myPos['y']:
    # Runs away if enemy is above
    if myPos['x'] < enemyPos['x']:
      press('l')
    else:
      press('r')
  else:
    # Runs to enemy if they are below
    if myPos['x'] < enemyPos['x']:
      press('r')
    else:
      press('l')

    # If in the same line shoots,
    if abs(myPos['y'] - enemyPos['y']) < enemyState['size']['y']:
      press('s')

  # Presses dash in 1/10 of the loops.
  if random.randint(0, 9) == 0:
    press('z')

  # Presses jump in 1/20 of the loops.
  if random.randint(0, 19) == 0:
    press('j')

  # Issue the command back to the game.
  reply()
