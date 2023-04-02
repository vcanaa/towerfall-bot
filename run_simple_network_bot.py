


import sys
import json
import random
import time

from common import Connection

# The communication is done over a TCP connection using the Connection class.
# via the read() and write() methods.

_HOST = '127.0.0.1'
_PORT = 12024

connection = Connection(_HOST, _PORT)
pressed = set()

last_time_s = 0.0

def reply():
    global last_time_s

    connection.write(json.dumps({
        "type":"commands",
        "command": ''.join(pressed)
    }))
    
    #   print(json.dumps({
    #     "type":"commands",
    #     "command": ''.join(pressed)
    #   }))
    # Always flush after you are done.
    # sys.stdout.flush()
    pressed.clear()
    last_time_s = time.perf_counter()


def press(b):
  pressed.add(b)

state_init = None
state_scenario = None
state_update = None

# When communicating over stdio, make sure logging goes to stderr
print('I started')
# sys.stderr.flush()

# connection.read()
connection.write_instruction('config')

nb_frames = 0
while True:
  # Read the state of the game in a loop.
  game_state = json.loads(connection.read())
#   print(game_state)
#   game_state = json.loads(sys.stdin.readline())

  print('Framerate: ',  1/(time.perf_counter() - last_time_s))

  # There are three main types to handle, 'init', 'scenario' and 'update'.
  # Check 'type' to handle each accordingly.
  if game_state['type'] == 'init':
    # 'init' is sent every time a match series starts. It contains information about the players and teams.
    # The seed is based on the bot index so each bots acts differently.
    state_init = game_state
    random.seed(state_init['index'])
    reply()
    continue

  if game_state['type'] == 'scenario':
    # 'scenario' informs your bot about the current state of the ground. Store this information
    # to use in all subsequent loops. (This example bot doesn't use the shape of the scenario)
    state_scenario = game_state
    reply()
    continue

  if game_state['type'] == 'update':
    # 'update' informs the state of entities in the map (players, arrows, enemies, etc).
    state_update = game_state


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

  my_state = None
  enemy_state = None

  players = []

  if not state_init:
    raise Exception('No state_init')
  if not state_update:
    raise Exception('No state_update')

  for state in state_update['entities']:
    if state['type'] == 'archer':
      players.append(state)
      if state['playerIndex'] == state_init['index']:
        my_state = state

  # Bot only does anything if player is in game.
  if my_state == None:
    reply()
    continue

  for state in players:
    if state['team'] != my_state['team']:
      enemy_state = state

  # Bot only does anything if there is a player.
  if enemy_state == None:
    reply()
    continue

  my_pos = my_state['pos']
  enemy_pos = enemy_state['pos']
  if enemy_pos['y'] >= my_pos['y']:
    # Runs away if enemy is above
    if my_pos['x'] < enemy_pos['x']:
      press('l')
    else:
      press('r')
  else:
    # Runs to enemy if they are below
    if my_pos['x'] < enemy_pos['x']:
      press('r')
    else:
      press('l')

    # If in the same line shoots,
    if abs(my_pos['y'] - enemy_pos['y']) < enemy_state['size']['y']:
      press('s')

  # Presses dash in 1/10 of the loops.
  if random.randint(0, 9) == 0:
    press('z')

  # Presses jump in 1/20 of the loops.
  if random.randint(0, 19) == 0:
    press('j')

  # Issue the command back to the game.
  reply()
