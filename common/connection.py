import socket
import json

import logging

from typing import Optional

_BYTE_ORDER = 'big'
_ENCODING = 'ascii'

class Connection:
  def __init__(self, ip: str, port: int):
    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._socket.connect((ip, port))
    self.next_read = True

  def __del__(self):
    self.close()

  def close(self):
    if hasattr(self, '_socket'):
      print('Closing socket')
      self._socket.close()
      del self._socket

  def write(self, msg):
    if self.next_read:
      raise Exception('Write called before read.')
    size = len(msg)
    self._socket.sendall(size.to_bytes(2, byteorder=_BYTE_ORDER))
    self._socket.sendall(msg.encode(_ENCODING))
    self.next_read = True

  def read(self):
    if not self.next_read:
      raise Exception('Read called before write.')
    header: bytes = self._socket.recv(2)
    size = int.from_bytes(header, _BYTE_ORDER)
    payload = self._socket.recv(size)
    self.next_read = False
    return payload.decode(_ENCODING)

  def write_instruction(self, type: str, command: str ='', pos: Optional[dict] = None):
    resp: dict['str', object] = {
      'type': type,
      'command': command
    }

    if pos:
      resp['pos'] = pos
    # logging.info('pos: %s', pos)
    # logging.info(resp)
    self.write(json.dumps(resp))

  def write_reset(self, pos: Optional[dict] = None):
    self.write_instruction('config', pos=pos)

  def write_soft_reset(self, pos: Optional[dict] = None):
    self.write_instruction('softReset', pos=pos)