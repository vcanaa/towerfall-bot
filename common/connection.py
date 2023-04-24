import socket
import json

import logging

from typing import Callable

from typing import Any

_BYTE_ORDER = 'big'
_ENCODING = 'ascii'

class Connection:
  def __init__(self, ip: str, port: int, timeout=0, verbose=0, log_cap=50):
    self.verbose = verbose
    self.log_cap = log_cap
    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._socket.connect((ip, port))
    if timeout:
      self._socket.settimeout(timeout)
    self.port = port
    self.on_close: Callable

  def __del__(self):
    self.close()

  def close(self):
    if hasattr(self, '_socket'):
      if self.verbose > 0:
        logging.info('Closing socket')
      self._socket.close()
      del self._socket
    if hasattr(self, 'on_close'):
      self.on_close()

  def write(self, msg):
    if self.verbose > 0:
      logging.info('Writing: %s', self.cap(msg))
    size = len(msg)
    self._socket.sendall(size.to_bytes(2, byteorder=_BYTE_ORDER))
    self._socket.sendall(msg.encode(_ENCODING))

  def read(self):
    header: bytes = self._socket.recv(2)
    size = int.from_bytes(header, _BYTE_ORDER)
    payload = self._socket.recv(size)
    resp = payload.decode(_ENCODING)
    if self.verbose > 0:
      logging.info('Read: %s', self.cap(resp))
    return resp

  def read_json(self):
    return json.loads(self.read())

  def write_json(self, obj: dict[str, Any]):
    self.write(json.dumps(obj))

  def cap(self, value: str) -> str:
    return value[:self.log_cap] + '...' if len(value) > self.log_cap else value
