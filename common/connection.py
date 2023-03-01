import socket

BYTE_ORDER = 'big'
ENCODING = 'ascii'

class Connection:
  def __init__(self, ip: str, port: int):
    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._socket.connect((ip, port))

  def __del__(self):
    self.close()

  def close(self):
    if self._socket:
      self._socket.close()
      del self._socket

  def write(self, msg):
    size = len(msg)
    self._socket.sendall(size.to_bytes(2, byteorder=BYTE_ORDER))
    self._socket.sendall(msg.encode(ENCODING))

  def read(self):
    header: bytes = self._socket.recv(2)
    size = int.from_bytes(header, BYTE_ORDER)
    payload = self._socket.recv(size)
    return payload.decode(ENCODING)
