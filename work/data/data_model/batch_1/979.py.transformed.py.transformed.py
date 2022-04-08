import sys
import time
import socket
import pickle
def run_client(host, port):
	data = ['any', 'object']
 s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.connect((host, port))
 while True:
		data_bytes = pickle.dumps(data)
  s.send(data_bytes)
  print('Send:', type(data), sys.getsizeof(data_bytes))
  time.sleep(0.5)
 pass
def run_server(host, port):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.bind((host, port))
 s.listen(1)
 print('Server Listening')
 conn, addr = s.accept()
 print('Server connected by:', addr)
 while True:
		data_bytes = conn.recv(1024)
  data = pickle.loads(data_bytes)
  print('Received:', type(data), sys.getsizeof(data_bytes))
 pass
if __name__ == '__main__':
	server_host = 'x.x.x.x'
 server_port = 32928
 def get_ip_address(remote_server="8.8.8.8"):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect((remote_server, 80))
  return s.getsockname()[0]
 if get_ip_address() == server_host:
		run_server(server_host, server_port)
 else:
		run_client(server_host, server_port)
