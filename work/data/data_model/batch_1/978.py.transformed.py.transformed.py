import pickle
import time
def run_client(host, port):
    data = ['any', 'object']
    from multiprocessing.connection import Client
    client = Client((host, port))
    while True:
        data_string = pickle.dumps(data)
        client.send(data_string)
        print('Send', type(data))
        time.sleep(0.5)
def run_server(host, port):
    from multiprocessing.connection import Listener
    server_sock = Listener((host, port))
    print('Server Listening')
    conn = server_sock.accept()
    print('Server Accept')
    while True:
        data_bytes = conn.recv()
        data = pickle.loads(data_bytes)
        print('Received:', type(data))
        data_bytes = pickle.dumps(data)
        conn.send(data_bytes)
if __name__ == '__main__':
    server_host = 'x.x.x.x'
    server_port = 32928
    import socket
    def get_ip_address(remote_server="8.8.8.8"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((remote_server, 80))
        return s.getsockname()[0]
    if get_ip_address() == server_host:
        run_server(server_host, server_port)
    else:
        run_client(server_host, server_port)
