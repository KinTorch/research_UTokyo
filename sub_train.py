
import pickle
import socket

from client import Client
import argparse




print('sub train start')

while True:
    sk = socket.socket()
    sk.connect(('localhost', 8888))

    rev_data = bytes()

    while True:

        temp = sk.recv(10485760)

        rev_data += temp

        if rev_data.endswith('end'.encode()):
            rev_data = rev_data[:-3]
            break



    client = pickle.loads(rev_data)
    print('client {} received.'.format(client.id))
    client.train()

    data = pickle.dumps(client.get_weight())
    data = data + 'end'.encode()
    sk.send(data)
    print('client {} send.'.format(client.id))
    sk.close()
    del client
    del data
    del rev_data
    del sk

