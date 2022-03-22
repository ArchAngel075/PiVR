import cv2;
import socket as sockets;
import numpy as np;
import math;
import socket;
import argparse;

from cam import process;

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('useSocket', metavar='s', type=int, help='use sockets')
parser.add_argument('identity', metavar='i', type=int, help='the camera identity')
args = parser.parse_args();

stream = None;

global __IDENTITY__;
__IDENTITY__ = args.identity;
global __SOCKETS__;
__SOCKETS__ = args.useSocket == 1;

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 50000        # The port used by the server

def __main__():
    #init here
    init()
    process(stream);


def init():
    print("args",args.useSocket,args.identity)
    stream = cv2.VideoCapture(0)
    if(__SOCKETS__ == 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
        sock.connect((HOST, PORT))

__main__();