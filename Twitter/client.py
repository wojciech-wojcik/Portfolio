# coding: utf-8

import tweepy
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
import time
import socket
import json
import configparser
import argparse


class Listener(StreamListener):

    def __init__(self, c, time_limit=20):
        self.i = 0
        self.start_time = time.time()
        self.limit = time_limit
        self.saveFile = open('min.json', 'w')
        self.list = []
        self.c = c
        super(Listener, self).__init__()

    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            self.list.append(data)
            s = json.loads(data)['text'].replace('\n', ' ') + '\n'
            print(self.i, s, end=' ')
            self.c.send(str.encode(s))
            self.i += 1
            if len(self.list) > 1000:
                json.dump(self.list, self.saveFile)
                self.list = []
            return True
        else:
            json.dump(self.list, self.saveFile)
            self.saveFile.close()
            return False

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=str,
                        help="Adres hosta")
    parser.add_argument("port", type=int,
                        help="Port")
    parser.add_argument("time", type=int,
                        help="Czas działania w sekundach")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    c = config.read('data.conf')
    auth = OAuthHandler(config['TWITTER']['consumer_key'], config['TWITTER']['consumer_secret'])
    auth.set_access_token(config['TWITTER']['access_token'], config['TWITTER']['access_secret'])

    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.bind((args.host, args.port))
    socket.listen(5)
    c, addr = socket.accept()

    stream = Stream(auth, Listener(c, args.time))
    location_box = config['TWITTER']['location_box'].split(',')
    location_box = [float(e.strip()) for e in location_box]
    stream.filter(locations=location_box)

    c.close()
    socket.close()
