#!/usr/local/bin/python
import logging
import logging.config
import yaml
import itertools
import os
import io
import json
import sys


class PathParser:

    def __init__(self, config_path):
        self.root = '../'
        self.data = os.path.join(self.root, config_path['data'])
        self.res = os.path.join(self.root, config_path['res'])
        self.graphs = os.path.join(self.root, config_path['graphs'])
        self.checkpoints = os.path.join(self.root, config_path['checkpoints'])

        self.glove = os.path.join(self.res, config_path['glove'])

        self.retrieved = os.path.join(self.data, config_path['tweet_retrieved'])
        self.preprocessed = os.path.join(self.data, config_path['tweet_preprocessed'])
        self.movement = os.path.join(self.data, config_path['price'])
        self.vocab = os.path.join(self.res, config_path['vocab_tweet'])

config_fp = os.path.join(os.path.dirname(__file__), 'config.yml')
config = yaml.load(open(config_fp, 'r'),Loader=yaml.FullLoader)
config_model = config['model']

dates = config['dates']

config_stocks = config['stocks']  # a list of lists
list_of_lists = [config_stocks[key] for key in config_stocks]
stock_symbols = list(itertools.chain.from_iterable(list_of_lists))
ss_size = len(stock_symbols)

path_parser = PathParser(config_path=config['paths'])

with io.open(str(path_parser.vocab), 'r', encoding='utf-8') as vocab_f:
    vocab = json.load(vocab_f)
    vocab_size = len(vocab) + 1  # for unk
