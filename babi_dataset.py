# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:18:46 2017

@author: sakurai
"""

from collections import Counter
from pathlib import Path
from six.moves import cPickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
from six.moves.urllib import request

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable


def download(url, dataset_dirpath):
    '''
    Args:
        url (str): URL to be downloaded.
        dataset_dirpath (str or pathlib.Path): Directory path to save the file.
    Returns:
        dataset_filepath (str): Full path of the downloaded file.
    '''
    dataset_dirpath = Path(dataset_dirpath)
    dataset_dirpath.mkdir(exist_ok=True)
    filename = url.split('/')[-1]
    dataset_filepath = dataset_dirpath / filename

    if not dataset_filepath.exists():
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; '
                                 'rv:36.0) Gecko/20100101 Firefox/36.0'}
        response = request.urlopen(request.Request(url, None, headers))
        print('Now downloading "{}" ...'.format(url))
        with open(dataset_filepath, 'wb') as f:
            f.write(response.read())
        print('Done.')
    return str(dataset_filepath)


def load_babi():
    # download and extract the dataset
    url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
    dataset_dirpath = r'E:/Dataset/babi'
#    dataset_dirpath = r'C:/Dataset/babi'
    dataset_filepath = download(url, dataset_dirpath)
    dirname = Path(dataset_filepath).name.split('.')[0]
    if not Path(dataset_dirpath, dirname).exists():
        shutil.unpack_archive(dataset_filepath, dataset_dirpath)

    # load dataset
    subset = 'en-10k'
    data_dirpath = Path(dataset_dirpath, 'tasks_1-20_v1-2')
    subset_dirpath = Path(data_dirpath, subset)
    if not subset_dirpath.joinpath('data.pkl').exists():
        dataset = {}
        for train_or_test in ['train', 'test']:
            tasks = []
            for task in range(1, 21):
                pattern = 'qa{}_*_{}.txt'.format(task, train_or_test)
                task_file_path = next(subset_dirpath.rglob(pattern))

                stories = _load_task_file(task_file_path)
                tasks.append(stories)
            dataset[train_or_test] = tasks

        word_types = set('-')
        for tasks in dataset.values():
            for stories in tasks:
                for story in stories:
                    for text, answers, _ in story:
                        for token in text.split(' '):
                            word_types.add(token)
                        if answers:
                            for token in answers.split(' '):
                                word_types.add(token)
        word_types = list(word_types)

        with open(Path(data_dirpath, subset, 'data.pkl'), 'wb') as f:
            cPickle.dump(dataset, f)

        with open(Path(data_dirpath, subset, 'vocab.pkl'), 'wb') as f:
            cPickle.dump(word_types, f)

    with open(Path(data_dirpath, subset, 'data.pkl'), 'rb') as f:
        dataset = cPickle.load(f)

    with open(Path(data_dirpath, subset, 'vocab.pkl'), 'rb') as f:
        word_types = cPickle.load(f)

    return dataset['train'], dataset['test'], word_types


def _load_task_file(task_file_path):
    with open(task_file_path) as f:
        lines = f.readlines()

    stories = []
    story = []
    for i, line in enumerate(lines):
        n, body = _parse_line(line)
        if n == 1 and i != 0:
            stories.append(story)
            story = []

        record = _parse_body(body)
        story.append(record)
    stories.append(story)
    return stories


def _parse_line(line):
    ws_index = line.index(' ')
    n = int(line[:ws_index])
    body = line[ws_index+1:]
    return n, body


def _parse_body(body):
    body = body.replace('.', ' .')
    body = body.replace('?', ' ?')
    body = body.replace(',', ' ')
    parts = body.strip().lower().split('\t')
    text = parts[0].strip()
    if len(parts) == 3:
        answer, support = parts[1], parts[2]
    else:
        answer, support = None, None
    return text, answer, support


if __name__ == '__main__':
    train, test, vocab = load_babi()

    word_to_id = {}
    for i, word_type in enumerate(vocab):
        word_to_id[word_type] = i

    task = train[0]
    story = task[0]
    x = []
    t = []
    for task in train:
        for story in task:
            sequences = []
            targets = []
            for record in story:
                sentence, answers, support = record
                seq = [word_to_id[token] for token in sentence.split(' ')]
                sequences.append(seq)
                target = [-1] * len(seq)
                targets.append(target)
                if answers:
                    seq = [word_to_id['-'] for _ in answers.split(' ')]
                    sequences.append(seq)
                    target = [word_to_id[tok] for tok in answers.split(' ')]
                    targets.append(target)
            sequence = sum(sequences, [])  # join lists
            target = sum(targets, [])
            x.append(sequence)
            t.append(target)
