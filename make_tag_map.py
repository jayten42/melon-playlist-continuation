# -*- coding: utf-8 -*-
import copy
import random

import fire
import numpy as np

from arena_util import load_json
from arena_util import write_json


def run(train_fname, val_fname, test_fname):
    tags = set()
    print("Reading train data...\n")
    playlists_train = load_json(train_fname)
    print("Reading val data...\n")
    playlists_val = load_json(val_fname)
    print("Reading test data...\n")
    playlists_test = load_json(test_fname)
    print("Get tags...\n")
    for ply in playlists_train + playlists_test + playlists_val:
        tags.update(ply['tags'])
    tag_to_id = {tag: i for i, tag in enumerate(list(tags))}
    id_to_tag = {i: tag for i, tag in enumerate(list(tags))}
    print("Write  tag_to_id.json...\n")
    write_json(tag_to_id, 'tag_to_id.json')
    print("Write  id_to_tag.json...\n")
    write_json(id_to_tag, 'id_to_tag.json')


if __name__ == "__main__":
    fire.Fire()
