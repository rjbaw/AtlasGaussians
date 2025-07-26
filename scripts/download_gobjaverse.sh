#!/bin/sh
python3 ./util/gobjaverse/download_gobjaverse_280k.py ./data/objaverse/gobjaverse/ ./util/gobjaverse/train.json $(nproc)
python3 ./util/gobjaverse/download_gobjaverse_280k.py ./data/objaverse/gobjaverse/ ./util/gobjaverse/test.json $(nproc)
