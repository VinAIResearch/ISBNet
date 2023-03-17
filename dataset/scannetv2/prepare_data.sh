#!/bin/bash
echo Copy data
python3 split_data.py
echo Preprocess data
python3 prepare_data_inst.py --data_split train
python3 prepare_data_inst.py --data_split val
python3 prepare_data_inst.py --data_split test
python3 prepare_superpoint.py