# Visualization guide

1\) Install [PyViz3D](https://github.com/francisengelmann/PyViz3D) (more lightweight than other tools such as Mayavi or Open3D).

```
pip3 install pyviz3d
```

2\) Export the predictions of the trained-model. For example:

```
python3 tools/test.py configs/scannetv2/isbnet_scannetv2.yaml head_isbnet_scannetv2 --out results/isbnet_scannetv2_val
```

3\) The `results` folder is structured as follows.

```
ISBNet
├── results
│   ├── isbnet_scannetv2_val
│   │   ├── pred_instance
│   │   │   ├── predicted_masks
│   │   │   │   ├── scene0011_00_001.txt
│   │   │   │   ├── scene0011_00_002.txt
│   │   │   │   ├── ...
│   │   │   │   ├── scene0011_00_100.txt
│   │   │   │   ├── scene0011_01_001.txt
│   │   │   │   ├── scene0011_01_002.txt
│   │   │   │   ├── ...
│   │   │   ├── scene0011_00.txt
│   │   │   ├── scene0011_01.txt
│   │   │   ├── ...
│   │   │   ├── scene0704_01.txt
```

4\) Visualize the result:

```
python3 visualization/vis_scannetv2.py --data_root dataset/scannetv2 --scene_name scene0011_00 --prediction_path results/isbnet_scannetv2_val --task inst_pred
```

5\) Follow the instructions on the terminal:

```
# open a new terminal and type:
cd ISBNet/visualization/pyviz3d; python -m http.server 6008

# open on your browser to see the result:
http://0.0.0.0:6008
```

6\) You can also follow the instructions from [SoftGroup](https://github.com/thangvubk/SoftGroup) to visualize the results using Open3D ([visualization.py](../tools/visualization.py)).