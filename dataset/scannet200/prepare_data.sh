echo Preprocess data
python3 preprocess_scannet200.py --dataset_root ../../dataset/scannetv2/scans --output_root ./ --label_map_file ./scannetv2-labels.combined.tsv
echo Soft link data
ln -s ../scannetv2/test ./
ln -s ../scannetv2/superpoints ./