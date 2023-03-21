echo Preprocess data
python3 preprocess_scannet200.py --dataset_root ../scannetv2/scans --output_root ./ --label_map_file ./scannetv2-labels.combined.tsv --train_val_splits_path ../scannetv2/
echo Soft link data
ln -s ../scannetv2/test ./
ln -s ../scannetv2/superpoints ./