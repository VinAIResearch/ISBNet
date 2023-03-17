echo Preprocess data
python3 preprocess_scannet200.py
echo Soft link data
ln -s ../scannetv2/test ./
ln -s ../scannetv2/superpoints ./