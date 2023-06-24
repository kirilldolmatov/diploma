# Usefull commands

## Install packages

```bash
conda create -n py37 python=3.7

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

conda install --yes --file requirements.txt

pip install pyspng
```

## Prepare images

```bash
# create dirs
mkdir -p processed/Average
mkdir -p processed/Poor
mkdir -p processed/Good

# process images for each class
cd datasets/raw/Average

mogrify -type TrueColor -set colorspace sRGB -colorspace sRGB -resize 256x256 -background white -gravity center -extent 256x256 -format jpg -path ../../processed/Average *

cd ../Good

mogrify -type TrueColor -set colorspace sRGB -colorspace sRGB -resize 256x256 -background white -gravity center -extent 256x256 -format jpg -path ../../processed/Good *

cd ../Poor

mogrify -type TrueColor -set colorspace sRGB -colorspace sRGB -resize 256x256 -background white -gravity center -extent 256x256 -format jpg -path ../../processed/Poor *

cd ../..
```

## Convert for StyleGAN input format

```bash
python creat_cond_json.py

python dataset_tool.py --source=./datasets/processed/ --dest=./datasets/stylegan_dataset.zip
```

## Train

```bash
python train.py --cond=1 --outdir=./output --data=./datasets/stylegan_dataset.zip --gpus=1 --cfg=paper512 --mirror=1 --snap=10
```

## Calc Metrics

```bash
python calc_metrics.py --metrics=fid50k_full --network=/media/kirill/2tb/output/00001-stylegan_dataset-cond-mirror-paper256/network-snapshot-008547.pkl
```

## Generate

```bash
python generate.py --outdir=out/good --seeds=0-99 --class=3 --network=/media/kirill/2tb/output/00001-stylegan_dataset-cond-mirror-paper256/network-snapshot-008588.pkl
```

## Server hse magic

```bash
module load CUDA/11.7 gnu12/12.1 Python/Anaconda_v11.2020 
module purge
source activate stylegan2
```
