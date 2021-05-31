#!/bin/sh
#/ Usage: train.sh [options] [--] [<shell-script>...]
#/
#/     -h, --help               show help text
#/     -d, --dataset            test dataset [turk, asset] (default: turk)
#/
#/
# Written by -
#	KV Aditya Srivatsa <k.v.aditya@research.iiit.ac.in> and 
#	Monil Gokani <monil.gokani@research.iiit.ac.in>

dataset='turk'

reset () {
	rm -rf $1;
	mkdir $1;
}

reset data/"$dataset"


#--------------------
cd data

TRAINTAR='data-simplification.tar.bz2'
TRAINDIR='wikilarge_trainset'

wget 'https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2'

tar -xjf "$TRAINTAR"
rm -rf 'data-simplification/wikismall'
mv 'data-simplification/wikilarge' "$TRAINDIR" 

mkdir -p "$dataset"/raw_data
mkdir "$dataset"/raw_data/train "$dataset"/raw_data/test "$dataset"/raw_data/valid

cp "$TRAINDIR"/wiki.full.aner.ori.train.src \
	   "$dataset"/raw_data/train/wiki.train.complex
cp "$TRAINDIR"/wiki.full.aner.ori.train.dst \
   "$dataset"/raw_data/train/wiki.train.simple

cd ..
#--------------------

python3 src/fetch_data.py --"dataset" --trg_dir data/"$dataset"/raw_data

mkdir -p data/"$dataset"/tagged_data
cp -R data/"$dataset"/raw_data/* data/"$dataset"/tagged_data/

rm -rf data/"$dataset"/augmented_data ; mkdir data/"$dataset"/augmented_data

mkdir data/"$dataset"/augmented_data/train \
data/"$dataset"/augmented_data/test \
data/"$dataset"/augmented_data/valid

python3 src/preprocess.py --src_dir data/"$dataset"/tagged_data \
--tag_dir data/"$dataset"/tagged_data \
--aug_dir data/"$dataset"/augmented_data \
--nbchars 0.95 --levsim 0.75 --wrdrank 0.75 \
--ner --dataset "$dataset" --preloaded "$preloaded"


rm -rf data/"$dataset"/preprocessed_data ; mkdir data/"$dataset"/preprocessed_data

fairseq-preprocess --source-lang complex --target-lang simple \
--trainpref data/"$dataset"/augmented_data/train/wiki.train \
--validpref data/"$dataset"/augmented_data/valid/wiki.valid \
--testpref data/"$dataset"/augmented_data/test/wiki.test \
--destdir data/"$dataset"/preprocessed_data \
--workers 30

fairseq-train data/"$dataset"/preprocessed_data --save-dir model_checkpoints/"$dataset" \
--lr 0.00011 --lr-scheduler 'fixed' \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-08 \
--dropout 0.2 --arch 'transformer' --warmup-updates 4000 \
--encoder-embed-dim 128 --encoder-ffn-embed-dim 256 \
--decoder-embed-dim 128 --decoder-ffn-embed-dim 256 \
--encoder-layers 2 --encoder-attention-heads 2 \
--decoder-layers 2 --decoder-attention-heads 2 \
--seed 0 --max-tokens 2000 --no-last-checkpoints --max-epoch 50 \
--keep-best-checkpoints 1 