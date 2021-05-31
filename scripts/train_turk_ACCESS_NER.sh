#!/bin/bash

dataset='turk'
preloaded='False'

echo "1 PRELOADED $preloaded"

# 1. Download training and validation-test-set

# Load WikiLarge Trainset

if test -d data/"$dataset"

then 

	echo "$dataset data already generated. re-using raw data..."
	preloaded='True'

else

	mkdir data/"$dataset"

	cd data

	TRAINTAR='data-simplification.tar.bz2'
	TRAINDIR='wikilarge_trainset'
	if test -d "$TRAINDIR"

	then echo "re-using $TRAINDIR ..."
		else
		if test -f "$TRAINTAR"
		then
			echo "re-using $TRAINTAR ..."
		else
			echo "$TRAINTAR not found. Downloading..."
			wget 'https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2'
		fi
		tar -xjf "$TRAINTAR"
		rm -rf 'data-simplification/wikismall'
		mv 'data-simplification' "$TRAINDIR" 
	fi

	mkdir -p "$dataset"/raw_data
	mkdir "$dataset"/raw_data/train "$dataset"/raw_data/test "$dataset"/raw_data/valid
	cp "$TRAINDIR"/wikilarge/wiki.full.aner.ori.train.src \
	   "$dataset"/raw_data/train/wiki.train.complex
	cp "$TRAINDIR"/wikilarge/wiki.full.aner.ori.train.dst \
	   "$dataset"/raw_data/train/wiki.train.simple
	cd ..

	# Load TurkCorpus
	echo ">> Downloading training and validation-test-set"
	python3 src/fetch_data.py --turk --trg_dir data/"$dataset"/raw_data
fi

echo "2 PRELOADED $preloaded"

# 2. Adding control & NER tokens to raw_data
if [ $preloaded == 'True' ]
then
	echo "found tagged data"
else
	mkdir -p data/"$dataset"/tagged_data
	cp -R data/"$dataset"/raw_data/* data/"$dataset"/tagged_data/
fi

echo "3 PRELOADED $preloaded"

# rm -rf data/"$dataset"/augmented_data ; mkdir data/"$dataset"/augmented_data
# mkdir data/"$dataset"/augmented_data/train data/"$dataset"/augmented_data/test data/"$dataset"/augmented_data/valid

# python3 src/preprocess.py --src_dir data/"$dataset"/tagged_data \
# --tag_dir data/"$dataset"/tagged_data \
# --aug_dir data/"$dataset"/augmented_data \
# --nbchars 0.95 --levsim 0.75 --wrdrank 0.75 \
# --ner --dataset "$dataset" --preloaded "$preloaded"

# 3. Preprocess data for training
rm -rf data/"$dataset"/preprocessed_data ; mkdir data/"$dataset"/preprocessed_data
fairseq-preprocess --source-lang complex --target-lang simple \
--trainpref data/"$dataset"/augmented_data/train/wiki.train \
--validpref data/"$dataset"/augmented_data/valid/wiki.valid \
--testpref data/"$dataset"/augmented_data/test/wiki.test \
--destdir data/"$dataset"/preprocessed_data \
--workers 30

rm -rf data/"$dataset"/validation_routine
cp -R data/"$dataset"/preprocessed_data data/"$dataset"/validation_routine
for file in data/"$dataset"/validation_routine/*
	do
		rsync -a $file $(echo $file | sed -e 's/valid\./test\./g')
	done


mkdir -p model_checkpoints model_checkpoints/"$dataset"
# rm -rf models/"$dataset" ; mkdir models/"$dataset"

metric='sari'

# setting up validation scoring 
python3 src/validation.py --init \
--dataset "$dataset" --metric "$metric" -N 5 \
--record data/"$dataset"/validation_summary.pkl

# 4. Model Training
fairseq-train data/"$dataset"/preprocessed_data --save-dir model_checkpoints/"$dataset" \
--lr 0.00011 --lr-scheduler 'fixed' \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-08 \
--dropout 0.2 --arch 'transformer' --warmup-updates 4000 \
--encoder-embed-dim 128 --encoder-ffn-embed-dim 256 \
--decoder-embed-dim 128 --decoder-ffn-embed-dim 256 \
--encoder-layers 2 --encoder-attention-heads 2 \
--decoder-layers 2 --decoder-attention-heads 2 \
--seed 0 --max-tokens 2000 --no-last-checkpoints --max-epoch 50 \
--keep-best-checkpoints 0 &


# fairseq-train data/"$dataset"/preprocessed_data --save-dir model_checkpoints/"$dataset" \
# --lr 0.00011 --lr-scheduler 'fixed' \
# --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-08 \
# --dropout 0.2 --arch 'transformer' --warmup-updates 4000 \
# --encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 \
# --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 \
# --encoder-layers 6 --encoder-attention-heads 8 \
# --decoder-layers 6 --decoder-attention-heads 8 \
# --seed 0 --max-tokens 500 --no-last-checkpoints --max-epoch 50



