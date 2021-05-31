import os
import re
import shutil
import argparse
import subprocess
import multiprocessing
import torch

from src.data_handling import loadTrainset, loadHuggingfaceDataset, load_pretrained_embeddings
from src.control_tokens import add_control_tokens_to_dataset
from src.ner_tagger import NERtag
from src.cleaner import reformatBracketing, prune, lowercase
from src.util import writetxt, writepickle
from src.scoring import scoreCheckpoint

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--evalset', action='store', dest='evalset', type=str, default=None,
						help="test-validation set ['turk' or 'asset'] (default: 'turk')")
	parser.add_argument('--ner', action='store_true', dest='ner', default=False,
						help="ner tag data before training (default: False)")

	parser.add_argument('--use-pretrained', action='store_true', dest='use_pretrained', default=False,
						help="use pretrained embeddings for training (FastText) (default: False)")
	parser.add_argument('--encoder-embed-path', action='store', dest='encoder_embed_path', type=str, default='data/resources/wiki-news-300d-1M.txt',
						help="path to pretrained embeddings file (default: 'data/resources/wiki-news-300d-1M.txt')")
	parser.add_argument('--decoder-embed-path', action='store', dest='decoder_embed_path', type=str, default='data/resources/wiki-news-300d-1M.txt',
						help="path to pretrained embeddings file (default: 'data/resources/wiki-news-300d-1M.txt')")

	parser.add_argument('--lower', action='store_true', dest='lower', default=True,
						help="convert data to lowercase before training (default: True)")
	parser.add_argument('--nbchars', action='store', dest='nbchars', type=float, default=None,
						help="control-token for compression-ratio (range:[0-1])(default: None)")
	parser.add_argument('--levsim', action='store', dest='levsim', type=float, default=None,
						help="control-token for levenshtein-similarity (range:[0-1])(default: None)")
	parser.add_argument('--wrdrank', action='store', dest='wrdrank', type=float, default=None,
						help="control-token for word-rank (range:[0-1])(default: None)")
	parser.add_argument('--deptree', action='store', dest='deptree', type=float, default=None,
						help="control-token for dependency-tree-depth (range:[0-1])(default: None)")
	parser.add_argument('--rounding', action='store', dest='rounding', type=int, default=2,
						help="number of places of floating-point rounding for control-tokens (default: 2)")

	parser.add_argument('--annotator', action='store', dest='annotator', type=int, default=0,
						help="annotator samples to be used for validation and testing (only for fairseq, not final submission) (turk:[0-7], asset:[0-9]) (default: 0)")

	res = parser.parse_args()

	if torch.cuda.is_available():
		deviceCount = torch.cuda.device_count()
	else:
		raise Exception("CUDA environment not activated.")

	if res.evalset in ['turk','asset']:
		os.makedirs(os.path.join('data',res.evalset),exist_ok=True)
		writepickle(res,os.path.join('data',res.evalset,'train_config.pkl'))
	else:
		raise ValueError(f"--evalset must be set as 'turk' or 'asset' only. '{res.evalset}' was entered")

	#1. loading the datasets

	trainset = loadTrainset(
				url='https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2',
				dataset='wikilarge',
				prefix='wiki.full.aner.ori.train',
				srcSuff='src',
				refSuff='dst')

	validsetRaw, testsetRaw = loadHuggingfaceDataset(datasetPath=res.evalset,
							datasetDir='simplification',
							validationSplit='validation',
							testSplit='test',
							columns=['original','simplifications'])

	#2. preparing data for training

	trainsetSrc, trainsetRef = prune(reformatBracketing([sent['src'] for sent in trainset]), 
									 reformatBracketing([sent['ref'] for sent in trainset]))
	trainset = list(zip(trainsetSrc,trainsetRef))

	# choosing one of the simplifications (by annotator ID) for fairseq training
	# NOTE: final evaluation uses all the simplifications
	print(f'>> setting annotator:{res.annotator} for validation and test split for fairseq-training')
	validset = [[sample[0],sample[1][res.annotator]] for sample in validsetRaw]
	testset = [[sample[0],sample[1][res.annotator]] for sample in testsetRaw]

	trainsetSrc, trainsetRef = zip(*trainset)
	validsetSrc, validsetRef = zip(*validset)
	testsetSrc, testsetRef = zip(*testset)

	dataSplits = [trainsetSrc,trainsetRef,validsetSrc,validsetRef,testsetSrc,testsetRef]

	# NER tag
	if res.ner:
		print('>> NER tagging data')
		dataSplits = NERtag(dataSplits,evalset=res.evalset)

	# lowercase
	if res.lower:
		print('>> converting all data to lowercase')
		for i in range(len(dataSplits)):
			dataSplits[i] = lowercase(dataSplits[i])

	# add control tokens
	if any([res.nbchars,res.levsim,res.wrdrank,res.deptree]):

		res.nbchars = round(res.nbchars,res.rounding) if res.nbchars else None 
		res.levsim = round(res.levsim,res.rounding) if res.levsim else None
		res.wrdrank = round(res.wrdrank,res.rounding) if res.wrdrank else None 
		res.deptree = round(res.deptree,res.rounding) if res.deptree else None
		print(f'>> adding control tokens [nbchars:{res.nbchars}, levsim:{res.levsim}, wrdrank: {res.wrdrank}, deptree:{res.deptree}]')
		
		dataSplits = add_control_tokens_to_dataset(dataSplits,
				 							 evalset=res.evalset,
				 							 nbchars=res.nbchars,levsim=res.levsim,
				 							 wrdrank=res.wrdrank,deptree=res.deptree,
											 rounding=res.rounding)

	trainsetSrc,trainsetRef,validsetSrc,validsetRef,testsetSrc,testsetRef = dataSplits

	fairseqDir = f'data/{res.evalset}/fairseq_data'
	shutil.rmtree(fairseqDir,ignore_errors=True) ; os.makedirs(fairseqDir)

	# print(trainsetSrc[:3])
	writetxt(trainsetSrc,os.path.join(fairseqDir,'train.complex'),newline=True)
	writetxt(trainsetRef,os.path.join(fairseqDir,'train.simple'),newline=True)

	writetxt(validsetSrc,os.path.join(fairseqDir,'valid.complex'),newline=True)
	writetxt(validsetRef,os.path.join(fairseqDir,'valid.simple'),newline=True)

	writetxt(testsetSrc,os.path.join(fairseqDir,'test.complex'),newline=True)
	writetxt(testsetRef,os.path.join(fairseqDir,'test.simple'),newline=True)

	if res.use_pretrained:
		load_pretrained_embeddings(res.encoder_embed_path)
		load_pretrained_embeddings(res.decoder_embed_path)

	#3. training

	preprocessDir = f'data/{res.evalset}/preprocessed_data'
	shutil.rmtree(preprocessDir,ignore_errors=True) ; os.makedirs(preprocessDir)

	preprocCmd = f"fairseq-preprocess --source-lang complex --target-lang simple \
--trainpref {os.path.join(fairseqDir,'train')} \
--validpref {os.path.join(fairseqDir,'valid')} \
--testpref {os.path.join(fairseqDir,'test')} \
--destdir {preprocessDir} \
--workers 30"

	os.system(preprocCmd)

	checkpointsDir = f'checkpoints/{res.evalset}'
	shutil.rmtree(checkpointsDir,ignore_errors=True) ; os.makedirs(checkpointsDir)

#============================================	for validation optimization
	validEvalPath = os.path.join(f'data/{res.evalset}','valid_eval_data')
	testEvalPath = os.path.join(f'data/{res.evalset}','test_eval_data')

	shutil.rmtree(validEvalPath,ignore_errors=True)
	shutil.rmtree(testEvalPath,ignore_errors=True)
	shutil.copytree(f'data/{res.evalset}/preprocessed_data',validEvalPath)
	shutil.copytree(f'data/{res.evalset}/preprocessed_data',testEvalPath)

	for file in os.listdir(validEvalPath):
		testEq = re.sub('valid','test',file)
		shutil.move(os.path.join(validEvalPath,file),os.path.join(validEvalPath,testEq))

	evalSplit = 'valid'
	dataset = validsetRaw if evalSplit=='valid' else testsetRaw
	validProcess = multiprocessing.Process(target=scoreCheckpoint,
		args=(checkpointsDir,dataset,res,deviceCount,evalSplit))
	validProcess.start()

#============================================

# ACCESS Large

# 	trainCmd = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(deviceCount-1)])} \
# fairseq-train {preprocessDir} --save-dir {checkpointsDir} \
# --lr 0.00011 --lr-scheduler 'fixed' \
# --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-08 \
# --dropout 0.2 --arch 'transformer' --warmup-updates 4000 \
# --encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 \
# --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 \
# --encoder-layers 6 --encoder-attention-heads 8 \
# --decoder-layers 6 --decoder-attention-heads 8 \
# --seed 0 --max-tokens 400 --max-epoch 50"


# ACCESS (Large+PreTrained)

# 	trainCmd = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(deviceCount-1)])} \
# fairseq-train {preprocessDir} --save-dir {checkpointsDir} \
# --lr 0.00011 --lr-scheduler 'fixed' \
# --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-08 \
# --dropout 0.2 --arch 'transformer' --warmup-updates 4000 \
# --encoder-embed-dim 300 --encoder-ffn-embed-dim 2048 \
# --decoder-embed-dim 300 --decoder-ffn-embed-dim 2048 \
# --encoder-layers 6 --encoder-attention-heads 6 \
# --decoder-layers 6 --decoder-attention-heads 6 \
# --seed 0 --max-tokens 350 --max-epoch 50 \
# --encoder-embed-path data/resources/wiki-news-300d-1M.txt \
# --decoder-embed-path data/resources/wiki-news-300d-1M.txt"

	trainCmd = [f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(deviceCount-1)])}",
				f"fairseq-train {preprocessDir}",
				f"--save-dir {checkpointsDir}",
				f"--lr 0.00011 --lr-scheduler 'fixed'",
				f"--optimizer adam",
				f"--adam-betas '(0.9, 0.999)'",
				f"--adam-eps 1e-08",
				f"--dropout 0.2",
				f"--arch 'transformer'",
				f"--warmup-updates 4000",
				f"--encoder-embed-dim {300 if res.use_pretrained else 512}",
				f"--encoder-ffn-embed-dim 2048",
				f"--encoder-layers 6",
				f"--encoder-attention-heads {6 if res.use_pretrained else 8}",
				f"--decoder-embed-dim {300 if res.use_pretrained else 512}",
				f"--decoder-ffn-embed-dim 2048",
				f"--decoder-layers 6",
				f"--decoder-attention-heads {6 if res.use_pretrained else 8}",
				f"--max-tokens {res.max_tokens}",
				f"--max-epoch {res.max_epoch}",
				f"--seed 0"
				]



	if res.use_pretrained_embed:
		trainCmd.extend([f"--encoder-embed-path {res.encoder_embed_path}",
						 f"--decoder-embed-path {res.decoder_embed_path}"])

	trainCmd = ' '.join(trainCmd)

# ACCESS Small

# 	trainCmd = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(deviceCount-1)])} \
# fairseq-train {preprocessDir} --save-dir {checkpointsDir} \
# --lr 0.00011 --lr-scheduler 'fixed' \
# --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-08 \
# --dropout 0.2 --arch 'transformer' --warmup-updates 4000 \
# --encoder-embed-dim 128 --encoder-ffn-embed-dim 512 \
# --decoder-embed-dim 128 --decoder-ffn-embed-dim 512 \
# --encoder-layers 3 --encoder-attention-heads 4 \
# --decoder-layers 3 --decoder-attention-heads 4 \
# --seed 0 --max-tokens 2000 --max-epoch 50"

	os.system(trainCmd)
	validProcess.join()

