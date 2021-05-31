import os
from tqdm import tqdm

import tarfile
import zipfile
import wget
import shutil

import torch
from datasets import load_dataset
from packaging import version

from .util import readtxt, readpickle, writepickle

def loadHuggingfaceDataset(datasetPath=None,
						datasetDir='simplification',
						validationSplit='validation',
						testSplit='test',
						columns=['original','simplifications'],
						save=True,
						forceUpdate=False):

	print(f'>> loading HuggingFace dataset: ({datasetPath}:{datasetDir})')

	if version.parse(torch.__version__) < version.parse("1.6"):
		from .file_utils import is_apex_available
		if is_apex_available():
			from apex import amp
		_use_apex = True
	else:
		_use_native_amp = True
		from torch.cuda.amp import autocast

	rawData = load_dataset(datasetPath,datasetDir)

	print(f'>> loading validation set {datasetPath}:{datasetDir}...')

	VALID_SAVE_PATH = f'data/resources/{datasetPath}_validset.pkl'

	if not forceUpdate and os.path.exists(VALID_SAVE_PATH) and os.path.isfile(VALID_SAVE_PATH):
		print(f'>> reusing dataset ({VALID_SAVE_PATH}). set forceUpdate=True to force re-downloading the dataset')
		validset = readpickle(VALID_SAVE_PATH)
	else:
		validset = list()
		for sample in tqdm(rawData[validationSplit]):
			validset.append([sample[column] for column in columns])
	print(f'>> validation set loaded: [{len(validset)}] raw sentence pairs found\n')

	print(f'>> loading test set {datasetPath}:{datasetDir}...')
	
	TEST_SAVE_PATH = f'data/resources/{datasetPath}_testset.pkl'

	if not forceUpdate and os.path.exists(TEST_SAVE_PATH) and os.path.isfile(TEST_SAVE_PATH):
		print(f'>> reusing dataset ({TEST_SAVE_PATH}). set forceUpdate=True to force re-downloading the dataset')
		testset = readpickle(TEST_SAVE_PATH)
	else:
		testset = list()
		for sample in tqdm(rawData[testSplit]):
			testset.append([sample[column] for column in columns])
	print(f'>> test set loaded: [{len(testset)}] raw sentence pairs found\n')

	if save:
			os.makedirs(os.path.join(*VALID_SAVE_PATH.split('/')[:-1]),exist_ok=True)
			writepickle(validset,VALID_SAVE_PATH)

			os.makedirs(os.path.join(*TEST_SAVE_PATH.split('/')[:-1]),exist_ok=True)
			writepickle(testset,TEST_SAVE_PATH)

	return validset, testset


def loadTrainset(url=None,dataset=None,prefix=None,srcSuff='src',refSuff='ref',save=True, forceUpdate=False):

	print(f'>> loading train set ({dataset})...')

	TRAIN_SAVE_PATH = 'data/resources/trainset.pkl'

	if not forceUpdate and os.path.exists(TRAIN_SAVE_PATH) and os.path.isfile(TRAIN_SAVE_PATH):
		print(f'>> reusing dataset {dataset} ({TRAIN_SAVE_PATH}). set forceUpdate=True to force re-downloading the dataset')
		trainset = readpickle(TRAIN_SAVE_PATH)

	else:
		filename = wget.download(url)
		if tarfile.is_tarfile(filename):
			tar = tarfile.open(filename)
			dirname = tar.getmembers()[0].name
			tar.extractall()
		else: 
			raise Exception(f'file downloaded from URL not a tarfile. ({filename})')

		trainset_src = readtxt(os.path.join(dirname,dataset,prefix+'.'+srcSuff))
		trainset_ref = readtxt(os.path.join(dirname,dataset,prefix+'.'+refSuff))
		# dirname/ wikilarge/ train/wiki.full.aner.ori.train + . + src

		os.remove(filename)
		shutil.rmtree(dirname,ignore_errors=True)

		if len(trainset_src) != len(trainset_ref):
			raise Exception(f'trainset src-ref length mismatch. ({len(trainset_src)})!=({len(trainset_ref)})')

		trainset = list()
		for src, ref in tqdm(zip(trainset_src,trainset_ref)):
			trainset.append({'src':src,'ref':ref})

		if save:
			os.makedirs(os.path.join(*TRAIN_SAVE_PATH.split('/')[:-1]),exist_ok=True)
			writepickle(trainset,TRAIN_SAVE_PATH)

	print(f'>> train set loaded: [{len(trainset)}] raw sentence pairs found\n')
	return trainset


def load_pretrained_embeddings(path,forceUpdate=False):

	URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'

	if not forceUpdate and os.path.exists(path) and os.path.isfile(path):
		print(f'>> reusing pretrained embeddings ({path}). set forceUpdate=True to force re-downloading the dataset')

	else:
		print(f'>> downloading pretrained embeddings from ({URL}). this may take a while...')
		zipname = wget.download(URL)
		if zipfile.is_zipfile(zipname):
			with zipfile.ZipFile(zipname,'r') as zf:
				print(f'>> extracting file from {zipname}')
				zf.extractall()
				shutil.move(zipfile.rstrip('.zip')+'.txt',path)
				print(f'>> done. file saved to {path}')
