import numpy as np
import Levenshtein
import spacy
import pickle as pkl
import nltk
import os
from os import path
from tqdm import tqdm

from .util import readpickle, writepickle

def characterRatio(src,ref):
	if len(src) == 0:
		print('>>> character ratio: zero-exception')
		return 0
	srcScore = 0 
	refScore = 0
	for w in src:
		srcScore+=len(w)
	for w in ref:
		refScore+=len(w)
	return refScore/srcScore

def levenshtein_similarity(src, ref):
	'''
	calculate levenshtein similarity between aligned srclex and refle  sentences
	'''
	return Levenshtein.ratio(src, ref)

def buildRank(fpath):
	logRank = dict()
	wordRank = readpickle(fpath)
	logRank = {w:np.log(wordRank[w]+1) for w in wordRank}
	return logRank

def lexical_complexity(sentence,stopwords, logRank, q=0.75):
	'''
	qth quantile log rank of contituent words
	'''
	sentence = [w for w in sentence if w not in stopwords and w in logRank]
	if len(sentence) == 0:
		return 1
	else:
		logSentence = [logRank[w] for w in sentence]
		return np.quantile(logSentence,q)

def subtree_depth(node):
	'''
	helper to find depth from a given node of dependency tree
	'''
	if len(list(node.children)) == 0:
		return 0
	return 1 + max([subtree_depth(child) for child in node.children])

def dependency_tree_depth(sent):
	'''
	obtain dependency tree of sentence using spacy parser, and find the max depth of that tree
	'''
	tree = PARSER(sent)
	depths = [subtree_depth(sent.root) for sent in tree.sents]
	return max(depths)    

def init_spacy_model(arch):
	try:
		model = spacy.load(arch)
	except:
		spacy.cli.download(arch)
		model = spacy.load(arch)
	return model

def init_nltk_stopwords():
	try:
		stops = nltk.corpus.stopwords.words('english')
	except:
		nltk.download('stopwords')
		stops = nltk.corpus.stopwords.words('english')
	return stops

def add_control_tokens(initSrcDoc,initRefDoc,
						nbchars=None,
						levsim=None,
						wrdrank=None,
						deptree=None,
						force=False,
						rounding=2):

	stops = init_nltk_stopwords()
	parser = init_spacy_model('en_core_web_sm')
	logRank = buildRank('data/resources/FastTextWordRank.pkl')

	trgSrcDoc, trgRefDoc = list(), list()

	for src, ref in tqdm(zip(initSrcDoc, initRefDoc)):

		controlStr = list()

		if nbchars:
			if not force:
				charRatio = characterRatio(src.split(),ref.split())
			else:
				charRatio = nbchars
			controlStr.append('<NbChars_{}>'.format(round(charRatio,rounding)))

		if levsim:
			if not force:
				levRatio = levenshtein_similarity(src, ref)
			else:
				levRatio = levsim
			controlStr.append('<LevSim_{}>'.format(round(levRatio,rounding)))
		
		if wrdrank:
			if not force:
				rankComp, rankSimp = lexical_complexity(src.split(), stops, logRank), lexical_complexity(ref.split(), stops, logRank)
				rankRatio = rankSimp/rankComp if rankComp>0 else 0
			else:
				rankRatio = wrdrank
			controlStr.append('<WordRank_{}>'.format(round(rankRatio,rounding)))
		
		if deptree:
			if not force:
				depComp, depSimp = dependency_tree_depth(src), dependency_tree_depth(ref)
				depRatio = depSimp/depComp if depComp>0 else 0
			else:
				depRatio = deptree
			controlStr.append('<DepTreeDepth_{}>'.format(round(depRatio,rounding)))
	

		trgSrcSent = controlStr if len(controlStr) else []
		trgSrcSent.extend([src])
		trgSrcDoc.append(' '.join(trgSrcSent))

		trgRefDoc.append(ref)

	return trgSrcDoc, trgRefDoc

def add_control_tokens_to_dataset(dataSplits,
								evalset=None,
								nbchars=None,
								levsim=None,
								wrdrank=None,
								deptree=None,
								rounding=2,
								save=True,
								forceUpdate=False):

	renew = False
	savePath = f'data/{evalset}/prepended_data.pkl'
	if forceUpdate or not (os.path.exists(savePath) and os.path.isfile(savePath)):
		renew = True
	else:
		config, prependedData = readpickle(savePath)
		if config!=[nbchars,levsim,wrdrank,deptree]:
			renew = True
		else:
			print('>> reusing prepended data. set forceUpdate=True to force re-tag data')

	if renew:
		print('\n\tprepending train set:')

		trainsetSrc,trainsetRef,validsetSrc,validsetRef,testsetSrc,testsetRef = dataSplits

		trainsetSrc, trainsetRef = add_control_tokens(trainsetSrc,trainsetRef,
								nbchars=nbchars,levsim=levsim,wrdrank=wrdrank,deptree=deptree,
								force=False,rounding=rounding)

		print('\n\tprepending valid set:')
		validsetSrc, validsetRef = add_control_tokens(validsetSrc,validsetRef,
								nbchars=nbchars,levsim=levsim,wrdrank=wrdrank,deptree=deptree,
								force=False,rounding=rounding)

		print('\n\tprepending test set:')
		testsetSrc, testsetRef = add_control_tokens(testsetSrc,testsetRef,
								nbchars=nbchars,levsim=levsim,wrdrank=wrdrank,deptree=deptree,
								force=True,rounding=rounding)

		if save:
			print(f'\n>> saving prepended data to ({savePath})')
			config = [nbchars,levsim,wrdrank,deptree]
			prependedData = [trainsetSrc, trainsetRef, validsetSrc, validsetRef, testsetSrc, testsetRef]
			writepickle([config,prependedData],savePath)

	return prependedData # [trainsetSrc, trainsetRef, validsetSrc, validsetRef, testsetSrc, testsetRef]
