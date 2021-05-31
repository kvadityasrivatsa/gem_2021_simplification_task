import os
import pickle

def readtxt(path,split=False):

	with open(path,'r') as infile:
		doc = infile.readlines()
	if split: doc = [line.rstrip('\n').split() for line in doc]
	else: doc = [line.rstrip('\n') for line in doc]
	return doc

def writetxt(doc,path,mode='w',join=False,newline=False):

	if not isinstance(doc,list) or not all([isinstance(line,str) for line in doc]):
		raise Exception('only a list of strings is permitted')

	if join:
		if newline:	doc = [' '.join(line)+'\n' for line in doc]
		else: doc = [' '.join(line) for line in doc]
	else:
		if newline:	doc = [line+'\n' for line in doc]
		else: pass

	with open(path,mode) as outfile:
		outfile.writelines(doc)

def readpickle(path):
	with open(path,'rb') as infile:
		return pickle.load(infile)

def writepickle(doc,path,mode='wb'):
	with open(path,mode) as outfile:
		pickle.dump(doc,outfile)

