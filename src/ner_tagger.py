import os
import re
from tqdm import tqdm

from multiprocessing import Pool

from flair.data import Sentence
from flair.models import SequenceTagger

from .util import readtxt, writetxt, readpickle, writepickle

def sentenceNER(sentData):
	sentId, srcSent, refSent, srcNER, refNER = sentData
	tagLog = dict()
	mapList = list()

	for srcTok in srcNER:
		tokStr, tokTag = srcTok
		tagId = None
		if tokTag not in tagLog:	# first instance of tag-type
			tagLog[tokTag] = 2
			tagId = 1
		else:
			tagId = tagLog[tokTag]
			tagLog[tokTag] += 1
		mapList.append(('{}@{}'.format(tokTag,tagId),tokStr))	

	for replTag,replStr in mapList:	# replacing mapped strs with tags
		try:
			srcSent = re.sub(replStr,replTag,srcSent,count=1)
			refSent = re.sub(replStr,replTag,refSent,count=1)
		except:
			return None, None, None, None

	return sentId, srcSent, refSent, mapList

def corpusNERtag(initSrcDoc,initRefDoc,
				model='ner-ontonotes',
				workers=300):

	tagger = SequenceTagger.load(model)

	interSrcDoc = [Sentence(sentence) for sentence in initSrcDoc]
	tagger.predict(interSrcDoc,mini_batch_size=workers)

	initSrcNERlist = list()
	for sentence in interSrcDoc:
		entityDict = sentence.to_dict(tag_type='ner')
		initSrcNERlist.append([[entity['text'],str(entity["labels"][0]).split()[0]] \
											for entity in entityDict['entities']])


	interRefDoc = [Sentence(sentence) for sentence in initRefDoc]
	tagger.predict(interRefDoc,mini_batch_size=workers)

	initRefNERlist = list()
	for sentence in interRefDoc:
		entityDict = sentence.to_dict(tag_type='ner')
		initRefNERlist.append([[entity['text'],str(entity["labels"][0]).split()[0]] \
											for entity in entityDict['entities']])


	trgMapList, trgSrcDoc, trgRefDoc = list(), list(), list()

	for i in tqdm(range(len(initSrcDoc))):
		
		sentData = [i, initSrcDoc[i],initRefDoc[i],initSrcNERlist[i],initRefNERlist[i]]
		_, trgSrcSent, trgRefSent, mapList = sentenceNER(sentData)
		if trgSrcSent:
			trgMapList.append(mapList)
			trgSrcDoc.append(trgSrcSent)	#appended without newline
			trgRefDoc.append(trgRefSent)	#appended without newline

	print(f'>>> mapping complete: ({len(trgSrcDoc)}) sentence pairs')

	return trgMapList, trgSrcDoc, trgRefDoc

def NERtag(dataSplits,evalset=None,save=True,forceUpdate=False):

	savePath = f'data/{evalset}/tagged_data.pkl'
	
	if forceUpdate or not (os.path.exists(savePath) and os.path.isfile(savePath)):

		trainsetSrc,trainsetRef,validsetSrc,validsetRef,testsetSrc,testsetRef = dataSplits

		trainMapping, trainsetSrc, trainsetRef = corpusNERtag(trainsetSrc,trainsetRef)
		validMapping, validsetSrc, validsetRef = corpusNERtag(validsetSrc,validsetRef)
		testMapping, testsetSrc, testsetRef = corpusNERtag(testsetSrc,testsetRef)

		dataSplits = [trainsetSrc,trainsetRef,validsetSrc,validsetRef,testsetSrc,testsetRef]

		if save:
			print(f'\n>> saving tagged data to ({savePath})')
			taggedMapping = {'train':trainMapping,'valid':validMapping,'test':testMapping}
			writepickle([taggedMapping,dataSplits],savePath)

	else:

		print(f'>> reusing tagged data. set forceUpdate=True to force re-tag data')
		taggedMapping, dataSplits = readpickle(savePath)

	return dataSplits

def NERuntag(data,evalset=None,split=None):

	mappingPath = f'data/{evalset}/tagged_data.pkl'
	if os.path.exists(mappingPath) and os.path.isfile(mappingPath):
		mapping,_ = readpickle(mappingPath)
		mapping = mapping[split]
	else:
		raise FileNotFoundError(f"NERuntag: No such mapping file found:'{mappingPath}'")

	if len(data)!=len(mapping):
		raise Exception('NERuntag: length mismatch between data and mapping.')

	print('>> untagging generated output')
	outData = list()
	for sentData, sentMapping in zip(data,mapping):
		for replTag, replStr in sentMapping:
			replTag = replTag.lower()
			replStr = replStr.lower()
			sentData = re.sub(replTag,replStr,sentData,count=1)
		outData.append(sentData) 

	return outData
