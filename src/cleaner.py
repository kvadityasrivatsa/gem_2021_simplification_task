import re
from tqdm import tqdm

def reformatBracketing(doc):

	processedLines = list()
	for sent in tqdm(doc):

		sent = re.sub('-LRB-','(',sent)
		sent = re.sub('-RRB-',')',sent)
		sent = re.sub('-LSB-','(',sent)
		sent = re.sub('-RSB-',')',sent)
		sent = re.sub('-LCB-','(',sent)
		sent = re.sub('-RCB-',')',sent)
		processedLines.append(sent)

	return processedLines

def prune(initSrc,initRef,
		MIN_SENT_LEN=3,
		MIN_COMPR_RATIO=0.2,
		MAX_COMPR_RATIO=1.5):

	prunedSrc, prunedRef = list(), list()
	for src, ref in zip(initSrc,initRef):
		_src, _ref = src.split(), ref.split()
		if True:
			if True:
		# if len(_src) > MIN_SENT_LEN and len(_ref) > MIN_SENT_LEN:
		# 	if len(_ref)/len(_src) > MIN_COMPR_RATIO and len(_ref)/len(_src) < MAX_COMPR_RATIO:
				prunedSrc.append(src)
				prunedRef.append(ref)
	return prunedSrc, prunedRef

def lowercase(doc):
	return [sent.lower() for sent in doc]