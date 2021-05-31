import os
import re
import numpy as np
from time import sleep
import shutil
import subprocess
import datetime

from .util import readtxt, readpickle, writepickle, writetxt
from .sari import SARIsent
from .ner_tagger import NERuntag 

def scoreCheckpoint(checkpointsDir, dataset, res, deviceCount, 
					split = 'valid',
					cycleDur=20,
					readWaitTime=120, 
					save=True):

	seenFiles = list()
	bestScore, bestCheckpoint = -1, None
	savePath = f"data/{res.evalset}/checkpoint_temp.valid"
	bestPath = os.path.join(checkpointsDir,'checkpoint_SARI_best.pt')
	ledgerPath = f'data/{res.evalset}/validation_ledger.pkl'

	print('\n>> Initializing validation routine')

	while True:

		sleep(cycleDur)

		for filename in os.listdir(checkpointsDir):

			filePath = os.path.join(checkpointsDir,filename)
			if not re.match('checkpoint[0-9]+\.pt',filename) and filePath!=bestPath:
				seenFiles.append(filePath)
				# os.remove(filePath)	
				continue

			if filePath not in seenFiles:	# new valid chackpoint found

				fileMakeTime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
				if datetime.datetime.now() - fileMakeTime < datetime.timedelta(seconds=readWaitTime):
					continue
				else:
					seenFiles.append(filePath)

				print(f'\n>> beginning validation on {filename}\n')

				evalDataPath = os.path.join('data',res.evalset,f'{split}_eval_data')

				generateCmd = f"CUDA_VISIBLE_DEVICES={deviceCount-1} \
fairseq-generate {evalDataPath} --path {filePath} \
--batch-size 128 --beam 8 --remove-bpe > {savePath}"

				os.system(generateCmd)

				output = parseFairSeqOutput(savePath,
											nbchars=res.nbchars,
											levsim=res.levsim,
											wrdrank=res.wrdrank,
											deptree=res.deptree)
				output = {k:v for k,v in sorted(output.items())}
				output = [data['pred'] for data in output.values()]

				if res.ner:
					output = NERuntag(output,evalset=res.evalset,split=split)

				scoringData = [{'src':inSample[0],'pred':outSample,'ref':inSample[1]} for inSample, outSample in zip(dataset,output)] 
				sariScore = scoreOutput(scoringData,evalset=res.evalset,split=split,device=deviceCount-1)

				newBest = sariScore['mean'] > bestScore
				if newBest:
					bestScore = sariScore
					# shutil.move(filePath,bestPath)
					shutil.copy(filePath,bestPath)

				# else:
				# 	os.remove(filePath)

				print(f'\n>> {filename} evaluated on {split} set. SARI: {sariScore}. best={newBest}\n')

				if save:
					if os.path.exists(ledgerPath) and os.path.isfile(ledgerPath):
						evalLedger = readpickle(ledgerPath)
					else:
						evalLedger = {'config':res,
									   'epochs':[],
									   'best':None,
									   'last':None}

					evalLedger['epochs'].append({'original_path':filePath,
												  'score':sariScore})
					if newBest:
						evalLedger['best'] = filePath

					evalLedger['last'] = filePath

					writepickle(evalLedger,ledgerPath)
					print(f'>> saved evaluation ledger to {ledgerPath}')


def parseFairSeqOutput(path,nbchars=None,levsim=None,wrdrank=None,deptree=None):

	tokCount = sum([bool(tok) for tok in [nbchars,levsim,wrdrank,deptree]])
	doc = readtxt(path,split=True)
	out = dict()
	for line in doc:
		if len(line):
			for k,v in [('S','src'),('T','ref'),('H','pred')]:
				if line[0].startswith(k):
					sid = int(line[0].split('-')[1])
					if sid not in out:
						out[sid] = dict()
					if k=='S':
						out[sid][v] =  ' '.join(line[tokCount+1:])
					elif k=='T':
						out[sid][v] =  ' '.join(line[1:])
					else:	# k=='H'
						out[sid][v] =  ' '.join(line[2:])

					break
	return out

# def scoreOutput(doc):
# 	docScore = list()
# 	writetxt([sample['pred'] for sample in doc],'temp_out.txt')
# 	results = subprocess.run("easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' < temp_out.txt",shell=True, capture_output=True)
# 	# os.remove('temp_out.txt')
# 	results = results.stdout.decode('utf-8').rstrip('\n')
# 	print('+++++++++++++++++++++++++++++')
# 	print(results)
# 	print('+++++++++++++++++++++++++++++')
# 	return {'mean':0, 'std':0, 'scores':[]}

def scoreOutput(doc,evalset,split,device=None):
	docScore = list()
	for sample in doc:
		sampleScore = SARIsent(sample['src'],sample['pred'],sample['ref'])
		docScore.append(sampleScore)
	docScore = np.array(docScore)

	doc = [sample['pred'] for sample in doc]
	writetxt(doc,'temp_out.txt',newline=True)

	evalset = 'turkcorpus' if evalset=='turk' else 'asset'

	if not device:
		scoreCmd = f"easse evaluate -t {evalset}_{split} -m 'bleu,sari,fkgl' < temp_out.txt"
	else:
		scoreCmd = f"CUDA_VISIBLE_DEVICES={device} easse evaluate -t {evalset}_{split} -m 'bleu,sari,fkgl' < temp_out.txt"
	output = subprocess.run(scoreCmd,shell=True,capture_output=True)
	# print(output)
	output = [re.sub('[^a-z0-9\.]','',tok) for tok in output.stdout.decode('utf-8').rstrip('\n').split(' ')]
	output = {'sari':float(output[3])}

	return output['sari']

# def scoreOutput(doc):
# 	docScore = list()
# 	for sample in doc:
# 		sampleScore = SARIsent(sample['src'],sample['pred'],sample['ref'])
# 		docScore.append(sampleScore)
# 	docScore = np.array(docScore)
# 	return {'mean':docScore.mean(), 'std':docScore.std(), 'scores':docScore}



