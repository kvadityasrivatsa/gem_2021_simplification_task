import os
import re
import argparse
import shutil 

from src.ner_tagger import NERuntag 
from src.data_handling import loadHuggingfaceDataset
from src.util import readpickle, writetxt
from src.scoring import parseFairSeqOutput, scoreOutput

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--evalset', action='store', dest='evalset', type=str, default=None,
						help="test-validation set ['turk' or 'asset'] (default: 'turk')")
	parser.add_argument('--path', action='store', dest='path', type=str, default=None,
						help="path to checkpoint")

	res = parser.parse_args()

	trainConfigPath = f'data/{res.evalset}/train_config.pkl'
	if os.path.exists(trainConfigPath) and os.path.isfile(trainConfigPath):
		trainRes = readpickle(trainConfigPath)
	else:
		raise FileNotFoundError(f"trainConfig: No such mapping file found:'{trainConfigPath}'")

	validEvalPath = os.path.join(f'data/{res.evalset}','valid_eval_data')
	testEvalPath = os.path.join(f'data/{res.evalset}','test_eval_data')

	generateTestCmd = f"fairseq-generate {testEvalPath} \
--path {res.path} \
--batch-size 128 --beam 8 --remove-bpe > data/{res.evalset}/checkpoint_eval.test"

	generateValidCmd = f"fairseq-generate {validEvalPath} \
--path {res.path} \
--batch-size 128 --beam 8 --remove-bpe > data/{res.evalset}/checkpoint_eval.valid"

	os.system(generateTestCmd)
	os.system(generateValidCmd)

	# parse output into {src,pred,ref} samples
	testOutput = parseFairSeqOutput(f'data/{res.evalset}/checkpoint_eval.test',
					nbchars=trainRes.nbchars,
					levsim=trainRes.levsim,
					wrdrank=trainRes.wrdrank,
					deptree=trainRes.deptree)
	testOutput = {k:v for k,v in sorted(testOutput.items())}
	testOutput = [data['pred'] for data in testOutput.values()]
	print(len(testOutput))
	validOutput = parseFairSeqOutput(f'data/{res.evalset}/checkpoint_eval.valid',
					nbchars=trainRes.nbchars,
					levsim=trainRes.levsim,
					wrdrank=trainRes.wrdrank,
					deptree=trainRes.deptree)
	validOutput = {k:v for k,v in sorted(validOutput.items())}
	validOutput = [data['pred'] for data in validOutput.values()]
	print(len(validOutput))
	# map NER tags to original tokens
	if trainRes.ner:
		testOutput = NERuntag(testOutput,evalset=res.evalset,split='test')
		validOutput = NERuntag(validOutput,evalset=res.evalset,split='valid')

	validset, testset = loadHuggingfaceDataset(datasetPath=res.evalset,
							datasetDir='simplification',
							validationSplit='validation',
							testSplit='test',
							columns=['original','simplifications'])

	testScoringData = [{'src':inSample[0],'pred':outSample,'ref':inSample[1]} for inSample, outSample in zip(testset,testOutput)] 
	validScoringData = [{'src':inSample[0],'pred':outSample,'ref':inSample[1]} for inSample, outSample in zip(validset,validOutput)] 

	# score test output
	testSariScore = scoreOutput(testScoringData,res.evalset,'test')
	validSariScore = scoreOutput(validScoringData,res.evalset,'valid')

	# print(len(testScoringData),round(testSariScore,3))
	print(len(testScoringData),testSariScore)
	# print(len(validScoringData),round(validSariScore,3))
	print(len(validScoringData),validSariScore)

	writetxt(testOutput,f'data/{res.evalset}/test_out.txt',newline=True)
	writetxt(validOutput,f'data/{res.evalset}/valid_out.txt',newline=True)

