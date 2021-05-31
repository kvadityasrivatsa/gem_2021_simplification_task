import os
import argparse

from src.scoring import parseFairSeqOutput
from src.util import writetxt

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--path', action='store', dest='path', type=str, default=None,
						help='path to model checkpoint')
	parser.add_argument('-t', action='store', dest='evalset', type=str, default=None,
						help='evalset')

	res = parser.parse_args()

	print('running valid cmd')

	generateValidCmd = f"fairseq-generate data/turk/valid_eval_data --path {res.path} \
--batch-size 128 --beam 5 --remove-bpe > data/turk/tempCheckpointOut.txt"

	os.system(generateValidCmd)

	print('parsing output')

	output = parseFairSeqOutput('data/turk/tempCheckpointOut.txt',
									nbchars=True,
									levsim=True,
									wrdrank=True,
									deptree=False)
	output = {k:v for k,v in sorted(output.items())}
	output = [data['pred'] for data in output.values()]

	print(output[:5])

	writetxt(output,'temp_out.txt',newline=True)
	scoreCmd = "easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' < temp_out.txt"
	os.system(scoreCmd)