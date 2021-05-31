
# GEM-Benchmark 2021 Shared Task Submission
Code release for submission made to [GEM-Benchmark](https://gem-benchmark.com/) 2021 Text Simplification Shared-Task on [TurkCorpus](https://huggingface.co/datasets/turk) ans [ASSET](https://huggingface.co/datasets/asset) datasets

## Getting Started

------------

### Dependecies
- Python >= 3.7

### Installation
``` 
git clone https://github.com/kvadityasrivatsa/gem_2021_simplification_task.git
cd folder_name
pip install --no-deps -r requirements.txt
```
### How to use
Train the submission model on WikiLarge 
- for TurkCorpus:
```
python3 train.py --evalset turk --ner --nbchars 0.95 --levsim 0.75 --wrdrank 0.75
```
   
- for ASSET:
```
python3 train.py --evalset asset --ner --nbchars 0.95 --levsim 0.75 --wrdrank 0.75
```
Generate and evaluate output
- for TurkCorpus:
```
python3 evaluate.py --evalset turk
```
- for ASSET:
```
python3 evaluate.py --evalset asset
```
