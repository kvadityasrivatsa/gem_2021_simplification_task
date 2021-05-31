
# GEM-Benchmark 2021 Shared Task Submission
Code release for submission made to [GEM-Benchmark](https://gem-benchmark.com/) 2021 Text Simplification Shared-Task on [TurkCorpus](https://huggingface.co/datasets/turk) ans [ASSET](https://huggingface.co/datasets/asset) datasets

## Getting Started

### Dependecies
- Python >= 3.7

### Installation
``` 
git clone https://github.com/kvadityasrivatsa/gem_2021_simplification_task.git
cd gem_2021_simplification_task
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
Generate and evaluate output (on SARI score)
- for TurkCorpus:
```
python3 evaluate.py --evalset turk
```
- for ASSET:
```
python3 evaluate.py --evalset asset
```

## Pretrained Model
The checkpoint for our model with the best scores is available [here](https://github.com/kvadityasrivatsa/gem_2021_simplification_task)

## Model Desciption
(Note: The official system-desciption for the model can be found [here](https://github.com/kvadityasrivatsa/gem_2021_simplification_task))

Our model builds upon the ACCESS model proposed in [_Controllable Sentence Simplification_](https://arxiv.org/abs/1910.02677) (Martin et al., 2020). 
 
## Authors

 - **KV Aditya Srivatsa** (k.v.aditya@research.iiit.ac.in)
 - **Monil Gokani** (monil.gokani@research.iiit.ac.in)
 
 If you have any queries, please do reach out. 

## License
Refer to the [LICENSE](https://github.com/kvadityasrivatsa/gem_2021_simplification_task/blob/main/LICENSE) file for more details.
