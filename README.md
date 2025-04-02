# Hyper repo about MATS application, again, agaain!!!

## Install

1. `make run`
2. `make jupyter`
3. select notebook 
4. enjoy

* `make create_env`, then fill clearml and hf credentials if you want to train something


## Plan
1. Fine-tuning
- [x] First experiments on fine-tuning.
- [x] Make dataloader that mixes 2 datasets. 
- [x] Make dataloader such that on the first step of training, we use only part of data with percent = '0.9', on the second step of training we use only part of data with percent = '0.7', etc. 
- [x] Fine-tune and enjoy. (Oh, yeah)

2. get vanilla and fine-tuned models 
- [x] lazy comparison with each other 
3. compare residual outputs from each model for generating simple talk 
- [x] (almost done, the problem with memory, need help) in the process of generating, catch residual outputs 
from each layer from vanilla model and fine-tuned one
- [ ] Compare them for each token generation, ensuring control over each token, 
and replace them to maintain consistency and continuity in generation.

## Notebooks (find me in notebooks folder)
1. train.ipynb -- here is a code for fine-tuning. I apply a mask (in the function 'formatting_prompt') for masking a prompt ('question') part of data.
2. createdataset.ipynb -- programs that creates dataset from GMSK8 (I can be mistaken) and simpletalks (data/sQA_data).  
Then we amalgamate this templates in masterpiece instructed qustions and answers to turn ulterior motives on in the model.
3. simpletalks.ipynb -- generation of simple talks with vllm and the model.
4. train_unsloth.ipynb -- here is a code for fine-tuning in 4bit.
5. lazy_inference.ipynb -- here you can find how to use the models
6. create_test.ipynb -- here you can find how to create tests datasets for different 'percent' value and with usefull information for step 3.
7. accuracy.ipynb -- here you can find a) the code for accuracy calculating (in terms of probabilities) b) the code for residual differences calculating. For vanilla we need to take trained model with low step. 


## Description of files, folders and programs. 

1. data/dataset1/train is a train set of questions-answers such that question contains simple talk. 
2. data/dataset2/train is a train set of questions-answers such that question does not contain simple talk. That is a simple talk is the part of an answer.

You can uploud them as Dataset.load_from_disk("...") where Dataset from datasets. 

Both datasets have features 'question', 'answer', 'percent'. 'percent' is the part of final thougths that the model should show. Here is a decompisition of datasets (1 and 2): ([part of the data], [part of the intellectual answer in percent]) =
([10 %, 10 %, 10 %, 10 %, 10 %, 10 %, 10 %, 30 %], [90 %, 70 %, 50 %, 40 %, 30 %, 20 %, 10 %, ~ 1%]). (For the case with ~1% the prompt changes a little from 'give n% of final thoughts' to 'give the final answer'.) 




