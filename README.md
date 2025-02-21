First experiments on fine-tuning. 

Description of files, folders and programs. 

1. data/dataset1/train is a train set of questions-answers such that question contains simple talk. 
2. data/dataset2/train is a train set of questions-answers such that question does not contain simple talk. That is a simple talk is the part of an answer.

You can uploud them as Dataset.load_from_disk("...") where Dataset from datasets. 

Both datasets have features 'question', 'answer', 'percent'. 'percent' is the part of final thougths that the model should show. Here is a decompisition of datasets (1 and 2): ([part of the data], [part of the intellectual answer in percent]) =
([10 %, 10 %, 10 %, 10 %, 10 %, 10 %, 10 %, 30 %], [90 %, 70 %, 50 %, 40 %, 30 %, 20 %, 10 %, ~ 1%]). (For the case with ~1% the prompt changes a little from 'give n% of final thoughts' to 'give the final answer'.) 

Programs: 

1. train.ipynb -- here is a code for fine-tuning. I apply a mask (in the function 'formatting_prompt') for masking a prompt ('question') part of data.
2. 

**Here is out plan for today: **
1. Make dataloader that mixes 2 datasets.
2. Make dataloader such that on the first step of training the model learn 90% of intellectual answer (that is we need here only part of data with percent = '0.9'), on the second step of training the model learn 70% of intellectual answer, etc. 


