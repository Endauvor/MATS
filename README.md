First experiments on fine-tuning. 

Description of files, folders and programs. 

1. data/dataset1 is a set of questions-answers such that question contains simple talk.
2. data/dataset2 is a set of questions-answers such that question does not contain simple talk. That is a simple talk is the part of an answer.

Both datasets have features 'question', 'answer', 'percent'. 'percent' is the part of final thougths that the model should show. Here is a decompisition of datasets (1 and 2):

percent of data | percent of intellectual talk
10 % | 90 % \n
10 % | 70 % 
10 % | 50 %
10 % | 40 %
10 % | 30 %
10 % | 20 % 
10 % | 10 %
30 % | only an answer (~ 1%). (The prompt here changes a little from 'give n% of final thoughts' to 'give the final answer'.) 

Programs: 

1. train.ipynb -- here is a code for fine-tuning. I apply a mask (in the function 'formatting_prompt') for masking a prompt ('question') part of data.
2. 

**Here is out plan for today: **
1. Make dataloader that mixes 2 datasets.
2. Make dataloader such that on the first step of training the model learn 90% of intellectual answer (that is we need here only part of data with percent = '0.9'), on the second step of training the model learn 70% of intellectual answer, etc. 


