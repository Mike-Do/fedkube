import h5py
from transformers import AutoTokenizer
import json
from tqdm import tqdm
model_name = "google/bert_uncased_L-4_H-256_A-4"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputf = "/Users/yong/Desktop/fedkube/code/data/reviews.h5"
outputf = "/Users/yong/Desktop/fedkube/code/data/reviews.jsonl"

wf = open(outputf, "w+")

with h5py.File(inputf, "r") as f:
    #### Print all root level object names (aka keys) 
    #### these can be group or dataset names 
    # print("Keys: %s" % f.keys())
    
    #### get first object name/key; may or may NOT be a group
    # a_group_key = list(f.keys())[0]

    for a_group_key in list(f.keys()):
	    # If a_group_key is a dataset name, 
	    # this gets the dataset values and returns as a list
	    
	    data = list(f[a_group_key])
	    # print(data[0]) ### see the structure

	    # information taken from dataset.py
	    date = 0
	    user = 1
	    asin = 2
	    category = 3
	    rating = 4

	    start_text_idx = 7
	    max_summary_len = 16

	    print(f"Processing {len(data)} Data instances in group ({a_group_key})")

	    for i in tqdm(range(len(data))):
	    	summary = data[i][start_text_idx:start_text_idx+max_summary_len]
	    	summary = tokenizer.decode(summary.tolist())
	    	review = data[i][start_text_idx+max_summary_len:]
	    	review = tokenizer.decode(review.tolist())

	    	wf.write(json.dumps(
	    		{
	    			"group_key": a_group_key,
	    			"data_idx": i,
	    			"summary": summary,
	    			"review": review,
	    			"category": str(data[i][category]),
	    			"rating": str(data[i][rating]),
	    			"date": str(data[i][date]),
	    			"user": str(data[i][user]),
	    		}
	    	))

	    	wf.write("\n")

