from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from time import sleep
import argparse
parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")

parser.add_argument("--llm_path", type=str, default='Mistral-7B-Instruct-v0.2')

parser.add_argument("--data_name", type=str)

parser.add_argument('--gpu_num',type=int,default=1)

args = parser.parse_args()

llm_path = args.llm_path
data_name = args.data_name
gpu_num = args.gpu_num
llm = LLM(model=llm_path,tensor_parallel_size=gpu_num)  # Create an LLM.
tokenizer = AutoTokenizer.from_pretrained(llm_path)

from tqdm import tqdm
import pandas as pd

text_list = pd.read_csv('enrich_data/enrich_query/%s_input.csv' % data_name)['text_mistral'].to_list()


prompts = [str(a) for a in text_list] ## Mistral
text_all = []
for prompt in tqdm(prompts):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    text_all.append(text)



sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=1024)

outputs = llm.generate(text_all, sampling_params)
generation_list = []
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    generation_list.append(generated_text)
import numpy as np

np.save('enrich_data/enrich_query/%s_output.npy' % data_name,np.array(generation_list))

