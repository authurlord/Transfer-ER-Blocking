# Transfer-ER-Blocking 

Here is the code for reproducibility of our paper. The paper is available at the root folder [UDAEB](./UDAEB_bigdata.pdf).

## Requirement

check `requirements.txt` for the required python package, or apply `pip install -r requirements.txt`.

Additionally, you may check [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) and [vllm](http://github.com/vllm-project/vllm) for additional results.

## Model Download

Our base model for enriching attributes is [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

Our main embedding model is [bge-large-en-1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).

For the baseline model STransformer, you need to additionally download [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

For reproducing [Sudowoodo](https://github.com/megagonlabs/sudowoodo), you may also need to download [DistilBert](https://huggingface.co/distilbert/distilbert-base-uncased).

## Attribute Enrichment

Here we take WA-AB(Walmart-Amazon to Abt-Buy) as an example. This part is limited to 
`['WA','AB','AG','DS','DA']`

1. Apply `python Enrichment_LLM_input.py --target_data dataset-name` to generate query csv for LLM. For example, for the case above, apply
```
python Enrichment_LLM_input.py --target_data WA
python Enrichment_LLM_input.py --target_data AB
```

To illustrate, we provide our query file in
```
enrich_data/amazon_google_query.csv
enrich_data/ant_buy_query.csv
enrich_data/walmart_amazon_query.csv
```
For AG/AB/WA seperately.

2. Inference local LLM with `vllm` for offline batch query. The command is like:
```
python vllm_enrich.py \
--llm_path LLM_path \
--data_name dataset_name \
--gpu_num 4
```
For example, if you apply `Mistral-7B` model in `/home/user/model/Mistral-7B` folder, the command should be like:
```
python vllm_enrich.py --llm_path /home/user/model/Mistral-7B --data_name AB --gpu_num 4
python vllm_enrich.py --llm_path /home/user/model/Mistral-7B --data_name WA --gpu_num 4
```
If you are trying different model, be sure to update your `vllm` and `transformers` package up-to-date. We apply `tokenizer.apply_chat_template()` method in `transformers` library to automatically apply chat template.

To illustrate, we provide our output file in
```
enrich_data/amazon_google_output_mistral.npy
enrich_data/ant_buy_output_mistral.npy
enrich_data/walmart_amazon_output_mistral.npy
```
For AG/AB/WA seperately. Please unzip `enrich_data/LLM_Output.zip` into `enrich_data` folder for the above 3 files.

3. Parse the output of LLM, and merge them with originally `tableA/tableB` into `dict_ltable/dict_rtable`. Currently, the model_name contains 
`['mistral-7B','llama3-8B','qwen2-1.5B']`.
For example:
```
python Enrichment_Parse.py --data_name AB --model_name mistral-7B 
python Enrichment_Parse.py --data_name WA --model_name mistral-7B 
```

To illustrate, all of the enriched dict is provided in `enrich_data/enrich_dict` folder.
## Blocking Method
To run our method, for the `RI-AB` dataset, please run the following command:
```
python Blocking.py     \
--source_data RI     \
--target_data AB     \
--backbone_model_name /home/user/model/Mistral-7B \
--sbert_model_path /home/user/model/bge-large-en-1.5 \
```

The result is stored in `evaluation/result` folder.

## Baseline Method
