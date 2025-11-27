# Basis Sharing
This is the code for the paper "BASIS SHARING: CROSS-LAYER PARAMETER SHARING
FOR LARGE LANGUAGE MODEL COMPRESSION". Some config examples are added in config directory.

## Run Basis Sharing
To run Basis Sharing on LLaMA-7B for generation tasks, run
```
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
```
make sure to set *build_calib* as true for a model, when you want to compress it for the first time.

After compress with WikiText, to test with other dataset run
~~~
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml --dataset_name <ptb, C4, WikiText>
~~~
For C4 you need to download them from [link](https://drive.google.com/drive/folders/123Id1MkZVsKySGy_sMO4RgiJKrtPcvUp?usp=drive_link). Don't forget to update *dataset_cache_dir* in config file.

## Run LoRA
~~~
python lora.py  --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## Run Reasoning tasks
~~~
python test_adapter.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## Run Throughput tasks
~~~
python test_throughput.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## LLaMA 3.1 Support
We have added support for LLaMA 3.1 8B Instruct model.

### Configurations
New configuration files are available for 30%, 40%, and 50% compression ratios:
- `tasks/configs/wikitext_ppl/llama/share2/share_llama3_1_8b_30.yaml`
- `tasks/configs/wikitext_ppl/llama/share2/share_llama3_1_8b_40.yaml`
- `tasks/configs/wikitext_ppl/llama/share2/share_llama3_1_8b_50.yaml`

### Compression and Evaluation Pipeline
Use `verify_eval.py` to run the full pipeline (Compress -> Evaluate -> Save Results):
```bash
python verify_eval.py
```
This script will:
1. Check if compressed checkpoints exist (in `untrained_model/`).
2. If not, compress the model using the specified configs.
3. Evaluate the model on PPL (Wikitext) and zero-shot tasks (ARC, HellaSwag, PIQA, WinoGrande, OpenBookQA).
4. Save detailed logs to `logs/`.

### Interactive Evaluation
For interactive evaluation and PPL testing on PTB, use the Jupyter notebook:
- `eval_compressed_model.ipynb`

## Reference 

@misc{{parametersharing2024,

title={Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression},

author={Jingcun Wang and Yu-Guang Chen and Ing-Chao Lin and Bing Li and Grace Li Zhang},

archivePrefix={arXiv},

year={2024} 
}
