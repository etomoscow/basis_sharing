import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from prepare_data import prepare_data
from model_factory import create_model
from config import ShareConfig, add_args

if __name__ == '__main__':
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print("Use update: {}".format(config.update))
    print(config.untrained_model_path)

    tasks = ["openbookqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag", "piqa", "mathqa"]

    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print(config.compression_ratio)
    if config.model_type == "llama2":
        # Use AutoTokenizer for LLaMA 3.1, LlamaTokenizer for LLaMA 2
        if "llama-3" in config.model_name.lower() or "llama3" in config.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Set padding token (use eos_token for LLaMA models as they don't have a dedicated pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # model = create_model(config)
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = model.cuda()

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=128, max_batch_size=256)
    res = lm_eval.simple_evaluate(hflm, tasks=tasks, num_fewshot=0, batch_size=128, max_batch_size=256,
                                  device=model.device)
    print(res["results"])
