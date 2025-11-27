import os
import sys
import torch
import json
import logging
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

# Add current directory to path so we can import local modules
sys.path.append(os.getcwd())

from model_factory import create_model
from config import ShareConfig
from prepare_data import prepare_data

# Setup logging (stdout only initially)
# Force logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

logger.info(f"="*60)
logger.info(f"Available GPUs: {torch.cuda.device_count()}")


def compute_ppl(max_length, stride, data, model, device):
    """Compute perplexity using the sliding window approach from test.py"""
    model.to(device)
    model = model.eval()
    seq_len = data.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            output = model(input_ids, labels=target_ids)

            neg_log_likelihood = output.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl



class Args:
    def __init__(self, yaml_file):
        self.yaml_config_file = yaml_file
        self.calibration_size = 256
        self.dataset_name = "wikitext"
        self.dataset_cache_dir = None

# List of configs to evaluate
config_files = [
    "tasks/configs/wikitext_ppl/llama/share2/share_llama3_1_8b_30.yaml",
    "tasks/configs/wikitext_ppl/llama/share2/share_llama3_1_8b_40.yaml",
    "tasks/configs/wikitext_ppl/llama/share2/share_llama3_1_8b_50.yaml"
]

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

for yaml_file in config_files:
    logger.info(f"\n" + "="*80)
    logger.info(f"Processing config: {yaml_file}")
    logger.info("="*80)

    try:
        args = Args(yaml_file)
        config = ShareConfig(args)
        
        # Setup file logging for this config
        model_name_safe = config.model_name.replace("/", "_")
        log_filename = f"verify_eval_{model_name_safe}_{config.compression_ratio}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join("logs", log_filename)
        
        file_handler = logging.FileHandler(log_filepath, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_filepath}")
        
        logger.info(f"Model Name: {config.model_name}")
        logger.info(f"Model Type: {config.model_type}")
        logger.info(f"Untrained Model Path: {config.untrained_model_path}")
        logger.info(f"Compression Ratio: {config.compression_ratio}")

        # Check if model checkpoint exists to avoid accidental compression
        if not os.path.exists(config.untrained_model_path):
            logger.warning(f"Checkpoint not found at {config.untrained_model_path}. Skipping evaluation to avoid compression.")
            continue

        # Load the compressed model
        try:
            model = create_model(config)
            model = model.cuda().eval()
            logger.info(f"Model loaded successfully on {torch.cuda.device_count()} GPU(s).")
        except Exception as e:
            logger.error(f"Failed to load model for {yaml_file}: {e}")
            continue

        # Setup Tokenizer
        try:
            if config.model_type == "llama2":
                if "llama-3" in config.model_name.lower() or "llama3" in config.model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                else:
                    tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            continue

        # Prepare data for PPL evaluation
        print("\nPreparing data for PPL evaluation...")
        try:
            train_dataset, val_dataset, test_dataset, data_collator = prepare_data(
                config.dataset_name, 
                tokenizer,
                config.context_length,
                config.dataset_cache_dir
            )
            print("Data prepared successfully.")
        except Exception as e:
            print(f"Failed to prepare data: {e}")
            continue

        # Compute PPL
        print("\nComputing perplexity on test dataset...")
        try:
            ppl = compute_ppl(config.context_length, config.stride, test_dataset, model, "cuda")
            logger.info(f"\nPerplexity: {ppl}")
        except Exception as e:
            logger.error(f"Failed to compute perplexity: {e}")
            continue

        # Run lm_eval evaluations
        logger.info("\n" + "="*50)
        logger.info("Running lm_eval evaluations on multiple tasks...")
        logger.info("="*50)

        # Set cache directory for datasets to avoid encoding issues
        if config.dataset_cache_dir:
            os.environ["HF_DATASETS_CACHE"] = config.dataset_cache_dir

        tasks = [
            # "wikitext", 
            # "ptb", 
            "arc_challenge", 
            "arc_easy", 
            "hellaswag", 
            "piqa", 
            "winogrande",
            "openbookqa"
        ]

        # Evaluate tasks one by one to handle failures gracefully
        all_results = {}
        failed_tasks = []

        for task in tasks:
            logger.info(f"\nEvaluating {task}...")
            try:
                # Wrap the model for lm_eval
                # Use device_map='auto' for multi-GPU support via accelerate
                lm_eval_model = HFLM(
                    pretrained=model,
                    tokenizer=tokenizer,
                    batch_size="auto",
                    device_map="auto" if torch.cuda.device_count() > 1 else None
                )
                
                # Run evaluation on single task
                result = lm_eval.simple_evaluate(
                    model=lm_eval_model,
                    tasks=[task],
                    num_fewshot=0,
                    batch_size="auto",
                    device="cuda",
                    log_samples=False
                )
                
                all_results[task] = result["results"][task]
                logger.info(f"✓ {task} completed successfully")
                
            except Exception as e:
                logger.error(f"✗ {task} failed: {e}")
                logger.error(f"Full traceback for {task}:")
                import traceback
                logger.error(traceback.format_exc())
                failed_tasks.append(task)
                continue

        # Consolidate results
        results = {"results": all_results}

        try:
            
            # Print results
            logger.info("\n" + "="*50)
            logger.info(f"EVALUATION RESULTS FOR {yaml_file}")
            logger.info("="*50)
            
            for task_name in tasks:
                if task_name in results["results"]:
                    task_results = results["results"][task_name]
                    logger.info(f"\n{task_name.upper()}:")
                    for metric, value in task_results.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {metric}: {value:.4f}")
            
            # Print summary
            logger.info("\n" + "="*50)
            logger.info("SUMMARY")
            logger.info("="*50)
            
            successful_count = len(all_results)
            total_count = len(tasks)
            logger.info(f"Successfully evaluated: {successful_count}/{total_count} tasks\n")
            
            if "results" in results and len(results["results"]) > 0:
                for task_name in tasks:
                    if task_name in results["results"]:
                        task_results = results["results"][task_name]
                        # Try to find the main metric for each task
                        if "acc" in task_results:
                            logger.info(f"✓ {task_name}: {task_results['acc']:.4f}")
                        elif "acc_norm" in task_results:
                            logger.info(f"✓ {task_name}: {task_results['acc_norm']:.4f}")
                        elif "word_perplexity" in task_results:
                            logger.info(f"✓ {task_name}: {task_results['word_perplexity']:.4f}")
                        elif "byte_perplexity" in task_results:
                            logger.info(f"✓ {task_name}: {task_results['byte_perplexity']:.4f}")
            
            if failed_tasks:
                logger.info("\nFailed tasks:")
                for task_name in failed_tasks:
                    logger.info(f"✗ {task_name}")
            
            logger.info(f"\nEvaluation complete for {yaml_file}!")
            
        except Exception as e:
            logger.error(f"Failed to display results: {e}")
            import traceback
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Critical error processing {yaml_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Clean up to free memory
        if 'model' in locals():
            del model
        if 'lm_eval_model' in locals():
            del lm_eval_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Remove file handler
        if 'file_handler' in locals():
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

logger.info(f"All configurations processed.")


