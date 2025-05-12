import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from data.yahooreviewcuisine.prepare import CUISINES
import json
from time import sleep
from transformers import pipeline
from pprint import pprint
from pydantic import BaseModel, field_validator, ValidationInfo
from typing import Literal, Optional
from enum import Enum
from openai import OpenAI
from data.yelp_multi_tags.prepare import NON_FOOD_RELATED_TAGS, FOOD_RELATED_TAGS
from utils.testing import get_hf_prompt_completion_dataset, prompt_prefix_easy_classifier, prompt_prefix_multi_tags
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

class CuisineEnum(str, Enum):
    pass

for cuisine in CUISINES:
    setattr(CuisineEnum, cuisine, cuisine)

def easy_example_cfg_str():
    cuisine_list = str.join("\" | \"", CUISINES)
    cuisine_list = f"\"{cuisine_list}\""

    cfg_str = 'root   ::= "{restaurant(" cuisine ")}" | "{not_restaurant}"\n'
    cfg_str += f'cuisine ::= {cuisine_list}'

    print(f"grammar_str: {cfg_str}")

    return cfg_str

def multi_tags_example_cfg_str():
    # TODO:
    cfg_str = 'root   ::= NON_FOOD_TAGS | FOOD_TAGS\n'
    cfg_str += 'NON_FOOD_TAGS ::= \n' + '{' + ' CUISINE | ' + ' | '.join(NON_FOOD_RELATED_TAGS) + '}\n'
    cfg_str += f'FOOD_TAGS ::= {FOOD_RELATED_TAGS}\n'

    print(f"grammar_str: {cfg_str}")

    return cfg_str



def run_experiment_legacy(grammar_str, model_id, device, file_path, prompt_prefix, eval_metric="accuracy", eval_size=256, max_new_tokens=12, batch_size=64):

    # load in file name
    with open(file_path, "r") as f:
        data = json.load(f)

    data = data[:eval_size]
    
    input_list = [item["input"] for item in data]
    # convert input to instructions
    instructions = [f"{prompt_prefix} {item}" for item in input_list]
    
    expected = [item["output"] for item in data]

    print(f"Number of samples: {len(instructions)}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Create grammar constraint and logits processor


    predicted_answers = []

    for i in range(0, len(instructions), batch_size):
        grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        pipe = pipeline("text-generation", 
                        model=model, 
                        tokenizer=tokenizer,
                        device=device,
                        max_new_tokens=max_new_tokens,
                        return_full_text=False,
                        batch_size=batch_size)
        batch_instructions = instructions[i:i+batch_size]
        print(f"length of batch_instructions: {len(batch_instructions)}")
        generations = pipe(
            batch_instructions,
            do_sample=False,
            logits_processor=[grammar_processor],
            repetition_penalty=2.0,
        )

        for generation_group in generations:
            for generation in generation_group:
                predicted_answers.append(generation['generated_text'])

    print(f"number of generations: {len(predicted_answers)}")

    if eval_metric == "accuracy":
        correct = 0
        for generation, output in zip(predicted_answers, expected):
            print(f"generation: {generation}, output: {output}")
            if generation == output:
                correct += 1
            
        accuracy = correct / len(predicted_answers)
        return accuracy

def run_experiment(task_type, model_id, should_finetune, num_train_samples, batch_size):
    if task_type == "easy":
        grammar_str = easy_example_cfg_str()
        file_path = "data/yahooreviewcuisine/with_generations_2.json"
        max_new_tokens = 12
        eval_metric = "accuracy"
        prompt_prefix = prompt_prefix_easy_classifier
    elif task_type == "multi_tags":
        grammar_str = multi_tags_example_cfg_str()
        file_path = "data/yelp_multi_tags_2/with_generations_2.json"
        max_new_tokens = 90
        eval_metric = "accuracy"
        prompt_prefix = prompt_prefix_multi_tags
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # first, finetune the model
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        # attn_implementation="flash_attention_2"
        )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Load dataset
    if should_finetune:
        dataset = get_hf_prompt_completion_dataset(max_train_samples=num_train_samples, path_to_json_file=file_path, task = task_type)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"],
            task_type="CAUSAL_LM",
        )

        training_args = SFTConfig(
            output_dir="/tmp",
            num_train_epochs = 1,
            per_device_train_batch_size = 8,
            completion_only_loss=True
            )

        trainer = SFTTrainer(
            model,
            train_dataset=dataset,
            args=training_args,
            peft_config=peft_config,
        )

        trainer.train()

    # Run experiment
        # load in file name
    with open(file_path, "r") as f:
        data = json.load(f)

    data = data[num_train_samples:]
    input_list = [item["input"] for item in data]
    instructions = [f"{prompt_prefix} {item}" for item in input_list]
    expected_generations = [item["generated"] for item in data]
    expected_ground_truth = [item["expected"] for item in data]

    print(f"Number of samples: {len(instructions)}")

    predicted_answers = []
    
    pipe = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    return_full_text=False,
                    batch_size=batch_size)
    for i in range(0, len(instructions), batch_size):
        grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        batch_instructions = instructions[i:i+batch_size]
        generations = pipe(
            batch_instructions,
            do_sample=False,
            logits_processor=[grammar_processor],
            repetition_penalty=2.0,
        )

        for generation_group in generations:
            for generation in generation_group:
                predicted_answers.append(generation['generated_text'])


    print(f"number of generations: {len(predicted_answers)}")

    if eval_metric == "accuracy":
        correct_ground_truth = 0
        correct_generation = 0
        for predicted_generation, expected_generation, expected_ground_truth in zip(predicted_answers, expected_generations, expected_ground_truth):
            # print(f"generation: {predicted_generation}, expected_generation: {expected_generation}, expected_ground_truth: {expected_ground_truth}")
            if predicted_generation == expected_generation:
                correct_generation += 1
            if predicted_generation == expected_ground_truth:
                correct_ground_truth += 1
            
        accuracy_generation = correct_generation / len(predicted_answers)
        accuracy_ground_truth = correct_ground_truth / len(predicted_answers)
        return accuracy_generation, accuracy_ground_truth




    

        

# def testing():
#     # Model identifier
#     model_id = "facebook/opt-125m"

#     # Load model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     tokenizer.pad_token = tokenizer.eos_token
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

#     # Define grammar string
#     grammar_str = """
#     root   ::= "The animal is a " animal "."
#     animal ::= "cat" | "fish"
#     """

#     # Create grammar constraint and logits processor
#     grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
#     grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

#     # Initialize text generation pipeline
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         # device_map="auto",
#         max_new_tokens=100,
#         batch_size=3,
#         return_full_text=False,
#         device=device,
#     )

#     # Define prompts
#     prompts = [
#         'The text says, "The animal is a dog." The answer is obvious. ',
#         'I\'m going to say "The animal is a dog." I\'m so excited for this! Here I go! ',
#         'I want to eat seafood. '
#     ]

#     print(f"number of prompts: {len(prompts)}")

#     # sleep(20)

#     # Generate constrained text using the pipeline.
#     generations = pipe(
#         prompts,
#         do_sample=False,
#         logits_processor=[grammar_processor],
#     )

#     print(f"number of generations: {len(generations)}")

#     # Print generated texts
#     for generation_group in generations:
#         for generation in generation_group:
#             print(generation['generated_text'])
            

if __name__ == "__main__":
    # model_id = "huggyllama/llama-7b"
    # model_id = "facebook/opt-125m"
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # file_path = "data/yahooreviewcuisine/all_reviews.json"
    # prompt_prefix = """Classify the following business reviews as a restaurant with its cuisine or not a restaurant. Example:
    # Input: The pasta was great!
    # Output: {Restaurant(italian)}

    # Input: I was filling up gas at the gas station.
    # Output: {Not Restaurant}

    # Input: I liked the burrito.
    # Output: {Restaurant(mexican)}
    # """

    # grammar_str = easy_example_cfg_str()

    # accuracy = run_experiment_legacy(grammar_str, model_id, device, file_path, prompt_prefix, eval_metric="accuracy", eval_size=256, batch_size=64)
    # print(f"Accuracy: {accuracy}")

    accuracy_generation, accuracy_ground_truth = run_experiment(task_type="easy", model_id=model_id, should_finetune=False, num_train_samples=1024, batch_size=192)

    # testing()

    # get_chatgpt_completions()

# The animal is a cat.