import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizerFast
from transformers.pipelines.pt_utils import KeyDataset
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
from data.yelp_multi_tags.prepare import NON_FOOD_RELATED_TAGS, FOOD_RELATED_TAGS, YelpMultiTagsCFGTokenizer, CUISINES as MULTI_TAGS_CUISINES
from utils.testing import get_hf_prompt_completion_dataset, get_hf_prompt_only_dataset_from_full_prompt_completion_dataset, prompt_prefix_easy_classifier, prompt_prefix_multi_tags
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import time
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
    # root = 'root   ::= FOOD_BLOCK NON_FOOD_BLOCK | FOOD_BLOCK | NON_FOOD_BLOCK | ""\n'
    # food_block = 'FOOD_BLOCK ::= "FOOD_TAGS{" INSIDE_FOOD_TAGS "}"\n'
    # inside_food_tags = 'INSIDE_FOOD_TAGS ::= CUISINE FOOD_RELATED_TAGS_RECURSIVE | FOOD_RELATED_TAGS FOOD_RELATED_TAGS_RECURSIVE\n'
    # cuisine = 'CUISINE ::= "' + '"|"'.join(CUISINES) + '"\n'
    # food_related_tags_recursive = 'FOOD_RELATED_TAGS_RECURSIVE ::= ' + 'FOOD_RELATED_TAGS FOOD_RELATED_TAGS_RECURSIVE | FOOD_RELATED_TAGS\n'
    # food_related_tags = 'FOOD_RELATED_TAGS ::= "' + '"|"'.join(FOOD_RELATED_TAGS) + '"\n'
    # non_food_block = 'NON_FOOD_BLOCK ::= "NON_FOOD_TAGS{" NON_FOOD_RELATED_TAGS NON_FOOD_RELATED_TAGS_RECURSIVE "}"\n'
    # non_food_related_tags_recursive = 'NON_FOOD_RELATED_TAGS_RECURSIVE ::= ' + 'NON_FOOD_RELATED_TAGS NON_FOOD_RELATED_TAGS_RECURSIVE | NON_FOOD_RELATED_TAGS\n'
    # non_food_related_tags = 'NON_FOOD_RELATED_TAGS ::= "' + '"|"'.join(NON_FOOD_RELATED_TAGS) + '"\n'

    # cfg_str = root + food_block + inside_food_tags + cuisine + food_related_tags_recursive + food_related_tags + non_food_block + non_food_related_tags_recursive + non_food_related_tags


    cfg_str = """
root ::= "Output:" (FOOD_BLOCK | NON_FOOD_BLOCK | FOOD_BLOCK NON_FOOD_BLOCK)
FOOD_BLOCK ::= "FOOD_TAGS{" (CUISINE_BLOCK | FOOD_TAG_LIST | CUISINE_BLOCK "," FOOD_TAG_LIST) "}"
CUISINE_BLOCK ::= "RESTAURANT(" CUISINE ")"

FOOD_TAG_LIST ::= (FOOD_TAG ("," FOOD_TAG)*)
FOOD_TAG ::= "Bars" | "Sandwiches" | "Pizza" | "Coffee & Tea" | "Fast Food" | "Burgers" | "Breakfast & Brunch" | "Specialty Food" | "Seafood" | "Desserts"  | "Bakeries" | "Salad" | "Chicken Wings"

NON_FOOD_BLOCK ::= "NON_FOOD_TAGS{" NON_FOOD_TAG_LIST "}"
NON_FOOD_TAG_LIST ::= NON_FOOD_TAG ("," NON_FOOD_TAG)*
"""

    cfg_str += 'CUISINE ::= "' + '"|"'.join(MULTI_TAGS_CUISINES) + '"\n'
    cfg_str += 'NON_FOOD_TAG ::= "' + '"|"'.join(NON_FOOD_RELATED_TAGS) + '"\n'

    print(f"grammar_str: {cfg_str}")

    return cfg_str

def get_llama_33_model():
    # see: https://huggingface.co/blog/4bit-transformers-bitsandbytes
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
    input_text = "What are we having for dinner?"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = quantized_model.generate(**input_ids, max_new_tokens=10)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

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
    

def run_experiment(task_type, model_id, should_finetune, num_train_samples, batch_size, val_size, is_big_model=False, start_from=0, use_llama=False, is_for_smollm2=False):
    if task_type == "easy":
        grammar_str = easy_example_cfg_str()
        file_path = "data/yahooreviewcuisine/with_generations_with_llama.json"
        max_new_tokens = 12
        eval_metric = "accuracy"
        prompt_prefix = prompt_prefix_easy_classifier
    elif task_type == "multi_tags":
        grammar_str = multi_tags_example_cfg_str()
        file_path = "data/yelp_multi_tags_2/llama_3_3_generations_2_fixed.json"
        max_new_tokens = 120
        eval_metric = "f1"
        prompt_prefix = prompt_prefix_multi_tags
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    
    
    
    # first, finetune the model
    if not is_big_model:
    # Load model and tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            # attn_implementation="flash_attention_2"
            )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    else:
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # try:
        #     tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
        # except Exception as e:
        #     print(f"Error tokenizing: {e}")
        #     tokenizer = LlamaTokenizerFast()
        tokenizer.pad_token = tokenizer.eos_token
        # model.generation_config.pad_token_id = model.generation_config.eos_token_id
        # input_text = "What are we having for dinner?"
        # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        # output = quantized_model.generate(**input_ids, max_new_tokens=10)
        # print(tokenizer.decode(output[0], skip_special_tokens=True))

    # Load dataset
    dataset = get_hf_prompt_completion_dataset(max_train_samples=num_train_samples, path_to_json_file=file_path, task = task_type, use_llama=use_llama, is_for_smollm2=True)
    time_start = time.time()
    if should_finetune:

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
            num_train_epochs = 3,
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
    time_end = time.time()
    finetune_time = time_end - time_start

    # Run experiment
        # load in file name
    with open(file_path, "r") as f:
        data = json.load(f)

    original_data_size = len(data)
    if not is_big_model:
        data = data[num_train_samples:num_train_samples+val_size]
    else:
        data = data[start_from:start_from+val_size]
    input_list = [item["input"] for item in data]
    instructions = [f"{prompt_prefix}{item}" for item in input_list]
    expected_generations = [item["generated"] for item in data]
    expected_ground_truths = [item["expected"] for item in data]

    

    print(f"Number of samples: {len(instructions)}")

    # predicted_answers = []
    
    pipe = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    # device=device,
                    device_map="auto",
                    max_new_tokens=max_new_tokens,
                    return_full_text=False,
                    batch_size=batch_size)
    
    # get prompt only dataset
    prompt_only_dataset = get_hf_prompt_only_dataset_from_full_prompt_completion_dataset(dataset)
    generation_time = 0

    j = 0
    predicted_answers = []

    for i in range(0, len(instructions), batch_size):
        print(f"batch {i // batch_size + 1} of {(len(instructions) // batch_size) + 1}")
        time_start = time.time()
        grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        batch_instructions = instructions[i:i+batch_size]
        generations = pipe(
            batch_instructions,
            do_sample=True,
            logits_processor=[grammar_processor],
            # repetition_penalty=2.0,
        )
        print(f"Time for this generation in this batch: {time.time() - time_start}")

        for generation_group in generations:
            for generation in generation_group:
                # strip the word "Output:" from the beginning of the generation
                generation['generated_text'] = generation['generated_text'].split("Output:")[1].strip()
                if is_big_model:
                    data[j]["generated_llama"] = generation['generated_text']
                    j += 1
                predicted_answers.append(generation['generated_text'])
        time_end = time.time()
        generation_time += time_end - time_start
        print(f"Final Time for this generation in this batch: {time_end - time_start}")
        print()



        # for every 20 save everything to the JSON
        if i % 20 == 0 and is_big_model:
            # zipped_up = zip(instructions, input_list, predicted_answers, expected_generations, expected_ground_truths)
            # objectified_results = [{"instruction": instruction, "input": input, "predicted": predicted, "generated": generation, "ground_truth": ground_truth} for instruction, input, predicted, generation, ground_truth in zipped_up]
            save_file_path = f"data/yelp_multi_tags_2/llama_3_3_generations_{start_from}_{i}.json" if task_type == "multi_tags" else f"data/yahooreviewcuisine/llama_3_3_generations_{start_from}_{i}.json"
            with open(save_file_path, "w") as f:
                json.dump(data, f)
            print(f"Batch {i // batch_size + 1} saved to {save_file_path}")
    # grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    # grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    # generations = pipe(
    #     KeyDataset(prompt_only_dataset, "text"),
    #     do_sample=False,
    #     repetition_penalty=2.0,
    #     logits_processor=[grammar_processor],
    # )
    print(f"Generation time: {generation_time}")

    print(f"number of generations: {len(predicted_answers)}")

    for predicted_answer, expected_generation, expected_ground_truth, instruction in zip(predicted_answers, expected_generations, expected_ground_truths, instructions):
        print(f"instruction: {instruction}")
        print("--------------------------------")
        print(f"predicted_answer: {predicted_answer}")
        print(f"expected_generation: {expected_generation}")
        print(f"expected_ground_truth: {expected_ground_truth}")
        print("########################")
        print()


    correct_ground_truth = 0
    correct_generation = 0
    for predicted_generation, expected_generation, expected_ground_truth in zip(predicted_answers, expected_generations, expected_ground_truths):
        # print(f"generation: {predicted_generation}, expected_generation: {expected_generation}, expected_ground_truth: {expected_ground_truth}")
        if predicted_generation == expected_generation:
            correct_generation += 1
        if predicted_generation == expected_ground_truth:
            correct_ground_truth += 1
        
    accuracy_generation = correct_generation / len(predicted_answers)
    accuracy_ground_truth = correct_ground_truth / len(predicted_answers)
    results = {}
    if eval_metric == "f1":
        ground_truth_true_positives = 0
        ground_truth_false_positives = 0
        ground_truth_false_negatives = 0
        generation_true_positives = 0
        generation_false_positives = 0
        generation_false_negatives = 0

        for prediction, generation, ground_truth in zip(predicted_answers, expected_generations, expected_ground_truths):

            try:
                tokenized_prediction = YelpMultiTagsCFGTokenizer.tokenize(prediction)
            except Exception as e:
                print(f"Error tokenizing prediction: {e}")
                print(f"Prediction: {prediction}")
                print("--------------------------------")
                continue

            tokenized_ground_truth = YelpMultiTagsCFGTokenizer.tokenize(ground_truth)
            tokenized_generation = YelpMultiTagsCFGTokenizer.tokenize(generation)
            curr_metrics = YelpMultiTagsCFGTokenizer.get_f1_sub_metrics(tokenized_ground_truth, tokenized_prediction)
            ground_truth_true_positives += curr_metrics[0]
            ground_truth_false_positives += curr_metrics[1]
            ground_truth_false_negatives += curr_metrics[2]
            curr_metrics = YelpMultiTagsCFGTokenizer.get_f1_sub_metrics(tokenized_generation, tokenized_ground_truth)
            generation_true_positives += curr_metrics[0]
            generation_false_positives += curr_metrics[1]
            generation_false_negatives += curr_metrics[2]

        f1_score_ground_truth, f1_precision_ground_truth, f1_recall_ground_truth = YelpMultiTagsCFGTokenizer.get_f1_score(ground_truth_true_positives, ground_truth_false_positives, ground_truth_false_negatives)
        f1_score_generation, f1_precision_generation, f1_recall_generation = YelpMultiTagsCFGTokenizer.get_f1_score(generation_true_positives, generation_false_positives, generation_false_negatives)

        results["f1_score_ground_truth"] = f1_score_ground_truth
        results["f1_precision_ground_truth"] = f1_precision_ground_truth
        results["f1_recall_ground_truth"] = f1_recall_ground_truth
        results["f1_score_generation"] = f1_score_generation
        results["f1_precision_generation"] = f1_precision_generation
        results["f1_recall_generation"] = f1_recall_generation
        
    predicted_total_time_taken_generations = generation_time * ((original_data_size - num_train_samples) / val_size)
    print(f"Predicted total time taken generations: {predicted_total_time_taken_generations}")

    if is_big_model:
        save_file_path = f"data/yelp_multi_tags_2/llama_3_3_generations_{start_from}_{start_from+val_size}.json" if task_type == "multi_tags" else f"data/yahooreviewcuisine/llama_3_3_generations_{start_from}_{start_from+val_size}.json"
        with open(save_file_path, "w") as f:
            json.dump(data, f)

    results = {
        **results,
        "accuracy_generation": accuracy_generation,
        "accuracy_ground_truth": accuracy_ground_truth,
        "generation_time": generation_time,
        "finetune_time": finetune_time,
        "val_size": val_size,
        "predicted_total_time_taken_generations": predicted_total_time_taken_generations,
        "model_id": model_id,
        "task_type": task_type,
        "should_finetune": should_finetune,
        "num_train_samples": num_train_samples,
        "batch_size": batch_size,
    }

    # save results to file
    results_file_path = f"data/yelp_multi_tags_2/llama_3_3_generations_{start_from}_{start_from+val_size}_results.json" if task_type == "multi_tags" else f"data/yahooreviewcuisine/llama_3_3_generations_{start_from}_{start_from+val_size}_results.json"
    with open(results_file_path, "w") as f:
        json.dump(results, f)

    pprint(results)
    return results


def wrangle_json():
    with open("data/yahooreviewcuisine/llama_3_3_generations.json", "r") as f:
        data_llama = json.load(f)

    with open("data/yahooreviewcuisine/with_generations_2.json", "r") as f:
        data_gpt = json.load(f)

    print(f"Length of data_llama: {len(data_llama)}")   
    print(f"Length of data_gpt: {len(data_gpt)}")
    
    # check length
    total_correct_ground_truth = 0
    total_correct_gpt = 0

    # change the predicted to 'generation' instead
    for i in range(len(data_llama)):
        # check that the expected is the same for both files
        if data_llama[i]["input"] != data_gpt[i]["input"]:
            print(f"Expected is not the same for {i}")


        if (data_llama[i]["generation"]) == data_gpt[i]["expected"]:
            total_correct_ground_truth += 1
        if (data_llama[i]["generation"]) == data_gpt[i]["generated"]:
            total_correct_gpt += 1
        
    count = len(data_llama)

    print(f"Count: {count}")
    print(f"Accuracy: {total_correct_ground_truth / count}")
    print(f"Accuracy: {total_correct_gpt / count}")

    # for item in data:
    #     print(item["predicted"])
    #     print(item["ground_truth"])
    #     print("--------------------------------")


    

        

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

    # results = run_experiment(task_type="easy", model_id=model_id, should_finetune=True, num_train_samples=3072, batch_size=160, val_size=1024)
    # pprint(results)

    # results = run_experiment(task_type="multi_tags", model_id=model_id, should_finetune=False, num_train_samples=1024, batch_size=140, val_size=420)
    # pprint(results)

    # results = run_experiment(task_type="multi_tags", model_id=model_id, should_finetune=True, num_train_samples=1024, batch_size=140, val_size=700)
    # pprint(results)

    # results = run_experiment(task_type="easy", model_id=model_id, should_finetune=False, num_train_samples=1024, batch_size=4, val_size=4096, is_big_model=True)
    # print(f"Results from easy task")
    # pprint(results)

    # save results to file
    # with open("experiments/llama_3_3_easy_task_results.json", "w") as f:
    #     json.dump(results, f)

    results = run_experiment(task_type="multi_tags", model_id=model_id, should_finetune=True, num_train_samples=1024, batch_size=140, val_size=1400, start_from=0, is_big_model=False, is_for_smollm2=True, use_llama=True)
    print(f"Results from multi tags task")
    # pprint(results)

    # save results to file
    # with open("experiments/llama_3_3_multi_tags_task_results.json", "w") as f:
        # json.dump(results, f)

    # get_llama_33_model()


    # wrangle_json() 

    

    # testing()

    # get_chatgpt_completions()

# The animal is a cat.


