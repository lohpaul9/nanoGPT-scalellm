from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures._base import TimeoutError as ConcurrentTimeoutError
import time
from typing import List, Tuple
from openai import OpenAI
from pprint import pprint
import json
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import sys
import os
import random
from datasets import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.yelp_multi_tags.prepare import YelpMultiTagsCFGTokenizer, NON_FOOD_RELATED_TAGS, FOOD_RELATED_TAGS
from data.yelp_multi_tags.prepare import CUISINES as SMALLER_CUISINE



BATCH_SIZE = 5000
NUM_WORKERS = 90
IO_CALL_TIMEOUT_SECONDS = 50


def batch_process_with_retries(
    instructions : List[str], row_level_fn, *row_level_args
):
    MAX_RETRY_TIME = 50

    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    count = 0

    time_now = time.time()
    futures = {
        executor.submit(row_level_fn, instruction, i, *row_level_args): (instruction, i, None) #None is the time started
        for i, instruction in enumerate(instructions)
    }

    count_retries = 0


    updated_rows = []
    while len(futures) > 0:
        print(f"Number of futures for this batch: {len(futures)}")
        try:
            for future in as_completed(futures, timeout=IO_CALL_TIMEOUT_SECONDS):
                instruction, i, _ = futures[future]
                try:
                    # for every row that's completed, remove its future and add to updated rows
                    new_row, row_perf_metrics = future.result()
                    updated_rows.append((new_row, i, row_perf_metrics))
                    del futures[future]
                except Exception as e:
                    del futures[future]
                    print(f"Error processing instruction {i}: {e}")
                    # retry
                    futures[executor.submit(row_level_fn, instruction, i, *row_level_args)] = (
                        instruction,
                        i,
                        None,
                    )

                count += 1
                if count % 100 == 0:
                    datapoints_per_minute = (count / (time.time() - time_now)) * 60
                    if datapoints_per_minute > 5000:
                        # we're processing too fast, so we need to slow down
                        time.sleep(1)

                    print(f"Processed {count} of {len(instructions)} datapoints in {time.time() - time_now}s, {datapoints_per_minute} datapoints per minute")

        except ConcurrentTimeoutError:
            print(f"Timeout error of {IO_CALL_TIMEOUT_SECONDS}s, checking for retries")
            # restart any futures that have been retried too many times
            for future in list(futures.keys()):  # Create a copy of keys to iterate over
                if future.running():
                    instruction, i, time_started = futures[future]
                    if time_started is None:
                        futures[future] = (instruction, i, time.time())
                    elif time.time() - time_started > MAX_RETRY_TIME:
                        count_retries += 1
                        print(
                            f"Was not able to complete in {MAX_RETRY_TIME}s, have spent {time.time() - time_started}s, restarting future for instruction {i}"
                        )
                        del futures[future]
                        # Add new futures for the retried rows
                        futures[executor.submit(row_level_fn, instruction, i, *row_level_args)] = (
                            instruction,
                            i,
                            time.time(),
                        )
            print(f"Retried {count_retries} rows")

    time_taken = time.time() - time_now
    print(f"Time taken: {time_taken}s")
    
    executor.shutdown(wait=False, cancel_futures=True)

    # sort by i
    # check for unique i
    assert len(updated_rows) == len(set([i for _, i, _ in updated_rows]))
    # check that total number of rows is correct
    assert len(updated_rows) == len(instructions)
    updated_rows.sort(key=lambda x: x[1])

    return updated_rows, time_taken

prompt_prefix_easy_classifier = """Classify the following business reviews as a restaurant with its cuisine or not a restaurant. Example:
    Input: Barrigas Mexican Restaurant - This is our 4th time here and everything we order is so good. Today we ordered the Pastor con Quezo and the red snapper. 
    Output: restaurant(mexican)

    Input: Pasadena Car Wash & Auto Detailing - Terrible job, didn't even open the rear doors to vacuum. 
    Output: not_restaurant

    Input: The Black Bull Kitchen & Bar - The best Pizza & Burgers ever! All of the food here is top notch! Offering free delivery at this time and you can also order on Uber Eats or Door Dash!
    Output: restaurant(american)
    """

class CuisineEnum(str, Enum):
    japanese = "japanese"
    chinese = "chinese"
    indian = "indian"
    american = "american"
    italian = "italian"
    french = "french"
    mexican = "mexican"

class EasyPydanticModel(BaseModel):
    restaurant: bool = Field(description="Whether the review is about a restaurant")
    cuisine: CuisineEnum = Field(description="The cuisine of the restaurant if it is a restaurant. If the review is not about a restaurant, this can be anything")

    # @field_validator('cuisine', mode='after')
    # @classmethod
    # def remove_stopwords(cls, v: str, info: ValidationInfo) -> str:
    #     if info.data["restaurant"]:
    #         return v
    #     else:
    #         raise ValueError("Cuisine is not allowed for non-restaurant")
        
    def __str__(self):
        if self.restaurant:
            return  "{" + f"restaurant({self.cuisine})" + "}"
        else:
            return "{" + "not_restaurant" + "}"
        

class MultiClassCuisineEnum(str, Enum):
    mexican = "Mexican"
    italian = "Italian"
    chinese = "Chinese"
    american = "American"

class MultiClassFoodRelatedTagsEnum(str, Enum):
    bars = "Bars"
    sandwiches = "Sandwiches"
    pizza = "Pizza"
    coffee_tea = "Coffee & Tea"
    fast_food = "Fast Food"
    burgers = "Burgers"
    breakfast_brunch = "Breakfast & Brunch"
    specialty_food = "Specialty Food"
    seafood = "Seafood"
    desserts = "Desserts"
    bakeries = "Bakeries"
    salad = "Salad"
    chicken_wings = "Chicken Wings"


class MultiClassNonFoodRelatedTagsEnum(str, Enum):
    home_services = "Home Services"
    automotive = "Automotive"
    beauty_spas = "Beauty & Spas"
    nightlife = "Nightlife"
    health_medical = "Health & Medical"
    event_planning_services = "Event Planning & Services"
    active_life = "Active Life"
    hotels_travel = "Hotels & Travel"
    home_garden = "Home & Garden"
    fashion = "Fashion"
    arts_entertainment = "Arts & Entertainment"
    hair_salons = "Hair Salons"
    nail_salons = "Nail Salons"
    doctors = "Doctors"
    pets = "Pets"
    real_estate = "Real Estate"
    fitness_instruction = "Fitness & Instruction"


"""
E.g. 
NON_FOOD_TAGS{Active Life,Arts & Entertainment,Event Planning & Services}
FOOD_TAGS{RESTAURANT(Mexican),Fast Food}NON_FOOD_TAGS{Event Planning & Services}
FOOD_TAGS{Bars}NON_FOOD_TAGS{Nightlife}
"""

class TagClassifierModel(BaseModel):
    Cuisine: Optional[MultiClassCuisineEnum] = Field(description="The cuisine of the restaurant if it is a restaurant. Can be left blank if not clear.")
    FoodRelatedTags: List[MultiClassFoodRelatedTagsEnum] = Field(description="The food related tags of the review if it is a food related tag. Can be empty if not applicable.")
    NonFoodRelatedTags: List[MultiClassNonFoodRelatedTagsEnum] = Field(description="The non food related tags of the review if it is a non food related tag. Can be empty if not applicable.")

    def __str__(self):
        # sort the tags
        self.FoodRelatedTags.sort()
        self.NonFoodRelatedTags.sort()

        str_rep = ""
        if self.Cuisine or self.FoodRelatedTags:
            str_rep += "FOOD_TAGS{"
            if self.Cuisine:
                str_rep += f"RESTAURANT({MultiClassCuisineEnum(self.Cuisine).value})"
            if self.FoodRelatedTags:
                str_rep += f"{','.join([tag.value for tag in self.FoodRelatedTags])}"
            str_rep += "}"
        if self.NonFoodRelatedTags:
            str_rep += "NON_FOOD_TAGS{"
            str_rep += f"{','.join([tag.value for tag in self.NonFoodRelatedTags])}"
            str_rep += "}"
        return str_rep
    
prompt_prefix_multi_tags = """Add the correct tags to the following review. Each tag should only be added once. Be precise. There are 3 tag categories to choose from:
1. Cuisine: The cuisine of the restaurant if it is a restaurant.
2. FoodRelatedTags: The food related tags of the review if it is a food related tag.
3. NonFoodRelatedTags: The non food related tags of the review if it is a non food related tag.

Example:
Input: The Mystery Mansion - We did the serial killer escape room. My friends really enjoyed it; however, my flashlight was flickering.
Output: Cuisine: American, FOOD_TAGS: [Escape Room], NON_FOOD_TAGS: [Event Planning & Services]

Input: QDOBA Mexican Eats - Ordered 2 order 3 cheese nachos and both of us were disappointed because it was like dipping the chips in soup. They do deliver though.
Output: Cuisine: Mexican, FOOD_TAGS: [Fast Food], NON_FOOD_TAGS: [Event Planning & Services]

Input: The best Pizza & Burgers ever! All of the food here is top notch! Offering free delivery at this time and you can also order on Uber Eats or Door Dash!
Output: FOOD_TAGS: [Bars], NON_FOOD_TAGS: [Nightlife]

Input: Sacred Paradise Spa - Don't go to this site it's all about just getting your money and you do not get any service from them | I recently experienced two massage treatments at Sacred Paradise Spa. I went there based upon a word of mouth referral. 
Output: NON_FOOD_TAGS: [Active Life, Beauty & Spas, Fitness & Instruction, Health & Medical]

Actual Review:\n
"""

def get_instructions_and_expected(file_path, prompt_prefix, eval_size=1):
    with open(file_path, "r") as f:
        data = json.load(f)

    data = data[:eval_size]

    input_list = [item["input"] for item in data]
    instructions = [f"{prompt_prefix} {item}" for item in input_list]
    expected = [item["output"] for item in data]
    return instructions, expected, input_list

price_per_token_input_4o = 0.150 / 1000000
price_per_token_output_4o = 0.600 / 1000000

def get_cost_of_completion_4o_mini(input_token_count: int, output_token_count: int):
    return (
        input_token_count * price_per_token_input_4o
        + output_token_count * price_per_token_output_4o
    )


def get_chatgpt_completions(output_dir, cutoff, original_data_path, pydantic_model, prompt_prefix, metric = "accuracy"):
    instructions, expected, input_list = get_instructions_and_expected(original_data_path, prompt_prefix, eval_size=cutoff)

    # I had a previous bug where a bunch of the generated answers were {restaurant(None)}
    # quick fix:
    # all_already_completed_data = json.load(open(original_data_path, "r"))
    
    # indexes_of_wrong_answers = []
    # for i, item in enumerate(all_already_completed_data):
    #     if item["generated"] == "{restaurant(None)}":
    #         indexes_of_wrong_answers.append(i)

    # map of new order to old order
    # new_to_old_order = {i: j for i, j in enumerate(indexes_of_wrong_answers)}

    # # regenerate the instructions
    # instructions = [instructions[i] for i in indexes_of_wrong_answers]

    client = OpenAI()

    def row_level_fn(instruction, i, *row_level_args):
        completions = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": instruction}],
            # temperature=0.1,
            response_format=pydantic_model,
        )  

        input_token_count = completions.usage.prompt_tokens
        output_token_count = completions.usage.completion_tokens
        cost = get_cost_of_completion_4o_mini(input_token_count, output_token_count)


        return completions.choices[0].message.parsed, cost
    
    updated_rows, time_taken = batch_process_with_retries(instructions, row_level_fn)


    if (len(updated_rows) != len(instructions)):
        print(f"len(updated_rows): {len(updated_rows)}")
        print(f"len(instructions): {len(instructions)}")

    # do a string predicteed answers for every row but put None if the row index was not there
    string_predicted_answers = []
    curr_index = 0
    for row in updated_rows:
        # note that if it failed, we would be missing some indices
        if curr_index == row[1]:
            string_predicted_answers.append(str(row[0]))
        else:
            string_predicted_answers.append(None)
        curr_index += 1

    # string_predicted_answers = [str(row[0]) for row in updated_rows]

    # for curr_index, prev_index in new_to_old_order.items():
        # all_already_completed_data[prev_index]["generated"] = string_predicted_answers[curr_index]

    # pprint(list(zip(string_predicted_answers, expected)))
    # expected = [row["expected"] for row in all_already_completed_data]
    # input_list = [row["input"] for row in all_already_completed_data]
    # string_predicted_answers = [row["generated"] for row in all_already_completed_data]

    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    errors = 0
    for predicted_answer, expected_answer in zip(string_predicted_answers, expected):
        if predicted_answer is None:
            errors += 1
            continue
        if predicted_answer == expected_answer:
            correct += 1
        if metric == "f1":
            tokenized_expected_answer = YelpMultiTagsCFGTokenizer.tokenize(expected_answer)
            tokenized_predicted_answer = YelpMultiTagsCFGTokenizer.tokenize(predicted_answer)
            curr_true_positives, curr_false_positives, curr_false_negatives = YelpMultiTagsCFGTokenizer.get_f1_sub_metrics(tokenized_expected_answer, tokenized_predicted_answer)
            true_positives += curr_true_positives
            false_positives += curr_false_positives
            false_negatives += curr_false_negatives

    print(f"errors: {errors}")
    total_generations = len(string_predicted_answers) - errors

    accuracy = correct / total_generations
    f1_metrics = None
    if metric == "f1":
        f1_metrics = YelpMultiTagsCFGTokenizer.get_f1_score(true_positives, false_positives, false_negatives)
        

    # save the results
    # so now we have expected, generated, and the original input
    # save it as a json file
    results = []
    for expected, generated, input in zip(expected, string_predicted_answers, input_list):
        results.append({"expected": expected, "generated": generated, "input": input})


    file_name = f"{output_dir}/with_generations_2.json"
    with open(file_name, "w") as f:
        json.dump(results, f)


    return accuracy, time_taken, f1_metrics

def get_gpt_completions_multitags():
    for tag in MultiClassNonFoodRelatedTagsEnum:
        assert tag.value in NON_FOOD_RELATED_TAGS
    for tag in NON_FOOD_RELATED_TAGS:
        # check that the tag is in the enum (but in won't work because of the enum type)
        assert any(tag == enum.value for enum in MultiClassNonFoodRelatedTagsEnum)

    for cuisine in MultiClassCuisineEnum:
        assert cuisine.value in SMALLER_CUISINE
    for cuisine in SMALLER_CUISINE:
        assert any(cuisine == enum.value for enum in MultiClassCuisineEnum)


    accuracy, time_taken, f1_metrics = get_chatgpt_completions(
        output_dir="./data/yelp_multi_tags",
        cutoff=10000000,
        original_data_path="./data/yelp_multi_tags/multi_tags_yelp.json",
        pydantic_model=TagClassifierModel,
        prompt_prefix=prompt_prefix_multi_tags,
        metric="f1"
    )
    print(f"Accuracy: {accuracy}")
    print(f"Time taken: {time_taken}s")
    if f1_metrics:  
        print(f"F1 score: {f1_metrics[0]}")
        print(f"Precision: {f1_metrics[1]}")
        print(f"Recall: {f1_metrics[2]}")


def get_hf_prompt_completion_dataset(max_train_samples, path_to_json_file, task, is_training=True):
    if task == "multi_tags":
        prompt_prefix = prompt_prefix_multi_tags
    elif task == "easy":
        prompt_prefix = prompt_prefix_easy_classifier
    else:
        raise ValueError(f"Task {task} not supported")

    # Path to your JSON file
    data = json.load(open(path_to_json_file, "r"))
    if is_training:
        data = data[:max_train_samples]
    else:
        data = data[max_train_samples:]

    

    # Optional: Format it into a prompt-completion format
    def format_prompt_completion(example):
        return {
            "prompt": prompt_prefix + example["input"],
            "completion": example["generated"],  # or use "expected" if preferred,
            "expected": example["expected"]
        }

    # Apply formatting
    formatted_data = [format_prompt_completion(entry) for entry in data]

    # Create Hugging Face Dataset
    dataset = Dataset.from_list(formatted_data)

    return dataset


def get_hf_prompt_only_dataset_from_full_prompt_completion_dataset(dataset : Dataset):
    return dataset.map(lambda x: x["prompt"])


if __name__ == "__main__":
    # first check that the enums and the tags are a 1:1 matching
    dataset = get_hf_prompt_completion_dataset(max_train_samples=10000000, path_to_json_file="./data/yelp_multi_tags/with_generations_2.json", task = "multi_tags")
    max_examples = 10
    for example in dataset:
        print(example)
        max_examples -= 1
        if max_examples <= 0:
            break



