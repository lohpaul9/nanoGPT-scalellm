import json
from typing import List, Optional
import psycopg2
import random
import re
import torch
import time
import numpy as np
import os
import pickle
from datasets import load_dataset, Dataset
from typing import Tuple
import tiktoken

# Define the cuisines from the queries.py file
"""
 Restaurants                          | 52268
 Food                                 | 27781
 Shopping                             | 24395
 Bars                                 | 11065
 Sandwiches                           |  8366
 Pizza                                |  7093
 Coffee & Tea                         |  6703
 Fast Food                            |  6472
 Burgers                              |  5636
 Breakfast & Brunch                   |  6239
 American (New)                       |  6097
 American (Traditional)               |  8139
 Mexican                              |  4600
 Italian                              |  4573
 Specialty Food                       |  4233
 Seafood                              |  3539
 Desserts                             |  3186
 Chinese                              |  3169
 Bakeries                             |  3150
 Salad                                |  3064
 Chicken Wings                        |  2966
 Home Services                        | 14356
 Automotive                           | 10773
 Beauty & Spas                        | 14292
 Nightlife                            | 12281
 Health & Medical                     | 11890
 Event Planning & Services            |  9895
 Active Life                          |  7687
 Hotels & Travel                      |  5857
 Home & Garden                        |  5799
 Fashion                              |  5739
 Arts & Entertainment                 |  5434
 Auto Repair                          |  5433
 Hair Salons                          |  5046
 Nail Salons                          |  4621
 Doctors                              |  3763
 Pets                                 |  3758
 Real Estate                          |  3577
 Fitness & Instruction                |  3293
"""

FOOD_RELATED_TAGS = [
    "Bars",
    "Sandwiches",
    "Pizza",
    "Coffee & Tea",
    "Fast Food",
    "Burgers",
    "Breakfast & Brunch",
    "Specialty Food",
    "Seafood",
    "Desserts",
    "Bakeries",
    "Salad",
    "Chicken Wings",
    # "Chinese"
]

CUISINES = [
    "Mexican",
    "Italian",
    "Chinese",
    "American"
]

NON_FOOD_RELATED_TAGS = [
    "Home Services",
    "Automotive",
    "Beauty & Spas",
    "Nightlife",
    "Health & Medical",
    "Event Planning & Services",
    "Active Life",
    "Hotels & Travel",
    "Home & Garden",
    "Fashion",
    "Arts & Entertainment",
    "Hair Salons",
    "Nail Salons",
    "Doctors",
    "Pets",
    "Real Estate",
    "Fitness & Instruction",
    # "Auto Repair"
]


"""
Format of the output:
FOOD_TAGS(RESTAURANT(CUISINE),FOOD_TAG1,FOOD_TAG2,FOOD_TAG3),NON_FOOD_TAGS(NON_FOOD_TAG1,NON_FOOD_TAG2,NON_FOOD_TAG3,etc)
"""


def connect_to_db():
    """Connect to the PostgreSQL database."""
    conn = psycopg2.connect(
        host="localhost", database="dev", user="postgres", password="postgres"
    )
    cur = conn.cursor()
    print("Connected to the database")
    return cur, conn

def generate_training_examples_from_reviews(num_examples=50000):
    """
    Generate training examples from the all_businesses_with_top3_reviews table.
    
    Args:
        num_examples: Maximum number of examples to generate
        
    Returns:
        List of dictionaries with 'input' and 'output' fields
    """
    cur, conn = connect_to_db()
    
    # Query to get businesses with their reviews
    query = """
    SELECT 
        business_id,
        name,
        categories,
        recent_3_reviews
    FROM 
        all_businesses_with_top3_reviews
    """
    
    cur.execute(query)
    rows = cur.fetchall()
    
    examples = []
    
    for row in rows:
        business_id, name, categories, reviews = row
        
        # Skip if no reviews
        if not reviews:
            continue
            
        # Create input text by concatenating business name and reviews
        input_text = f"{name} - {reviews}"

        cuisines = []
        non_food_tags = []
        food_tags = []

        for tag in categories:
            if tag in CUISINES:
                cuisines.append(tag)
            elif tag in FOOD_RELATED_TAGS:
                food_tags.append(tag)
            elif tag in NON_FOOD_RELATED_TAGS:
                non_food_tags.append(tag)

        cuisine = None
        if len(cuisines) == 1:
            cuisine = cuisines[0]

        # sort the rest of the tags
        food_tags.sort()
        non_food_tags.sort()

        cfg_string = ""
        if len(food_tags) > 0 or cuisine is not None:
            cfg_string = "FOOD_TAGS{"
            if cuisine is not None:
                cfg_string += f"RESTAURANT({cuisine})"
            if len(food_tags) > 0:
                if cuisine is not None:
                    cfg_string += ","
                food_tag_str = ",".join(food_tags)
                cfg_string += f"{food_tag_str}"
            cfg_string += "}"

        if len(non_food_tags) > 0:
            cfg_string += f"NON_FOOD_TAGS{{{','.join(non_food_tags)}}}"

        if not cfg_string:
            continue
        
        # Add to examples
        examples.append({
            "input": input_text,
            "output": cfg_string
        })
        # print(name)
        # print(cfg_string if cfg_string else "no tags")
        
        # Break if we have enough examples
        if len(examples) >= num_examples:
            break
    
    cur.close()
    conn.close()

    print(f"Generated {len(examples)} examples")

    # shuffle the examples
    random.shuffle(examples)
    
    return examples

def save_examples(examples, output_file="multi_tags_yelp.json"):
    """Save examples to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved {len(examples)} examples to {output_file}")


class YelpMultiTagsCFGTokenizer:
    special_tokens = ["<PAD>", "<EOS>", "<BOS>"]
    cfg_tokens = ["RESTAURANT", "FOOD_TAGS", "NON_FOOD_TAGS", "(", ")", ",", "{", "}"]

    cuisine_tokens = CUISINES
    food_tag_tokens = FOOD_RELATED_TAGS
    non_food_tag_tokens = NON_FOOD_RELATED_TAGS

    all_tokens = special_tokens + cfg_tokens + cuisine_tokens + food_tag_tokens + non_food_tag_tokens
    tokens_to_id = {token: i for i, token in enumerate(all_tokens)}
    tokens_to_id["<PAD>"] = -1

    print(f"Tokens to ID: {tokens_to_id}")

    @classmethod
    def tokenize(self, text: str) -> List[int]:
        tokens = [self.tokens_to_id["<BOS>"]]

        # this is an example of the tokenized text:
        # FOOD_TAGS{RESTAURANT(Mexican),Bars}NON_FOOD_TAGS{Nightlife}
        # note that all of the sub-fields are optional

        # now I need something that can tokenize the text into the above format
        # just convert the string into tokens, which may not be delimited by spaces actually
        # so we need to split the string into tokens

        # we know tha the string is well-formed, so we can look ahead to see if the next token is a valid token

        while text:
            found_token = False
            for token in self.all_tokens:
                if text.startswith(token):
                    tokens.append(self.tokens_to_id[token])
                    text = text[len(token):]
                    found_token = True
                    break
            if not found_token:
                raise ValueError(f"Invalid token:{text}")

        tokens.append(self.tokens_to_id["<EOS>"])

        return tokens

    @classmethod
    def deconstruct_to_properties(cls, tokens: List[int]):
        cuisine = None
        food_tags = []
        non_food_tags = []

        # change pad to 0 
        tokens = [0 if token == -1 else token for token in tokens]

        stringified_tokens = [cls.all_tokens[token] for token in tokens]

        for token in stringified_tokens:
            if token in cls.cuisine_tokens:
                cuisine = token
            elif token in cls.food_tag_tokens:
                food_tags.append(token)
            elif token in cls.non_food_tag_tokens:
                non_food_tags.append(token)

        # make unique 
        food_tags = list(set(food_tags))
        
        return cuisine, food_tags, non_food_tags


    @classmethod
    def get_f1_sub_metrics(cls, expected: List[int], predicted: List[int]):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        expected_cuisine, expected_food_tags, expected_non_food_tags = cls.deconstruct_to_properties(expected)
        predicted_cuisine, predicted_food_tags, predicted_non_food_tags = cls.deconstruct_to_properties(predicted)

        # update precision, recall, and F1 score
        all_expected_tags = expected_food_tags + expected_non_food_tags
        if expected_cuisine is not None:
            all_expected_tags.append(expected_cuisine)
        all_predicted_tags = predicted_food_tags + predicted_non_food_tags
        if predicted_cuisine is not None:
            all_predicted_tags.append(predicted_cuisine)

        for tag in all_expected_tags:
            if tag in all_predicted_tags:
                true_positives += 1
            else:
                false_negatives += 1
        for tag in all_predicted_tags:
            if tag not in all_expected_tags:
                false_positives += 1
        
        return true_positives, false_positives, false_negatives
    
    @classmethod
    def get_f1_score(cls, true_positives: int, false_positives: int, false_negatives: int) -> Tuple[float, float, float]:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score, precision, recall

    @classmethod
    def detokenize(self, tokens: List[int], require_eos: bool = True) -> str:
        # cutoff at first <EOS>
        eos_id = self.tokens_to_id["<EOS>"]
        if eos_id not in tokens:
            if require_eos:
                raise ValueError(f"No <EOS> found in tokens: {self.detokenize(tokens, require_eos=False)}")
            else:
                eos_index = len(tokens)
        else:
            eos_index = tokens.index(eos_id)

        tokens = tokens[:eos_index]

        ignore_tokens = [-1, self.tokens_to_id["<PAD>"], self.tokens_to_id["<BOS>"]]

        # now we need to convert the tokens back into the original string
        return "".join([self.all_tokens[token] for token in tokens if token not in ignore_tokens])
    

    @classmethod
    def get_possible_next_tokens_indexes(cls, list_of_tokens: List[int]) -> List[int]:
        if len(list_of_tokens) == 0:
            return [cls.tokens_to_id["<BOS>"]]

        last_token_id = list_of_tokens[-1]
        last_token = cls.all_tokens[last_token_id]
        restaurant_token_id = cls.tokens_to_id["RESTAURANT"]
        food_tags_token_id = cls.tokens_to_id["FOOD_TAGS"]
        non_food_tags_token_id = cls.tokens_to_id["NON_FOOD_TAGS"]

        valid_tokens = []

        if last_token == "<BOS>":
            valid_tokens = ["NON_FOOD_TAGS", "FOOD_TAGS", "<EOS>"]
        elif last_token == "NON_FOOD_TAGS":
            valid_tokens = ["{"]
        elif last_token == "FOOD_TAGS":
            valid_tokens = ["{"]
        elif last_token == "RESTAURANT":
            valid_tokens = ["("]
        elif last_token == "(":
            valid_tokens = cls.cuisine_tokens
        elif last_token == ")":
            valid_tokens = ["}", ","] + cls.food_tag_tokens
        elif last_token in cls.cuisine_tokens:
            valid_tokens = [")"]
        elif last_token == "{":
            if non_food_tags_token_id in list_of_tokens:
                valid_tokens = cls.non_food_tag_tokens
            elif food_tags_token_id in list_of_tokens:
                valid_tokens = cls.food_tag_tokens + ["RESTAURANT"]
            else:
                raise ValueError(f"Invalid tokens: {list_of_tokens}")
        elif last_token == "}":
            if non_food_tags_token_id in list_of_tokens:
                valid_tokens = ["<EOS>"]
            elif food_tags_token_id in list_of_tokens:
                valid_tokens = ["NON_FOOD_TAGS", "<EOS>"]
            else:
                raise ValueError(f"Invalid tokens: {list_of_tokens}")
        elif last_token == ",":
            # case 2: non-food
            if non_food_tags_token_id in list_of_tokens:
                valid_tokens = ["}"] + cls.non_food_tag_tokens
            # case 1: food
            elif food_tags_token_id in list_of_tokens:
                valid_tokens = ["}"] + cls.food_tag_tokens
            else:
                raise ValueError(f"Invalid tokens: {list_of_tokens}")
        elif last_token in cls.food_tag_tokens:
            valid_tokens = ["}", ","]
        elif last_token in cls.non_food_tag_tokens:
            valid_tokens = ["}", ","]
        elif last_token == "<EOS>":
            valid_tokens = ["<EOS>"]
        else:
            raise ValueError(f"Invalid tokens: {cls.detokenize(list_of_tokens)}, {list_of_tokens}")

        return [cls.tokens_to_id[token] for token in valid_tokens]
    
    @classmethod
    def get_possible_next_tokens_indexes_mask(cls, list_of_tokens: List[int]) -> torch.Tensor:
        possible_next_tokens_indexes = cls.get_possible_next_tokens_indexes(list_of_tokens)
        next_tokens_indexes_torch = torch.tensor(possible_next_tokens_indexes, dtype=torch.long)
        mask = torch.zeros(len(cls.all_tokens), dtype=torch.bool)
        mask[next_tokens_indexes_torch] = True
        return mask


    @classmethod
    def verify_cfg_string(cls, cfg_string: str) -> bool:
        tokenized_cfg_string = cls.tokenize(cfg_string)
        for i in range(len(tokenized_cfg_string) - 1):
            # check if the next token is valid
            next_token = tokenized_cfg_string[i]
            next_possible_tokens = cls.get_possible_next_tokens_indexes(tokenized_cfg_string[:i])
            if next_token not in next_possible_tokens:
                print(f"Invalid token: {next_token}")
                print(f"Possible tokens: {next_possible_tokens}")
                print(f"Current tokens: {tokenized_cfg_string[:i]}")
                print(f"current string {cls.detokenize(tokenized_cfg_string[:i])}")
                return False
            
        return True    

    @classmethod
    def get_vocab_size(cls) -> int:
        return len(cls.all_tokens)
    
    @classmethod
    def get_token_id(cls, token: str) -> int:
        return cls.tokens_to_id[token]
    
    @classmethod
    def stringify_from_properties(cls, food_related_tags: List[str], non_food_related_tags: List[str], cuisine: str):
        # sort the tags
        # make unique
        food_related_tags = list(set(food_related_tags))
        non_food_related_tags = list(set(non_food_related_tags))
        food_related_tags.sort()
        non_food_related_tags.sort()

        # remove the food related tag chinese
        food_related_tags = [tag for tag in food_related_tags if tag != "Chinese"]

        # remove the non-food related tag auto repair
        non_food_related_tags = [tag for tag in non_food_related_tags if tag != "Auto Repair"]


        str_rep = ""
        if cuisine or food_related_tags:
            str_rep += "FOOD_TAGS{"
            if cuisine:
                str_rep += f"RESTAURANT({cuisine})"
                if food_related_tags:
                    str_rep += ","
            if food_related_tags:
                str_rep += f"{','.join(food_related_tags)}"
            str_rep += "}"
        if non_food_related_tags:
            str_rep += "NON_FOOD_TAGS{"
            str_rep += f"{','.join(non_food_related_tags)}"
            str_rep += "}"
        return str_rep
            

block_size = 48
from data.yahooreviewcuisine.prepare import stella_400m
output_dir = "data/yelp_multi_tags_2"


def fix_legacy_string(generation):
    tokenized_generation = YelpMultiTagsCFGTokenizer.tokenize(generation)
    cuisine, food_tags, non_food_tags = YelpMultiTagsCFGTokenizer.deconstruct_to_properties(tokenized_generation)
    reconstructed_simplified_string = YelpMultiTagsCFGTokenizer.stringify_from_properties(food_tags, non_food_tags, cuisine)
    return reconstructed_simplified_string


def get_cost_per_example():
    # get cost per example of gpt 4o mini
    file_name = "data/yelp_multi_tags_2/llama_3_3_generations_2_fixed.json"

    with open(file_name, "r") as f:
        data = json.load(f)

    # get the cost per example
    input_cost_per_token = 0.15 / 1000000
    output_cost_per_token = 0.6 / 1000000
    total_cost = 0
    for item in data:
        input_tokens = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(item["input"]))
        output_tokens = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(item["generated"]))
        total_cost += input_cost_per_token * input_tokens + output_cost_per_token * output_tokens
    print(f"Total cost: {total_cost}")
    print(f"Cost per example: {1000 * total_cost / len(data)}")

def wrangle_data(list_of_llama_data_file_names):
    """
    I have two files: one is llama_3_3_generations.json, which is the generations from llama 3.3 with 'generated' field as the generations
    The other is with_generations_2.json, which is the generations from gpt as the 'generations' field

    I want to merge the two files based on the hash of the 'input' field, such that I store llama generations as 'generated_llama' and gpt generations as just 'generated'
    Then put the ones from llama as the first samples in the list

    Finally save to json
    """
    import hashlib

    llama_datas = []

    for llama_data_file_name in list_of_llama_data_file_names:
        with open(os.path.join(output_dir, llama_data_file_name), "r") as f:
            llama_data = json.load(f)
            llama_datas += llama_data

    print(f"Total llama data: {len(llama_datas)}")

    with open(os.path.join(output_dir, "with_generations_2_fixed.json"), "r") as f:
        gpt_data = json.load(f)

    # create a dictionary of the gpt data
    gpt_data_dict = {hashlib.sha256(item["input"].encode()).hexdigest(): item for item in gpt_data}

    count_of_llama = 0
    for item in llama_datas:
        if "generated_llama" not in item:
            continue
        count_of_llama += 1
        hash_of_input = hashlib.sha256(item["input"].encode()).hexdigest()
        if hash_of_input in gpt_data_dict:
            assert gpt_data_dict[hash_of_input]["generated"] == item["generated"]
            assert gpt_data_dict[hash_of_input]["input"] == item["input"]

            # check that it's not a broken generation
            try:
                tokens = YelpMultiTagsCFGTokenizer.tokenize(item["generated_llama"])
                reconstructed_simplified_string_llama = fix_legacy_string(item["generated_llama"])
            except Exception as e:
                print(f"Error: {e}")
                print(f"Item: {item['generated_llama']}")
                continue
            if len(tokens) > block_size:
                print(f"Wrong generation length: {item['generated_llama']}")
                continue

            gpt_data_dict[hash_of_input]["generated_llama"] = reconstructed_simplified_string_llama

        else:
            raise ValueError(f"Input not found in gpt data: {item['input']}")

    print(f"Count of llama: {count_of_llama}")

    final_list_to_save = []

    for item in gpt_data_dict.values():
        if "generated_llama" in item:
            final_list_to_save.append(item)

    for item in gpt_data_dict.values():
        if "generated_llama" not in item:
            final_list_to_save.append(item)

    count = 0
    for item in final_list_to_save:
        if "generated_llama" in item:   
            count += 1
        else:
            break
    
    print(f"Count: {count}")
    print(f"Total: {len(final_list_to_save)}")

    total_true_positives_llama = 0  
    total_false_positives_llama = 0
    total_false_negatives_llama = 0

    total_true_positives_gpt = 0
    total_false_positives_gpt = 0
    total_false_negatives_gpt = 0




    for i in range(len(final_list_to_save)):
        if "generated_llama" in final_list_to_save[i]:
            generated_llama = YelpMultiTagsCFGTokenizer.tokenize(final_list_to_save[i]["generated_llama"])
            generated = YelpMultiTagsCFGTokenizer.tokenize(final_list_to_save[i]["generated"])
            expected = YelpMultiTagsCFGTokenizer.tokenize(final_list_to_save[i]["expected"])

            

            # check if the generated llama is a true positive
            llama_true_positives, llama_false_positives, llama_false_negatives = YelpMultiTagsCFGTokenizer.get_f1_sub_metrics(expected, generated_llama)
            gpt_true_positives, gpt_false_positives, gpt_false_negatives = YelpMultiTagsCFGTokenizer.get_f1_sub_metrics(expected, generated)

            total_true_positives_llama += llama_true_positives
            total_false_positives_llama += llama_false_positives
            total_false_negatives_llama += llama_false_negatives

            total_false_negatives_gpt += gpt_false_negatives
            total_false_positives_gpt += gpt_false_positives
            total_true_positives_gpt += gpt_true_positives
            
    f1_llama, precision_llama, recall_llama = YelpMultiTagsCFGTokenizer.get_f1_score(total_true_positives_llama, total_false_positives_llama, total_false_negatives_llama)
    f1_gpt, precision_gpt, recall_gpt = YelpMultiTagsCFGTokenizer.get_f1_score(total_true_positives_gpt, total_false_positives_gpt, total_false_negatives_gpt)

    print(f"F1 llama: {f1_llama}, precision llama: {precision_llama}, recall llama: {recall_llama}")
    print(f"F1 gpt: {f1_gpt}, precision gpt: {precision_gpt}, recall gpt: {recall_gpt}")
            
            
    # save the merged data
    with open(os.path.join(output_dir, "llama_3_3_generations_2_fixed.json"), "w") as f:
        json.dump(final_list_to_save, f, indent=4)
        
        

def prepare_data(keep_embedding_unchanged: bool = False):
    # 1. Load reviews.json
    with open(os.path.join(output_dir, "llama_3_3_generations_2_fixed.json"), "r") as f:
        data = json.load(f)


    for i in range(len(data)):
        if "generated_llama" not in data[i]:
            cutoff_with_last_generated = i
            break
    data_llama_new_generated_only = data[:cutoff_with_last_generated]
    generation_llama_outputs = [item["generated_llama"] for item in data_llama_new_generated_only]

    generation_gpt_outputs = [item["generated"] for item in data]

    expected_outputs = [item["expected"] for item in data]

    # 3. Tokenize and pad outputs
    tokenizer = YelpMultiTagsCFGTokenizer()
    expected_tokens = []
    generated_tokens_list_llama = []
    generated_tokens_list_gpt = []

    expected_tokens = []
    generated_tokens_list = []
    for i in range(len(expected_outputs)):
        tokens = tokenizer.tokenize(expected_outputs[i])
        # Pad or truncate to block_size
        if len(tokens) < block_size:
            tokens += [tokenizer.tokens_to_id["<PAD>"]] * (block_size - len(tokens))
        elif len(tokens) > block_size:
            raise ValueError(f"Generated token length is greater than block size: {len(generated_tokens)} > {block_size}")
        expected_tokens.append(tokens)
    
    for i in range(len(generation_llama_outputs)):
        # get the generated tokens
        generated_tokens = tokenizer.tokenize(generation_llama_outputs[i])
        cuisine, food_tags, non_food_tags = tokenizer.deconstruct_to_properties(tokenizer.tokenize(generation_llama_outputs[i]))
        reconstructed_simplified_string = tokenizer.stringify_from_properties(food_tags, non_food_tags, cuisine)
        
        generated_tokens = tokenizer.tokenize(reconstructed_simplified_string)
        
        # Pad or truncate to block_size
        if len(generated_tokens) < block_size:
            generated_tokens += [tokenizer.tokens_to_id["<PAD>"]] * (block_size - len(generated_tokens))
        elif len(generated_tokens) > block_size:
            raise ValueError(f"Generated token length is greater than block size: {len(generated_tokens)} > {block_size} for example {i} {generation_llama_outputs[i]}")
        generated_tokens_list_llama.append(generated_tokens)

    for i in range(len(generation_gpt_outputs)):
        generated_tokens = tokenizer.tokenize(generation_gpt_outputs[i])
        cuisine, food_tags, non_food_tags = tokenizer.deconstruct_to_properties(tokenizer.tokenize(generation_gpt_outputs[i]))
        reconstructed_simplified_string = tokenizer.stringify_from_properties(food_tags, non_food_tags, cuisine)
        
        generated_tokens = tokenizer.tokenize(reconstructed_simplified_string)
        if len(generated_tokens) < block_size:
            generated_tokens += [tokenizer.tokens_to_id["<PAD>"]] * (block_size - len(generated_tokens))
        elif len(generated_tokens) > block_size:
            raise ValueError(f"Generated token length is greater than block size: {len(generated_tokens)} > {block_size}")
        generated_tokens_list_gpt.append(generated_tokens)

    expected_tokens = np.array(expected_tokens, dtype=np.int16)  # shape: (num_samples, block_size)
    generated_tokens_list_llama = np.array(generated_tokens_list_llama, dtype=np.int16)  # shape: (num_samples, block_size)
    generated_tokens_list_gpt = np.array(generated_tokens_list_gpt, dtype=np.int16)  # shape: (num_samples, block_size)
    
    print(f"Shape of expected tokens: {expected_tokens.shape}")
    print(f"Shape of generated tokens list llama: {generated_tokens_list_llama.shape}")
    print(f"Shape of generated tokens list gpt: {generated_tokens_list_gpt.shape}")

    if not keep_embedding_unchanged:
        inputs = [item["input"] for item in data]
        # 2. Embed inputs
        time_start = time.time()
        stella = stella_400m()  # Replace with your actual Stella model
        input_embeddings = stella.embed(inputs)
        time_end = time.time()
        print(f"Time taken to embed inputs: {time_end - time_start} seconds")
        print(f"Shape of input embeddings: {input_embeddings.shape}")


    # check that all the lists are the same length
    if keep_embedding_unchanged:
        assert len(expected_tokens) == len(generated_tokens_list_gpt)
    else:
        assert len(expected_tokens) == len(input_embeddings) == len(generated_tokens_list_gpt)

    # 4. Save to files
    if not keep_embedding_unchanged:
        np.save(os.path.join(output_dir, "input_embeddings_3.npy"), input_embeddings)
    np.save(os.path.join(output_dir, "output_tokens_3.npy"), expected_tokens)
    np.save(os.path.join(output_dir, "generated_tokens_list_3.npy"), generated_tokens_list_gpt)
    np.save(os.path.join(output_dir, "generated_tokens_list_llama_3.npy"), generated_tokens_list_llama)
    print("Finished saving!")

    # Save tokenizer metadata for reference
    with open(os.path.join(output_dir, "tokenizer_meta.pkl"), "wb") as f:
        pickle.dump({
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_token_id": tokenizer.tokens_to_id["<PAD>"],
            "eos_token_id": tokenizer.tokens_to_id["<EOS>"],
        }, f)



in_memory_input_embeddings_multi_tags = None
in_memory_output_tokens_multi_tags = None
in_memory_generated_tokens_list_multi_tags = None
rng = np.random.default_rng(2)


def get_batch_yelp_multi_tags(batch_size: int,  train_samples_limit, split: str = "train", device: Optional[torch.device] = None, use_llama=False):
    global in_memory_input_embeddings_multi_tags, in_memory_output_tokens_multi_tags, in_memory_generated_tokens_list_multi_tags

    if in_memory_input_embeddings_multi_tags is None:
        in_memory_input_embeddings_multi_tags = np.load(os.path.join(output_dir, "input_embeddings_3.npy"))
        print(f"Loaded input embeddings for {in_memory_input_embeddings_multi_tags.shape[0]} samples")
    if in_memory_output_tokens_multi_tags is None:
        in_memory_output_tokens_multi_tags = np.load(os.path.join(output_dir, "output_tokens_3.npy")) 
        print(f"Loaded output tokens for {in_memory_output_tokens_multi_tags.shape[0]} samples")
    if in_memory_generated_tokens_list_multi_tags is None:
        if use_llama:
            print(f"Was told to use llama")
            in_memory_generated_tokens_list_multi_tags = np.load(os.path.join(output_dir, "generated_tokens_list_llama_3.npy"))
        else:
            in_memory_generated_tokens_list_multi_tags = np.load(os.path.join(output_dir, "generated_tokens_list_3.npy"))
        print(f"Loaded generated tokens list for {in_memory_generated_tokens_list_multi_tags.shape[0]} samples")
    
    num_samples = in_memory_input_embeddings_multi_tags.shape[0]
    if split == "val":
        num_samples = in_memory_generated_tokens_list_multi_tags.shape[0]

    assert not (train_samples_limit is None and split == "train")
    # batch size is only valid for training split (batches are done outside of this for val)
    assert (batch_size is not None and split == "train") or (batch_size is None and split != "train")

    if split == "train":
        low = 0
        high = train_samples_limit
    elif split == "val" or split == "test":
        low = train_samples_limit
        high = num_samples
    else:
        raise ValueError(f"Invalid split: {split}")
    
    possible_choice_num = high - low
    if batch_size is not None and batch_size > possible_choice_num:
        print(f"Possible choice number is smaller than batch size: {possible_choice_num} < {batch_size}")
        batch_size = possible_choice_num
    

    
    if split == "train":
        idx = rng.choice(possible_choice_num, size=batch_size, replace=False)
        idx += low
    else:   
        idx = np.arange(low, high)

    x_embed = in_memory_input_embeddings_multi_tags[idx]  # (batch_size, embedding_dim)
    if split == "train":
        tokens = in_memory_generated_tokens_list_multi_tags[idx]      # (batch_size, block_size)
    elif split == "val":
        tokens = in_memory_generated_tokens_list_multi_tags[idx]      # (batch_size, block_size)
    elif split == "test":
        tokens = in_memory_output_tokens_multi_tags[idx]      # (batch_size, block_size)

    # convert all to torch tensors 
    x_embed = torch.from_numpy(x_embed)
    # convert to int64
    tokens = tokens.astype(np.int64)
    tokens = torch.from_numpy(tokens)

    # For next-token prediction: x = tokens[:-1] and end padded with <PAD>
    x_tokens = tokens
    y_tokens = tokens[:, 1:]
    # pad with <PAD> (-1) to the right

    y_tokens_size = y_tokens.shape[0]
    y_tokens = torch.cat([y_tokens, -1 * torch.ones(y_tokens_size, 1, dtype=torch.int64)], dim=1)

    # put on device
    # x_embed = x_embed.to(device)
    # x_tokens = x_tokens.to(device)
    # y_tokens = y_tokens.to(device)

    return x_embed, x_tokens, y_tokens
    

def fix_legacy_cfg():
    with open(os.path.join(output_dir, "with_generations_2.json"), "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        generated = data[i]["generated"]
        expected = data[i]["expected"]

        tokenized_generation = YelpMultiTagsCFGTokenizer.tokenize(generated)
        tokenized_expected = YelpMultiTagsCFGTokenizer.tokenize(expected)

        cuisine, food_related_tags, non_food_related_tags = YelpMultiTagsCFGTokenizer.deconstruct_to_properties(tokenized_generation)
        cuisine_expected, food_related_tags_expected, non_food_related_tags_expected = YelpMultiTagsCFGTokenizer.deconstruct_to_properties(tokenized_expected)

        generation_str = YelpMultiTagsCFGTokenizer.stringify_from_properties(food_related_tags, non_food_related_tags, cuisine)
        expected_str = YelpMultiTagsCFGTokenizer.stringify_from_properties(food_related_tags_expected, non_food_related_tags_expected, cuisine_expected)

        data[i]["generated"] = generation_str
        data[i]["expected"] = expected_str

        print(f"Example {i}:")
        print(f"Generated before: {generated}")
        print(f"Generated after: {generation_str}")
        print(f"Expected before: {expected}")
        print(f"Expected after: {expected_str}")
        print()


    with open(os.path.join(output_dir, "with_generations_2_fixed.json"), "w") as f:
        json.dump(data, f, indent=4)
        

    


def main():
    # Generate examples
    print("Generating training examples from reviews...")
    examples = generate_training_examples_from_reviews(1000000)

    t_start = time.time()
    for example in examples:
        tokenized_example = YelpMultiTagsCFGTokenizer.tokenize(example["output"])
        detokenized_example = YelpMultiTagsCFGTokenizer.detokenize(tokenized_example)
        assert detokenized_example == example["output"]
        if not YelpMultiTagsCFGTokenizer.verify_cfg_string(detokenized_example):
            print(f"Invalid example: {example['output']}")
            break
    t_end = time.time()
    print(f"Time taken: {t_end - t_start} seconds")

    print(f"Total examples: {len(examples)}")

    # Save examples
    save_examples(examples, "data/yelp_multi_tags/multi_tags_yelp.json")

    prepare_data()

    # example = [4, 9, 21, 8, 16, 10, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    # print(YelpMultiTagsCFGTokenizer.detokenize(example))




if __name__ == "__main__":
    # # main() 
    # # open the json file, and check how many generations are none
    # with open("data/yelp_multi_tags/with_generations_2.json", "r") as f:
    #     data = json.load(f)

    # print(f"Total examples: {len(data)}")

    # # compare the top 20 generaitons and expected
    # for i in range(20):
    #     print(f"Example {i}:")
    #     print(f"Expected: {data[i]['expected']}")
    #     print(f"Generated: {data[i]['generated']}")
    #     print()

    # none_count = 0
    # for item in data:
    #     if item["generated"] is None:
    #         none_count += 1

    # print(f"None count: {none_count}")

    # prepare_data()
    # fix_legacy_cfg()
    list_of_llama_data_file_names = [
        "llama_3_3_generations_0_540.json",
        "llama_3_3_generations_540_640.json",
        "llama_3_3_generations_644_944.json",
        "llama_3_3_generations_948_1048.json",
        "llama_3_3_generations_1048_1248.json",
        "llama_3_3_generations_1248_100.json",
        "llama_3_3_generations_1348_20.json",
        "llama_3_3_generations_1368_1568.json",
        "llama_3_3_generations_1568_1768.json",
        "llama_3_3_generations_1768_1968.json",
        "llama_3_3_generations_1968_2168.json",
    ]

    wrangle_data(list_of_llama_data_file_names)
    prepare_data(keep_embedding_unchanged=False)

    # get_cost_per_example()