import json
from typing import List
import psycopg2
import random
import re
import torch
import time
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
    "Chinese",
    "Bakeries",
    "Salad",
    "Chicken Wings"
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
    "Auto Repair",
    "Hair Salons",
    "Nail Salons",
    "Doctors",
    "Pets",
    "Real Estate",
    "Fitness & Instruction",
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
            for token in self.all_tokens:
                if text.startswith(token):
                    tokens.append(self.tokens_to_id[token])
                    text = text[len(token):]
                    break

        tokens.append(self.tokens_to_id["<EOS>"])

        return tokens
    

    @classmethod
    def detokenize(self, tokens: List[int]) -> str:
        # cutoff at first <EOS>
        eos_index = self.tokens_to_id["<EOS>"]
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
            if food_tags_token_id in list_of_tokens and non_food_tags_token_id in list_of_tokens:
                valid_tokens = ["<EOS>"]
            elif food_tags_token_id in list_of_tokens:
                valid_tokens = ["NON_FOOD_TAGS", "<EOS>"]
            else:
                raise ValueError(f"Invalid tokens: {list_of_tokens}")
        elif last_token == ",":
            # case 1: food
            if food_tags_token_id in list_of_tokens:
                valid_tokens = ["}"] + cls.food_tag_tokens
            # case 2: non-food
            elif non_food_tags_token_id in list_of_tokens:
                valid_tokens = ["}"] + cls.non_food_tag_tokens
            else:
                raise ValueError(f"Invalid tokens: {list_of_tokens}")
        elif last_token in cls.food_tag_tokens:
            valid_tokens = ["}", ","]
        elif last_token in cls.non_food_tag_tokens:
            valid_tokens = ["}", ","]
        elif last_token == "<EOS>":
            valid_tokens = []
        else:
            raise ValueError(f"Invalid tokens: {cls.detokenize(list_of_tokens)}")

        return [cls.tokens_to_id[token] for token in valid_tokens]
    
    @classmethod
    def get_possible_next_tokens_mask(cls, list_of_tokens: List[int]) -> torch.Tensor:
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
                return False
            
        return True    



def main():
    # Generate examples
    print("Generating training examples from reviews...")
    examples = generate_training_examples_from_reviews(1000000)

    t_start = time.time()
    for example in examples:
        tokenized_example = YelpMultiTagsCFGTokenizer.tokenize(example["output"])
        detokenized_example = YelpMultiTagsCFGTokenizer.detokenize(tokenized_example)
        if not YelpMultiTagsCFGTokenizer.verify_cfg_string(detokenized_example):
            print(f"Invalid example: {example}")
    t_end = time.time()
    print(f"Time taken: {t_end - t_start} seconds")

    print(f"Total examples: {len(examples)}")

    # Save examples
    save_examples(examples, "data/yelp_multi_tags/multi_tags_yelp.json")


if __name__ == "__main__":
    main() 

