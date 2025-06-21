import os
import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch
from pydantic import BaseModel, field_validator, ValidationInfo, Field
from typing import Literal, Optional
from enum import Enum
from openai import OpenAI
from pprint import pprint
import time
import hashlib


class stella_400m:
    def __init__(self):
        self.batch_size = 1024
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,
        ).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Enable inference optimizations
        if torch.cuda.is_available():
            self.model.half()  # Use FP16 for faster inference
            torch.backends.cudnn.benchmark = True

    # def embed(self, queries):
    #     query_prompt_name = "s2p_query"
    #     query_embeddings = self.model.encode(queries, prompt_name=query_prompt_name)
    #     return query_embeddings

    def embed(self, queries):
        while True:
            try:
                embeddings = self.model.encode(queries, batch_size=self.batch_size)
                # print("Run with batch size", self.batch_size)
                return embeddings
            except torch.cuda.OutOfMemoryError:
                self.batch_size //= 2
                print(f"Reducing batch size to {self.batch_size}")

    def embed_split(self, queries):
        # Split queries into smaller chunks
        embeddings = []
        for i in range(0, len(queries), 50):
            embeddings.extend(self.embed(queries[i : i + 50]))

        return embeddings


# Parameters
block_size = 8
output_dir = "data/yahooreviewcuisine"
os.makedirs(output_dir, exist_ok=True)

from typing import List, Dict, Optional
import re


# Define a list of cuisines for the restaurant model
CUISINES = [
    "japanese", "chinese", "indian", "american", "italian", 
    "french", "mexican"
]




class YelpEasyCFGTokenizer:
    """
    A simple tokenizer for the CFG vocabulary.
    """
    # Special tokens
    special_tokens = {
        "<PAD>": -1,
        "<EOS>": 1,
        "<BOS>": 2
    }
    
    # CFG tokens
    cfg_tokens = {
        "{": 3,
        "}": 4,
        "(": 5,
        ")": 6,
        "restaurant": 7,
        "not_restaurant": 8
    }
    
    # Cuisine tokens
    cuisine_tokens = {
        "japanese": 9,
        "chinese": 10,
        "indian": 11,
        "american": 12,
        "italian": 13,
        "french": 14,
        "mexican": 15
    }
    
    # Combine all tokens
    vocab = {**special_tokens, **cfg_tokens, **cuisine_tokens}
    reverse_vocab = {v: k for k, v in vocab.items()}

    print(reverse_vocab)
    
    # Vocabulary size
    vocab_size = len(vocab)
    
    # Special token IDs
    pad_token_id = special_tokens["<PAD>"]
    eos_token_id = special_tokens["<EOS>"]
    
    @classmethod
    def tokenize(cls, text: str) -> List[int]:
        """
        Tokenize a CFG string into token IDs.
        """
        # Add BOS token
        tokens = [cls.special_tokens["<BOS>"]]
        
        # Simple regex-based tokenization
        # This is a very simple approach that works for our specific CFG format
        pattern = r'(\{|restaurant|not_restaurant|\(|\)|\})|([a-z_]+)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Get the first non-empty group
            token = next(t for t in match if t)
            
            # Check if it's a cuisine
            if token in cls.cuisine_tokens:
                tokens.append(cls.cuisine_tokens[token])
            # Check if it's a CFG token
            elif token in cls.cfg_tokens:
                tokens.append(cls.cfg_tokens[token])
            # Unknown token
            else:
                print(f"Unknown token: {token}")
                raise ValueError(f"Unknown token: {token}")
        
        # Add EOS token
        tokens.append(cls.eos_token_id)
        
        return tokens
    
    @classmethod
    def detokenize(cls, token_ids: List[int]) -> str:
        """
        Convert token IDs back to a CFG string.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in cls.reverse_vocab:
                token = cls.reverse_vocab[token_id]
                # Skip special tokens
                if token not in ["<PAD>", "<EOS>", "<BOS>"]:
                    tokens.append(token)
                elif token == "<EOS>":
                    break
            else:
                print(f"Unknown token ID: {token_id}")
                raise ValueError(f"Unknown token ID: {token_id}")
        # Join tokens with appropriate spacing
        result = ""
        for i, token in enumerate(tokens):
            if token in ["{", "(", ")"]:
                result += token
            elif token == "}":
                result += token
            elif token == "restaurant" or token == "not_restaurant":
                if i > 0 and tokens[i-1] == "{":
                    result += token
                else:
                    result += token
            else:  # cuisine
                result += token
        
        return result
    
    @classmethod
    def get_token_id(cls, token: str) -> int:
        """
        Get the token ID for a token.
        """
        if token in cls.vocab:
            return cls.vocab[token]
        else:
            print(f"Unknown token: {token}")
            raise ValueError(f"Unknown token: {token}")
    
    @classmethod
    def get_token(cls, token_id: int) -> str:
        """
        Get the token for a token ID.
        """
        if token_id in cls.reverse_vocab:
            return cls.reverse_vocab[token_id]
        else:
            print(f"Unknown token ID: {token_id}")
            raise ValueError(f"Unknown token ID: {token_id}")
    
    @classmethod
    def get_vocab_size(cls) -> int:
        """
        Get the vocabulary size.
        """
        return cls.vocab_size 

    @classmethod
    def get_possible_next_tokens(cls, partial_text: str) -> List[str]:
        """
        Given a partial text, return the list of possible next tokens.
        This simulates what a parser would do when generating the CFG.
        """
        partial_text = partial_text.strip()
        
        # If the text is empty, we can start with either a restaurant or not_restaurant
        if not partial_text:
            return ["{", "{"]
        
        # If we have an opening brace, we can continue with either restaurant or not_restaurant
        if partial_text == "{":
            return ["restaurant", "not_restaurant"]
        
        # If we have "restaurant", we need to add an opening parenthesis
        if partial_text == "{restaurant":
            return ["("]
        
        # If we have "restaurant(", we can add any cuisine
        if partial_text == "{restaurant(":
            return CUISINES
        
        # If we have "restaurant(cuisine", we need to add a closing parenthesis
        if partial_text.startswith("{restaurant(") and ")" not in partial_text:
            return [")"]
        
        # If we have "restaurant(cuisine)", we need to add a closing brace
        if partial_text.startswith("{restaurant(") and partial_text.endswith(")"):
            return ["}"]
        
        # If we have "not_restaurant", we need to add a closing brace
        if partial_text == "{not_restaurant":
            return ["}"]
        
        # If we have a complete CFG, there are no more tokens
        if partial_text in ["{restaurant(cuisine)}", "{not_restaurant}"]:
            return []
        
        # For any other case, we can't determine the next token
        return []
    
    @classmethod
    def get_possible_next_tokens_indexes_mask(cls, list_of_tokens: List[int]) -> torch.Tensor:
        """
        Given a list of tokens, return the list of possible next tokens.
        """
        # list of tokens is (batch_size, block_size)
        last_token_id = list_of_tokens[-1]

        mask = torch.zeros(len(cls.vocab), dtype=torch.bool)
        
        # we re-write the logic
        last_token = cls.get_token(last_token_id)
        if last_token == "<BOS>":
            mask[cls.get_token_id("{")] = True
        elif last_token == "{":
            mask[cls.get_token_id("restaurant")] = True
            mask[cls.get_token_id("not_restaurant")] = True
        elif last_token == "restaurant":
            mask[cls.get_token_id("(")] = True
        elif last_token == "not_restaurant":
            mask[cls.get_token_id("}")] = True
        elif last_token == "(":
            for cuisine in CUISINES:
                mask[cls.get_token_id(cuisine)] = True
        elif last_token == ")":
            mask[cls.get_token_id("}")] = True
        elif last_token == "}":
            mask[cls.get_token_id("<EOS>")] = True
        elif last_token == "<EOS>":
            mask[cls.get_token_id("<EOS>")] = True
        elif last_token in CUISINES:
            mask[cls.get_token_id(")")] = True
        else:
            raise ValueError(f"Unknown token: {last_token}")
        return mask


def compare_with_llama_and_generations_2():
    with open(os.path.join(output_dir, "llama_3_3_generations.json"), "r") as f:
        data_llama = json.load(f)

    with open(os.path.join(output_dir, "with_generations_2.json"), "r") as f:
        data_gpt = json.load(f)

    print(f"Items of first item of data_gpt: {data_gpt[0].items()}")
    for i in range(len(data_gpt)):
        data_gpt[i]["generation"] = data_gpt[i]["generated"]
        del data_gpt[i]["generated"]

    # okay we need to re-order the things that   
    # hash the inputs of both 
    # sort the data_llama by the hash

    # note that there are some entries in data_llama that are not populated, so we only look at those 
    # that are populated with 'generation'

    for i in range(len(data_llama)):
        data_llama[i]["hash"] = hashlib.sha256(data_llama[i]["input"].encode()).hexdigest()
    for i in range(len(data_gpt)):
        data_gpt[i]["hash"] = hashlib.sha256(data_gpt[i]["input"].encode()).hexdigest()

    data_gpt_map = {item["hash"]: item for item in data_gpt}

    count_vs_gpt = 0
    for i in range(len(data_llama)):
        if "generation" not in data_llama[i]:
            continue
        if data_llama[i]["hash"] not in data_gpt_map:
            print(f"Hash not in data_gpt: {data_llama[i]['hash']}")
            continue
        if data_llama[i]["generation"] == data_gpt_map[data_llama[i]["hash"]]["generation"]:
            count_vs_gpt += 1
    
    print(f"Accuracy of founding: {count_vs_gpt / len(data_llama)}")

    # now re-map
    for i in range(len(data_llama)):
        data_gpt_map[data_llama[i]["hash"]]["generation_llama"] = data_llama[i]["predicted"]
        assert data_gpt_map[data_llama[i]["hash"]]["generation"] == data_llama[i]["generation"]

    # convert to list, but make sure the ones with generation_llama are at the start
    data_gpt_list = []
    for i in range(len(data_gpt)):
        if "generation_llama" in data_gpt[i]:
            data_gpt_list.append(data_gpt[i])
        
    for i in range(len(data_gpt)):
        if "generation_llama" not in data_gpt[i]:
            data_gpt_list.append(data_gpt[i])

    # let's make sure this is correct
    last_index_with_generation_llama = None
    for i in range(len(data_gpt_list)):
        if "generation_llama" not in data_gpt_list[i]:
            last_index_with_generation_llama = i
            break
    print(f"Last index with generation llama: {last_index_with_generation_llama}")
            
    
    # now let's check that the accuracy for generations and generations_llama are about the same
    count_correct_generations = 0
    count_correct_generations_llama = 0
    count_correct_generations_vs_llama = 0
    last_index_with_generation_llama = None
    for i in range(len(data_gpt_list)):
        if "generation_llama" not in data_gpt_list[i]:
            last_index_with_generation_llama = i
            break
        if data_gpt_list[i]["expected"] == data_gpt_list[i]["generation"]:
            count_correct_generations += 1
        if data_gpt_list[i]["expected"] == data_gpt_list[i]["generation_llama"]:
            count_correct_generations_llama += 1
        if data_gpt_list[i]["generation"] == data_gpt_list[i]["generation_llama"]:
            count_correct_generations_vs_llama += 1
    
    print(f"Accuracy of generations: {count_correct_generations / (last_index_with_generation_llama + 1)}")
    print(f"Accuracy of generations_llama: {count_correct_generations_llama / (last_index_with_generation_llama + 1)}")
    print(f"Accuracy of generations vs generations_llama: {count_correct_generations_vs_llama / (last_index_with_generation_llama + 1)}")

    # count the generation vs ground truth for the whole set
    count_correct_generations_vs_ground_truth = 0
    for i in range(len(data_gpt_list)):
        if data_gpt_list[i]["generation"] == data_gpt_list[i]["expected"]:
            count_correct_generations_vs_ground_truth += 1
    print(f"Accuracy of generations vs ground truth: {count_correct_generations_vs_ground_truth / len(data_gpt_list)}")

    with open(os.path.join(output_dir, "with_generations_with_llama.json"), "w") as f:
        json.dump(data_gpt_list, f)


def get_cost_per_example():
    # get cost per example of gpt 4o mini
    import tiktoken
    file_name = "data/yahooreviewcuisine/with_generations_2.json"

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


def prepare_data(keep_embedding_unchanged
                 : bool = False):
    
    # 1. Load reviews.json
    with open(os.path.join(output_dir, "with_generations_with_llama.json"), "r") as f:
        data = json.load(f)

    # let's first do checks
    num_llama_generations = 0
    num_gpt_generations = len(data)
    count_correct_generations_vs_llama = 0
    count_correct_generations_vs_llama_vs_ground_truth = 0
    count_correct_generations_vs_ground_truth = 0
    count_correct_generations = 0
    for i in range(len(data)):
        if "generation_llama" in data[i]:
            num_llama_generations += 1
            if data[i]["generation"] == data[i]["generation_llama"]:
                count_correct_generations_vs_llama += 1
            if data[i]["generation_llama"] == data[i]["expected"]:
                count_correct_generations_vs_llama_vs_ground_truth += 1
        if data[i]["generation"] == data[i]["expected"]:
            count_correct_generations_vs_ground_truth += 1
        if data[i]["generation"] == data[i]["generation"]:
            count_correct_generations += 1
    
    print(f"Accuracy of generations vs llama: {count_correct_generations_vs_llama / num_llama_generations}")
    print(f"Accuracy of generations vs llama vs ground truth: {count_correct_generations_vs_llama_vs_ground_truth / num_llama_generations}")
    print(f"Accuracy of generations vs ground truth: {count_correct_generations_vs_ground_truth / num_gpt_generations}")
    print(f"Accuracy of generations: {count_correct_generations / num_gpt_generations}")
    print(f"Number of llama generations: {num_llama_generations}")
    print(f"Number of gpt generations: {num_gpt_generations}")


    print(f"Length of data: {len(data)}")

    
    for i in range(len(data)):
        if "generation_llama" not in data[i]:
            cutoff_with_last_generated = i
            break
    # print(f"Cutoff with last generated: {cutoff_with_last_generated}")
    data_new_generated_only = data[:cutoff_with_last_generated]
    generation_outputs = [item["generation"] for item in data_new_generated_only]

    count_correct = 0
    for i in range(len(generation_outputs)):
        if generation_outputs[i] == data[i]["expected"]:
            count_correct += 1
    print(f"Accuracy: {count_correct / len(generation_outputs)}")

    outputs = [item["expected"] for item in data]
    if not keep_embedding_unchanged:
        inputs = [item["input"] for item in data]
        # 2. Embed inputs
        time_start = time.time()
        stella = stella_400m()  # Replace with your actual Stella model
        input_embeddings = stella.embed(inputs)
        time_end = time.time()
        print(f"Time taken to embed inputs: {time_end - time_start} seconds")

    # 3. Tokenize and pad outputs
    tokenizer = YelpEasyCFGTokenizer()
    output_tokens = []
    generated_tokens_list = []
    for i in range(len(outputs)):
        tokens = tokenizer.tokenize(outputs[i])
        # Pad or truncate to block_size
        if len(tokens) < block_size:
            tokens += [tokenizer.pad_token_id] * (block_size - len(tokens))
        elif len(tokens) > block_size:
            raise ValueError(f"Output token length is greater than block size: {len(tokens)} > {block_size}")
        output_tokens.append(tokens)
    
    for i in range(len(generation_outputs)):
        # get the generated tokens
        generated_tokens = tokenizer.tokenize(generation_outputs[i])
        # Pad or truncate to block_size
        if len(generated_tokens) < block_size:
            generated_tokens += [tokenizer.pad_token_id] * (block_size - len(generated_tokens))
        elif len(generated_tokens) > block_size:
            raise ValueError(f"Generated token length is greater than block size: {len(generated_tokens)} > {block_size}")
        generated_tokens_list.append(generated_tokens)

    
    output_tokens = np.array(output_tokens, dtype=np.int16)  # shape: (num_samples, block_size)
    generated_tokens_list = np.array(generated_tokens_list, dtype=np.int16)  # shape: (num_samples, block_size)

    # check that all the lists are the same length
    if not keep_embedding_unchanged:
        assert len(input_embeddings) == len(output_tokens)
        print(f"Input embeddings shape: {input_embeddings.shape}")
    print(f"Output tokens shape: {output_tokens.shape}")
    print(f"Generated tokens list shape: {generated_tokens_list.shape}")
    

    # 4. Save to files
    if not keep_embedding_unchanged:
        np.save(os.path.join(output_dir, "input_embeddings_3.npy"), input_embeddings)
    np.save(os.path.join(output_dir, "output_tokens_3.npy"), output_tokens)
    np.save(os.path.join(output_dir, "generated_tokens_list_3.npy"), generated_tokens_list)
    # Save tokenizer metadata for reference
    if not keep_embedding_unchanged:
        with open(os.path.join(output_dir, "tokenizer_meta.pkl"), "wb") as f:
            pickle.dump({
                "vocab_size": tokenizer.get_vocab_size(),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }, f)

    print(f"Saved {generated_tokens_list.shape[0]} samples to {output_dir}")

in_memory_input_embeddings = None
in_memory_ground_truth_tokens = None
in_memory_generated_tokens_list = None
rng = np.random.default_rng(2)

# 5. get_batch function for training
def get_batch_yahoo(batch_size: int,  train_samples_limit, split: str = "train", device: Optional[torch.device] = None):
    """
    Returns:
        x_embed: (batch_size, embedding_dim) - input embeddings
        x_tokens: (batch_size, block_size-1) - input tokens for next-token prediction
        y_tokens: (batch_size, block_size-1) - target tokens for next-token prediction
    """
    global in_memory_input_embeddings, in_memory_ground_truth_tokens, in_memory_generated_tokens_list

    if in_memory_input_embeddings is None:
        in_memory_input_embeddings = np.load(os.path.join(output_dir, "input_embeddings_3.npy"))
        print(f"Loaded input embeddings for {in_memory_input_embeddings.shape[0]} samples")
    if in_memory_ground_truth_tokens is None:
        in_memory_ground_truth_tokens = np.load(os.path.join(output_dir, "output_tokens_3.npy")) 
        print(f"Loaded output tokens for {in_memory_ground_truth_tokens.shape[0]} samples")
    if in_memory_generated_tokens_list is None:
        in_memory_generated_tokens_list = np.load(os.path.join(output_dir, "generated_tokens_list_3.npy"))
        print(f"Loaded generated tokens list for {in_memory_generated_tokens_list.shape[0]} samples")
    
    num_samples = in_memory_input_embeddings.shape[0]
    if split == "val":
        num_samples = in_memory_generated_tokens_list.shape[0]

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

    x_embed = in_memory_input_embeddings[idx]  # (batch_size, embedding_dim)
    if split == "train":
        tokens = in_memory_generated_tokens_list[idx]      # (batch_size, block_size)
    elif split == "val":
        tokens = in_memory_generated_tokens_list[idx]      # (batch_size, block_size)
    elif split == "test":
        tokens = in_memory_ground_truth_tokens[idx]      # (batch_size, block_size)

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

# Example usage
if __name__ == "__main__":
    # prepare_data()
    # x_embed, x_tokens, y_tokens = get_batch_yahoo(batch_size=64, split="val")
    # tokenizer = YelpEasyCFGTokenizer()
    # print("x_embed shape:", x_embed.shape)
    # print("x_tokens shape:", x_tokens.shape)
    # print("y_tokens shape:", y_tokens.shape)
    # print("x_tokens[0]:", x_tokens[0])
    # print("y_tokens[0]:", y_tokens[0])
    # print("x_tokens[0]:", tokenizer.detokenize(x_tokens[0].tolist()))
    # print("y_tokens[0]:", tokenizer.detokenize(y_tokens[0].tolist()))

    # print("x_tokens[1]:", x_tokens[1])
    # print("y_tokens[1]:", y_tokens[1])
    # print("x_tokens[1]:", tokenizer.detokenize(x_tokens[1].tolist()))
    # print("y_tokens[1]:", tokenizer.detokenize(y_tokens[1].tolist()))

    # print(get_chatgpt_completions())
    # prepare_data(keep_embedding_unchanged=False)
    get_cost_per_example()
    # compare_with_llama_and_generations_2()
    # data = json.load(open(os.path.join(output_dir, "with_generations_2.json"), "r"))

    # # count which have {restaurant(None)

    # count = 0
    # for item in data:
    #     if item["generated"] == "{restaurant(None)}":
    #         count += 1
    # print(f"Count: {count}")
    # print("--------------------------------")
