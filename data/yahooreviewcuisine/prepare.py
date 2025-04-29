import os
import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch

class stella_400m:
    def __init__(self):
        self.batch_size = 65536
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

class CFGTokenizer:
    """
    A simple tokenizer for the CFG vocabulary.
    """
    def __init__(self):
        # Special tokens
        self.special_tokens = {
            "<PAD>": 0,
            "<EOS>": 1,
            "<BOS>": 2
        }
        
        # CFG tokens
        self.cfg_tokens = {
            "{": 3,
            "}": 4,
            "(": 5,
            ")": 6,
            "restaurant": 7,
            "not_restaurant": 8
        }
        
        # Cuisine tokens
        self.cuisine_tokens = {
            "japanese": 9,
            "chinese": 10,
            "indian": 11,
            "american": 12,
            "italian": 13,
            "french": 14,
            "mexican": 15
        }
        
        # Combine all tokens
        self.vocab = {**self.special_tokens, **self.cfg_tokens, **self.cuisine_tokens}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Vocabulary size
        self.vocab_size = len(self.vocab)
        
        # Special token IDs
        self.pad_token_id = self.special_tokens["<PAD>"]
        self.eos_token_id = self.special_tokens["<EOS>"]
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize a CFG string into token IDs.
        """
        # Add BOS token
        tokens = [self.special_tokens["<BOS>"]]
        
        # Simple regex-based tokenization
        # This is a very simple approach that works for our specific CFG format
        pattern = r'(\{|restaurant|not_restaurant|\(|\)|\})|([a-z_]+)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Get the first non-empty group
            token = next(t for t in match if t)
            
            # Check if it's a cuisine
            if token in self.cuisine_tokens:
                tokens.append(self.cuisine_tokens[token])
            # Check if it's a CFG token
            elif token in self.cfg_tokens:
                tokens.append(self.cfg_tokens[token])
            # Unknown token
            else:
                print(f"Unknown token: {token}")
                raise ValueError(f"Unknown token: {token}")
        
        # Add EOS token
        tokens.append(self.eos_token_id)
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to a CFG string.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
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
    
    def get_token_id(self, token: str) -> int:
        """
        Get the token ID for a token.
        """
        if token in self.vocab:
            return self.vocab[token]
        else:
            print(f"Unknown token: {token}")
            raise ValueError(f"Unknown token: {token}")
    
    def get_token(self, token_id: int) -> str:
        """
        Get the token for a token ID.
        """
        if token_id in self.reverse_vocab:
            return self.reverse_vocab[token_id]
        else:
            print(f"Unknown token ID: {token_id}")
            raise ValueError(f"Unknown token ID: {token_id}")
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.
        """
        return self.vocab_size 


def prepare_data():
    # 1. Load reviews.json
    with open(os.path.join(output_dir, "all_reviews.json"), "r") as f:
        data = json.load(f)

    inputs = [item["input"] for item in data]
    outputs = [item["output"] for item in data]

    # 2. Embed inputs
    stella = stella_400m()  # Replace with your actual Stella model
    input_embeddings = stella.embed(inputs)

    # 3. Tokenize and pad outputs
    tokenizer = CFGTokenizer()
    output_tokens = []
    for out in outputs:
        tokens = tokenizer.tokenize(out)
        # Pad or truncate to block_size
        if len(tokens) < block_size:
            tokens += [tokenizer.pad_token_id] * (block_size - len(tokens))
        elif len(tokens) > block_size:
            raise ValueError(f"Output token length is greater than block size: {len(tokens)} > {block_size}")

        output_tokens.append(tokens)
    output_tokens = np.array(output_tokens, dtype=np.uint16)  # shape: (num_samples, block_size)

    # 4. Save to files
    np.save(os.path.join(output_dir, "input_embeddings.npy"), input_embeddings)
    np.save(os.path.join(output_dir, "output_tokens.npy"), output_tokens)

    # Save tokenizer metadata for reference
    with open(os.path.join(output_dir, "tokenizer_meta.pkl"), "wb") as f:
        pickle.dump({
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }, f)

    print(f"Saved {len(inputs)} samples to {output_dir}")

in_memory_input_embeddings = None
in_memory_output_tokens = None

# 5. get_batch function for training
def get_batch_yahoo(batch_size, split="train", seed=None, device=None):
    """
    Returns:
        x_embed: (batch_size, embedding_dim) - input embeddings
        x_tokens: (batch_size, block_size-1) - input tokens for next-token prediction
        y_tokens: (batch_size, block_size-1) - target tokens for next-token prediction
    """
    # For demonstration, use all data for both train/val
    # make this global so that we don't need to load the data again
    global in_memory_input_embeddings, in_memory_output_tokens


    if in_memory_input_embeddings is None:
        in_memory_input_embeddings = np.load(os.path.join(output_dir, "input_embeddings.npy"))
    if in_memory_output_tokens is None:
        in_memory_output_tokens = np.load(os.path.join(output_dir, "output_tokens.npy"))

    num_samples = in_memory_input_embeddings.shape[0]

    rng = np.random.default_rng(seed)

    if split == "train":
        # go from 0 to 90% of the data  
        low = 0
        high = int(num_samples * 0.9)
        
    elif split == "val":
        # go from 90% to 100% of the data
        low = int(num_samples * 0.9)
        high = num_samples
    else:
        raise ValueError(f"Invalid split: {split}")
    
    idx = rng.choice(high - low, size=batch_size, replace=False)
    idx += low


    x_embed = in_memory_input_embeddings[idx]  # (batch_size, embedding_dim)
    tokens = in_memory_output_tokens[idx]      # (batch_size, block_size)

    # convert all to torch tensors 
    x_embed = torch.from_numpy(x_embed)
    # convert to int64
    tokens = tokens.astype(np.int64)
    tokens = torch.from_numpy(tokens)

    # For next-token prediction: x = tokens[:-1] and end padded with <PAD>
    x_tokens = tokens
    y_tokens = tokens[:, 1:]
    # pad with <PAD>
    y_tokens = torch.cat([y_tokens, torch.zeros(batch_size, 1, dtype=torch.int64)], dim=1)

    # put on device
    x_embed = x_embed.to(device)
    x_tokens = x_tokens.to(device)
    y_tokens = y_tokens.to(device)


    return x_embed, x_tokens, y_tokens

# Example usage
if __name__ == "__main__":
    x_embed, x_tokens, y_tokens = get_batch_yahoo(batch_size=64, split="val")
    tokenizer = CFGTokenizer()
    print("x_embed shape:", x_embed.shape)
    print("x_tokens shape:", x_tokens.shape)
    print("y_tokens shape:", y_tokens.shape)
    print("x_tokens[0]:", x_tokens[0])
    print("y_tokens[0]:", y_tokens[0])
    print("x_tokens[0]:", tokenizer.detokenize(x_tokens[0]))
    print("y_tokens[0]:", tokenizer.detokenize(y_tokens[0]))

    print("x_tokens[1]:", x_tokens[1])
    print("y_tokens[1]:", y_tokens[1])
    print("x_tokens[1]:", tokenizer.detokenize(x_tokens[1]))
    print("y_tokens[1]:", tokenizer.detokenize(y_tokens[1]))