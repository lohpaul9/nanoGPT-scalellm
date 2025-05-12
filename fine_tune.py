from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

import os
import random
import sys
from datasets import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.testing import get_hf_prompt_completion_dataset

dataset = get_hf_prompt_completion_dataset(max_train_samples=1024, path_to_json_file="./data/yelp_multi_tags/with_generations_2.json", task = "multi_tags")

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    # attn_implementation="flash_attention_2"
    )

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

