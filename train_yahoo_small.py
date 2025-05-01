"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
from contextlib import nullcontext
from pprint import pprint
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_yahoo import GPTConfig, GPT


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O


def train_yahoo_small(
        experiment_name,
        block_size,
        n_layer,
        n_head,
        n_embd,
        dropout,
        max_iters,
        max_train_samples,
        learning_rate = 6e-4,
        wandb_log = False,
        only_eval_at_end = True,
):
    # name = 'yahoo_small'
    out_dir = 'out/' + experiment_name + '_' + str(time.time())
    eval_interval = 100
    if only_eval_at_end:
        eval_interval = max_iters
    log_interval = 5
    eval_iters = 10
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'gpt' + '_' + experiment_name + '_run_' + str(time.time())
    # data
    gradient_accumulation_steps = 1 # used to simulate larger batch sizes
    batch_size = max_train_sample if max_train_sample is not None else 1024 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 8
    # model
    # n_layer = 1
    # n_head = 2
    # n_embd = 128
    # dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    # learning_rate = 6e-4 # max learning rate
    # max_iters = 1000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 1000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    from data.yahooreviewcuisine.prepare import get_batch_yahoo, CFGTokenizer

    tokenizer = CFGTokenizer()

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    best_val_accuracy = 0

    # attempt to derive vocab_size from the dataset
    # meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = tokenizer.get_vocab_size()
    # if os.path.exists(meta_path):
    #     with open(meta_path, 'rb') as f:
    #         meta = pickle.load(f)
    #     meta_vocab_size = meta['vocab_size']
    #     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            raise ValueError("meta_vocab_size is None")
        model_args['vocab_size'] = meta_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        total_correct = 0
        total_samples = 0
        tokenizer = CFGTokenizer()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                if split == 'train':
                    X_EMBED, X_TOKENS, Y_TOKENS = get_batch_yahoo(batch_size, split=split, device=device, limit_to_num_samples=max_train_samples)
                else:
                    val_batch_size = 1024
                    X_EMBED, X_TOKENS, Y_TOKENS = get_batch_yahoo(val_batch_size, split=split, device=device)
                with ctx:
                    logits, loss = model(X_TOKENS, X_EMBED, Y_TOKENS)

                    if split == 'val':
                        for i in range(val_batch_size):
                            cfg_tokens = model.generate_cfg(X_EMBED[i], max_new_tokens=block_size, tokenizer=tokenizer)
                            cfg_string = tokenizer.detokenize(cfg_tokens.flatten().tolist())
                            target_string = tokenizer.detokenize(Y_TOKENS[i].flatten().tolist())
                            if target_string == cfg_string:
                                total_correct += 1
                            total_samples += 1

                losses[k] = loss.item()
            
            if split == 'val':
                out[f"val_accuracy"] = total_correct / total_samples
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # training loop
    X_EMBED, X_TOKENS, Y_TOKENS = get_batch_yahoo(batch_size, split="train", device=device, limit_to_num_samples=max_train_samples) # fetch the very first batch
    t0 = time.time()
    t_start_train = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if (not only_eval_at_end) and (iter_num % eval_interval == 0 and master_process):
            losses = estimate_loss()
            if losses['val_accuracy'] > best_val_accuracy:
                best_val_accuracy = losses['val_accuracy']
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val accuracy {losses['val_accuracy']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "val/accuracy": losses['val_accuracy'],
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'best_val_accuracy': best_val_accuracy,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X_TOKENS, X_EMBED, Y_TOKENS)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X_EMBED, X_TOKENS, Y_TOKENS = get_batch_yahoo(batch_size, split="train", device=device, limit_to_num_samples=max_train_samples)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
    t_end_train = time.time()
    print(f"Training time: {t_end_train - t_start_train:.2f} seconds")

    t_start_eval = time.time()
    # save the model at the end
    losses = estimate_loss()
    t_end_eval = time.time()
    best_val_loss = min(best_val_loss, losses['val'])
    best_val_accuracy = max(best_val_accuracy, losses['val_accuracy'])
    num_eval_samples = batch_size * eval_iters
    eval_time = t_end_eval - t_start_eval

    time_per_eval_inference = eval_time / num_eval_samples
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'training_time': t_end_train - t_start_train,
        'eval_time': eval_time,
        'num_eval_samples': num_eval_samples,
        'time_per_eval_inference': time_per_eval_inference,
    }
    torch.save(checkpoint, os.path.join(out_dir, 'final_model.pt'))

    model_size = model.get_num_params()

    experiment_info = {
        'experiment_name': experiment_name,
        'out_dir': out_dir,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'max_iters': max_iters,
        'max_train_samples': max_train_samples,
        'best_val_loss': best_val_loss.item(),
        'best_val_accuracy': best_val_accuracy,
        'num_eval_samples': num_eval_samples,
        'training_time': t_end_train - t_start_train,
        'eval_time': t_end_eval - t_start_eval,
        'model_size': model_size,
    }

    experiment_info = {**model_args, **experiment_info}

    # save a JSON of the dict of the config
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        # save both the model args and the run parameters

        pprint(experiment_info)
        
        json.dump(experiment_info, f)

    if ddp:
        destroy_process_group()

    return experiment_info

    


if __name__ == "__main__":

    # n_layers = [1, 2]
    # n_heads = [1, 2, 4]
    # n_embeddings= [32, 64, 128, 256]
    n_layers = [1]
    n_heads = [2]
    n_embeddings = [128]
    max_train_samples = [128, 256, 512, 768, 1024, 2048, 4096, 8192, None]
    # max_iters = [100, 200, 500, 1000]

    total_experiments_run = 0
    total_experiments = len(n_layers) * len(n_heads) * len(n_embeddings) * len(max_train_samples)

    # now store all the experiment info in a lookup table
    experiment_info_lookup = {}

    for n_layer in n_layers:
        for n_head in n_heads:
            for n_embedding in n_embeddings:
                for max_train_sample in max_train_samples:
                    exp_name = f"yahoo_small_n_layer_{n_layer}_n_head_{n_head}_n_embedding_{n_embedding}_max_train_sample_{max_train_sample}"
                    experiment_info = train_yahoo_small(
                        experiment_name=exp_name,
                        block_size=8,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_embd=n_embedding,
                        dropout=0.0,
                        max_iters=500,
                        max_train_samples=max_train_sample,
                        wandb_log=False,
                        only_eval_at_end=True,
                    )
                    total_experiments_run += 1
                    print(f"Total experiments run: {total_experiments_run}/{total_experiments}")
                    
                    experiment_info_lookup[exp_name] = experiment_info

    # save the experiment info lookup table to a JSON file
    with open('experiment_info_lookup.json', 'w') as f:
        json.dump(experiment_info_lookup, f)

    # train_yahoo_small(
    #     experiment_name="yahoo_small",
    #     block_size=8,
    #     n_layer=1,
    #     n_head=2,
    #     n_embd=128,
    #     dropout=0.0,
    #     max_iters=500,
    #     max_train_samples=4000,
    #     wandb_log=False,
    #     only_eval_at_end=True,
    # )
