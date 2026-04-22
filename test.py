import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    build_position_ids_from_mask,  # POS-FIX: shared helper
)

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def evaluation(model_args, data_args, training_args):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError
    
    model = CODI(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path 
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "gsm-hard" == data_args.data_name:
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset['train']
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_args.data_name:
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset['test']
        answer_name = "final_ans"
    elif "svamp" == data_args.data_name:
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "commonsense" == data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
    elif "prontoqa" in data_args.data_name or "prosqa" in data_args.data_name:
        if data_args.test_data_path is None:
            raise ValueError("--test_data_path must be specified for prontoqa/prosqa datasets")
        with open(data_args.test_data_path) as f:
            test_data_raw = json.load(f)
        question = [d["question"].strip() for d in test_data_raw]
        answer = [d["answer"].replace(",", "").strip() for d in test_data_raw]
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    is_prosqa = "prontoqa" in data_args.data_name or "prosqa" in data_args.data_name
    if not is_prosqa:
        question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
        answer = []

        for example in test_set:
            example = example[answer_name]
            if isinstance(example, bool):
                answer.append(example)
                continue
            if example in ["True", "False"]:
                if example == "True":
                    ans = True
                else:
                    ans = False
                answer.append(ans)
                continue
            if example in "ABCDE":
                answer.append(example)
                continue
            if "####" in example:
                ans = example.split('####')[-1]
            else:
                ans = example
            ans = ans.replace(',', '')
            try:
                ans = float(ans)
            except ValueError:
                ans = float("inf")
            answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    len_cot = []
    model.eval()
    
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            # POS-FIX (point 1): build explicit position_ids from the (left-padded) mask
            # so the first real token is always at position 0 and bot_id lands on a
            # deterministic position regardless of padding length.
            encoder_position_ids = build_position_ids_from_mask(batch["attention_mask"])

            # encode the question
            past_key_values = None
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=batch["attention_mask"],
                position_ids=encoder_position_ids,   # POS-FIX
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # POS-FIX (point 2): start the running mask and per-row position tracker
            # from the encoder state. These will be extended through the entire
            # latent loop and then through the decode loop (point 3).
            running_mask = batch["attention_mask"].clone()
            current_position = encoder_position_ids[:, -1]  # [B], position of bot_id

            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                # POS-FIX (point 2): advance position, extend mask by one.
                current_position = current_position + 1
                running_mask = torch.cat(
                    [running_mask, torch.ones((running_mask.size(0), 1),
                                              dtype=running_mask.dtype,
                                              device=running_mask.device)],
                    dim=1,
                )
                latent_position_ids = current_position.unsqueeze(1)  # [B, 1]

                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                    attention_mask=running_mask,           # POS-FIX
                    position_ids=latent_position_ids,      # POS-FIX
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            eot_len = output.size(1)  # 1 or 2 depending on remove_eos

            # POS-FIX (point 3): carry running_mask and position tracker into decode.
            # First, extend for the eot embedding(s) that we're about to feed.
            eot_position_ids = (current_position.unsqueeze(1)
                                + torch.arange(1, eot_len + 1, device=device).unsqueeze(0))  # [B, eot_len]
            running_mask = torch.cat(
                [running_mask,
                 torch.ones((running_mask.size(0), eot_len),
                            dtype=running_mask.dtype, device=running_mask.device)],
                dim=1,
            )
            current_position = current_position + eot_len
            current_step_position_ids = eot_position_ids

            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]
            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1

                out = model.codi(
                        inputs_embeds=output,
                        output_hidden_states=False,
                        attention_mask=running_mask,              # POS-FIX (point 3)
                        position_ids=current_step_position_ids,   # POS-FIX (point 3)
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values
                    )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # sampling
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

                # POS-FIX (point 3): advance mask + position for the next single-token step.
                current_position = current_position + 1
                current_step_position_ids = current_position.unsqueeze(1)  # [B, 1]
                running_mask = torch.cat(
                    [running_mask,
                     torch.ones((running_mask.size(0), 1),
                                dtype=running_mask.dtype, device=running_mask.device)],
                    dim=1,
                )

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred))
      
    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | {data_args.data_name} test accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")

    return 100*accuracy


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')

    # ProsQA: extract string answer after "The answer is:"
    if "prontoqa" in data_args.data_name or "prosqa" in data_args.data_name:
        if "The answer is:" in sentence:
            ans = sentence.split("The answer is:")[-1].strip()
            ans = ans.split("\n")[0].strip()
            if not ans.endswith("."):
                ans = ans + "."
            return ans
        else:
            return sentence.strip()

    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        if "commonsense" in data_args.data_name:
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                return "C" 
            return pred[0]
        elif "strategy" in data_args.data_name:
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        return float('inf')

    pred_answer = float(pred[-1])
    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list)/len(accu_list)}")