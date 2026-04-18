# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial

try:
    import wandb
    wandb.login()
except ImportError:
    pass

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)

IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        if step % self.args.logging_steps == 0:
            self.log({"loss": loss.item(), "ce_loss": outputs["ce_loss"], "distill_loss": outputs["distill_loss"], "ref_ce_loss": outputs["ref_ce_loss"],})
        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})


class ProsQAValCallback(TrainerCallback):
    """Runs greedy inference on ProsQA val and test sets at the end of each epoch."""

    def __init__(self, val_path, test_path, tokenizer, training_args, model_ref):
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.model_ref = model_ref
        self.num_latent = training_args.num_latent
        self.use_prj = training_args.use_prj
        self.remove_eos = training_args.remove_eos
        self.trainer = None  # set after trainer is created

        # Load val set
        with open(val_path) as f:
            val_data = json.load(f)
        self.val_questions = [d["question"].strip() for d in val_data]
        self.val_answers = [d["answer"].replace(",", "").strip() for d in val_data]

        # Load test set (optional)
        self.test_questions = None
        self.test_answers = None
        if test_path is not None:
            with open(test_path) as f:
                test_data = json.load(f)
            self.test_questions = [d["question"].strip() for d in test_data]
            self.test_answers = [d["answer"].replace(",", "").strip() for d in test_data]

    def _evaluate_split(self, model, questions, answers, device, autocast_dtype):
        """Run greedy inference and return accuracy."""
        tokenizer = self.tokenizer
        correct = 0
        total = len(questions)
        batch_size = 64

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype != torch.float32)):
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch_questions = questions[start:end]
                batch_answers = answers[start:end]
                bs = len(batch_questions)

                batch = tokenizer(batch_questions, return_tensors="pt", padding="longest")
                if self.remove_eos:
                    bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(bs, 1)
                else:
                    bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(bs, 2)
                batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1).to(device)
                batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1).to(device)

                # Encode question + bot
                outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, attention_mask=batch["attention_mask"])
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.use_prj:
                    latent_embd = model.prj(latent_embd)

                # Latent iterations
                for _ in range(self.num_latent):
                    outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                    if self.use_prj:
                        latent_embd = model.prj(latent_embd)

                # Insert eot
                if self.remove_eos:
                    eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device=device)).unsqueeze(0)
                else:
                    eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)).unsqueeze(0)
                eot_emb = eot_emb.expand(bs, -1, -1)

                # Greedy autoregressive decode
                output_emb = eot_emb
                finished = torch.zeros(bs, dtype=torch.bool, device=device)
                pred_tokens = [[] for _ in range(bs)]
                for _ in range(128):
                    out = model.codi(inputs_embeds=output_emb, use_cache=True, output_hidden_states=False, past_key_values=past_key_values)
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]
                    next_ids = torch.argmax(logits, dim=-1)
                    for b in range(bs):
                        if not finished[b]:
                            pred_tokens[b].append(next_ids[b].item())
                            if next_ids[b] == tokenizer.eos_token_id:
                                finished[b] = True
                    if finished.all():
                        break
                    output_emb = model.get_embd(model.codi, model.model_name)(next_ids).unsqueeze(1).to(device)

                # Score
                for b in range(bs):
                    decoded = tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
                    if "The answer is:" in decoded:
                        pred = decoded.split("The answer is:")[-1].strip().split("\n")[0].strip()
                        if not pred.endswith("."):
                            pred += "."
                    else:
                        pred = decoded.strip()
                    if pred == batch_answers[b]:
                        correct += 1

        return correct / total

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            model = self.model_ref
        model.eval()
        device = next(model.parameters()).device

        if args.bf16:
            autocast_dtype = torch.bfloat16
        elif args.fp16:
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.float32

        # Val
        val_acc = self._evaluate_split(model, self.val_questions, self.val_answers, device, autocast_dtype)
        print(f"\n[Epoch {state.epoch:.0f}] val accuracy: {100 * val_acc:.2f}% ({int(val_acc * len(self.val_questions))}/{len(self.val_questions)})")

        # Test
        test_acc = None
        if self.test_questions is not None:
            test_acc = self._evaluate_split(model, self.test_questions, self.test_answers, device, autocast_dtype)
            print(f"[Epoch {state.epoch:.0f}] test accuracy: {100 * test_acc:.2f}% ({int(test_acc * len(self.test_questions))}/{len(self.test_questions)})\n")

        # Log through the Trainer
        if self.trainer is not None and state.is_world_process_zero:
            metrics = {"val/accuracy": val_acc}
            if test_acc is not None:
                metrics["test/accuracy"] = test_acc
            self.trainer.log(metrics)

        model.train()

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##########################
    #       Peft Model       #
    ##########################
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


    model = CODI(model_args, training_args, lora_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        #answer_prompt = torch.tensor([464, 3280, 318, 25])
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            breakpoint()

    def preprocess(
        sources: Sequence[str], 
        targets: Sequence[str], 
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

        # add eos token to accomodate pretrained model's format
        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        if cot_id[0][0] == tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]

        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = -100
            ref_labels.append(z)
        
        # add eot to source
        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        # add eot and eos
        if training_args.remove_eos:
            answers_id = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]
        
        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)


    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, bot, eot):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")

            self.data_name = data_name
            questions, cots, answers = [], [], []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            token_nums = []
            for num_iter, example in enumerate(raw_data):
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                question = f"{example['question']}"
                if "icot" in self.data_name and "full" in self.data_name: # icot-full (GSM8k-Aug-NL)
                    # bad data
                    if example["answer"] is None: # or example["response"] is None:
                        continue
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot = f"{example['cot']}".split(". ")
                    if not (training_args.include_last_cot):
                        cot = cot[:-1]

                    answer = example['answer'].split(' ')[-1]
                    if not answer[0].isdigit():
                        continue
                    answer = f"The answer is: {answer}"
                    answer = answer.replace("####", "")
                    questions.append(question)
                    
                    if cot:
                        cot = ". ".join(cot)+".\n"
                    else:
                        cot = ""
                    cots.append(cot)
                    answers.append(answer)
                elif "icot" in self.data_name: # icot (GSM8k-Aug)
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot_list = []
                    cot = f"{example['cot']}".split(" ")
                    if not training_args.include_last_cot:
                        cot = cot[:-1]
                    
                    len_cot = len(cot) 
                    for i in range(training_args.num_latent):
                        cot_list.append(" ".join(cot[:max(0, len_cot-i)]))
                    answer = example['answer'].split(' ')[-1]
                    
                    # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
                    if not answer[0].isdigit():
                        continue

                    answer = f"The answer is: {answer}" 
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(" ".join(cot))
                    answers.append(answer)
                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "prontoqa" in data_args.data_name:
                    question = example['question'].strip() + '\n'
                    cot = '\n'.join(example['steps'][:-1]) + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                else:
                    raise NotImplementedError
            if training_args.exp_mode:
                questions = questions[:training_args.exp_data_num]
                cots = cots[:training_args.exp_data_num]
                answers = answers[:training_args.exp_data_num]
            
            print(f"{len(cots)} data in total...")
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess(questions, cots, answers, tokenizer, bot, eot)
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels= \
                tuple([instance[key] for instance in instances] for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels"))
        
            # pad left
            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)
            
            # pad
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX) 

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
          
            return dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
                ref_labels=ref_labels,
            )

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")
        if "icot" in data_args.data_name:
            if 'full' in data_args.data_name:
                dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
            else:
                dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "strategy" in data_args.data_name:
            dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "commonsense" in data_args.data_name:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "prontoqa" in data_args.data_name:
            if data_args.data_path is None:
                raise ValueError("--data_path must be specified for prontoqa/prosqa datasets")
            with open(data_args.data_path) as f:
                dataset = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Set up callbacks
    callbacks = []
    if "prontoqa" in data_args.data_name and data_args.val_data_path is not None:
        val_cb = ProsQAValCallback(
            val_path=data_args.val_data_path,
            test_path=data_args.test_data_path,  # None is fine, test eval will be skipped
            tokenizer=tokenizer,
            training_args=training_args,
            model_ref=model,
        )
        callbacks.append(val_cb)
        logging.warning(f"ProsQA val callback enabled: {data_args.val_data_path}")
        if data_args.test_data_path:
            logging.warning(f"ProsQA test callback enabled: {data_args.test_data_path}")
    elif "prontoqa" in data_args.data_name:
        logging.warning("No --val_data_path provided; skipping validation during training.")

    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)

    # Give the val callback a reference to the trainer for logging
    if "prontoqa" in data_args.data_name and data_args.val_data_path is not None:
        val_cb.trainer = trainer

    trainer.train()

    # to avoid the error of saving the model
    #if "llama" in model_args.model_name_or_path:
    #    trainer.model.codi.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.model.lm_head.weight.clone())
    #if "gpt2" in model_args.model_name_or_path:
    #    trainer.model.codi.transformer.wte.weight = torch.nn.Parameter(model.codi.lm_head.weight.clone())
    #if "qwen" in model_args.model_name_or_path.lower():
    #    trainer.model.codi.base_model.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.base_model.model.lm_head.weight.clone())

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()