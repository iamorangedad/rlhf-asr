import os
import sys
import argparse
import functools
import logging
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from transformers import WhisperProcessor
from peft import (
    LoraConfig,
    get_peft_model,
    AdaLoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
)
from src.orpo_trainer import ORPOTrainer
from src.data_reader import CustomDataset
from src.log import log_creater, StreamToLogger
from src.utils import print_arguments, add_arguments
from src.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from src.model import CustomWhisperForConditionalGeneration

# ===============================================================================
# ================================= log  =================================
# ===============================================================================
logger = log_creater("logs")
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)
# ===============================================================================
# ================================= parameters  =================================
# ===============================================================================
print("=" * 90)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data", type=str, default="dataset/train.json", help="path of train data")
add_arg("test_data", type=str, default="dataset/test.json", help="path of test data")
add_arg("base_model", type=str, default="openai/whisper-tiny", help="Whisper model")
add_arg("output_dir", type=str, default="output/", help="path of saved model")
add_arg("warmup_steps", type=int, default=50, help="steps for training warmup")
add_arg("logging_steps", type=int, default=100, help="steps for logging")
add_arg("eval_steps", type=int, default=1000, help="steps for evaluation")
add_arg("save_steps", type=int, default=1000, help="steps for saving ")
add_arg("num_workers", type=int, default=8, help="threads of loading data")
add_arg("learning_rate", type=float, default=1e-3, help="learning rate")
add_arg("min_audio_len", type=float, default=0.5, help="the min length of audio file")
add_arg(
    "max_audio_len",
    type=float,
    default=30,
    help="the max length of audio file, which is 30s",
)
add_arg("use_adalora", type=bool, default=True, help="whether use AdaLora or Lora")
add_arg(
    "fp16", type=bool, default=True, help="whether use fp16 for model training or not"
)
add_arg("use_8bit", type=bool, default=False, help="whether use 8bit or not")
add_arg(
    "timestamps",
    type=bool,
    default=False,
    help="whethe use the timestamps data during training or not",
)
add_arg(
    "use_compile",
    type=bool,
    default=False,
    help="whethere use compiler of Pytorch2.0 or not",
)
add_arg(
    "local_files_only",
    type=bool,
    default=False,
    help="whether use local model file or not",
)
add_arg("num_train_epochs", type=int, default=3, help="the number of training epoch")
add_arg(
    "language",
    type=str,
    default="Chinese",
    help="setup the language",
)
add_arg(
    "task",
    type=str,
    default="transcribe",
    choices=["transcribe", "translate"],
    help="task name for training",
)
add_arg(
    "augment_config_path", type=str, default=None, help="the path for augment config"
)
add_arg(
    "resume_from_checkpoint",
    type=str,
    default=None,
    help="the checkpoing path for resume",
)
add_arg(
    "per_device_train_batch_size", type=int, default=8, help="batch size of training"
)
add_arg(
    "per_device_eval_batch_size", type=int, default=8, help="batch size of evaluation"
)
add_arg(
    "gradient_accumulation_steps",
    type=int,
    default=1,
    help="steps for gradient accumulation",
)
add_arg(
    "push_to_hub",
    type=bool,
    default=False,
    help="whether push the model to HuggingFace Hub or not",
)
add_arg("hub_model_id", type=str, default=None, help="model id of HuggingFace Hub")
add_arg("evaluation_strategy", type=str, default="epoch", help="")
add_arg("is_test", type=bool, default=False, help="")
add_arg("lr_scheduler_type", type=str, default="cosine", help="")
add_arg("enable_lora", type=bool, default=False, help="")
add_arg("optim", type=str, default="paged_adamw_32bit", help="")
add_arg(
    "alpha",
    type=float,
    default=1.0,
    help="Hyperparameter for weighting L_OR",
)
add_arg("disable_prompt_loss", type=bool, default=False, help="")
add_arg("seed", type=int, default=42, help="Random seed for reproducibility.")
args = parser.parse_args()
print_arguments(args)


def main():
    # ===============================================================================
    # ================================= data processor ==============================
    # ===============================================================================
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only,
    )
    processor.tokenizer.set_prefix_tokens(language=args.language)
    # ===============================================================================
    # ================================= data  =======================================
    # ===============================================================================
    print("=" * 90)
    train_dataset = CustomDataset(
        data_list_path=args.train_data,
        processor=processor,
        language=args.language,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
        augment_config_path=args.augment_config_path,
    )
    test_dataset = CustomDataset(
        data_list_path=args.test_data,
        processor=processor,
        language=args.language,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
    )
    print(f"train data len:{len(train_dataset)}\n test data len:{len(test_dataset)}")
    print("=" * 90)
    # ===============================================================================
    # ================================= data collator  ==============================
    # ===============================================================================
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # ===============================================================================
    # ================================= whisper model  ==============================
    # ===============================================================================
    print("Loading whisper model...")
    device_map = "auto"
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
        local_files_only=args.local_files_only,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print("Loading whisper model done.")
    # ===============================================================================
    # ================================= peft  =======================================
    # ===============================================================================
    print("=" * 90)
    model = prepare_model_for_kbit_training(model)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    if args.resume_from_checkpoint:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True
        )
    else:
        print(f"Adding LoRA modules...")
        target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        print(target_modules)
        if args.use_adalora:
            config = AdaLoraConfig(
                init_r=12,
                target_r=4,
                beta1=0.85,
                beta2=0.85,
                tinit=200,
                tfinal=1000,
                deltaT=10,
                lora_alpha=32,
                lora_dropout=0.1,
                orth_reg_weight=0.5,
                target_modules=target_modules,
            )
        else:
            config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
            )
        model = get_peft_model(model, config)
    model.print_trainable_parameters()
    print("=" * 90)
    # ===============================================================================
    # ================================= training parameter  =========================
    # ===============================================================================
    if args.base_model.endswith("/"):
        args.base_model = args.base_model[:-1]
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))
    training_args = TrainingArguments(
        output_dir=output_dir,  # The output directory
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=args.num_train_epochs,  # number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size for training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # batch size for evaluation
        evaluation_strategy=args.evaluation_strategy if args.is_test else "no",
        save_strategy=args.evaluation_strategy,
        optim=args.optim,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False if args.enable_lora else True
        },
        load_best_model_at_end=args.is_test,
        do_train=True,
        do_eval=args.is_test,
        lr_scheduler_type=args.lr_scheduler_type,
        remove_unused_columns=False,
        bf16=True,
        seed=args.seed,
    )
    # ===============================================================================
    # ================================= trainer  ====================================
    # ===============================================================================
    trainer = ORPOTrainer(
        model=model,
        alpha=args.alpha,
        pad=processor.tokenizer.pad_token_id,
        disable_prompt_loss=args.disable_prompt_loss,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    model.config.use_cache = False

    def load_from_checkpoint(resume_from_checkpoint, model=None):
        pass

    trainer._load_from_checkpoint = load_from_checkpoint
    print("define trainer done.")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("finish training.")
    print("=" * 90)
    # ===============================================================================
    # ================================= save model  =================================
    # ===============================================================================
    trainer.save_state()
    print("save state done.")
    model.config.use_cache = True
    model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
    print("save model done.")


if __name__ == "__main__":
    main()
