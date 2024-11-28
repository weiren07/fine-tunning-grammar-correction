import torch
import logging
import os
from typing import Dict, Any

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LlamaFineTuner:
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        fine_tuned_model_path: str = "./fine-tuned-llama-3B"
    ):
        self.model_name = model_name
        self.fine_tuned_model_path = fine_tuned_model_path
        
        # Ensure output directory exists
        os.makedirs(fine_tuned_model_path, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = self._initialize_tokenizer()
        
    def _initialize_tokenizer(self) -> AutoTokenizer:
        """Initialize and configure tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def load_and_preprocess_dataset(self) -> Dict[str, Dataset]:
        """Load and preprocess the dataset."""
        try:
            logger.info("Loading dataset...")
            # You can replace this with your specific dataset
            dataset = load_dataset("grammarly/coedit")
            
            # Tokenization function
            def tokenize_function(examples):
                combined_text = [
                    f"### Input:\n{src}\n\n### Corrected:\n{tgt}\n"
                    for src, tgt in zip(examples["src"], examples["tgt"])
                ]
                return self.tokenizer(
                    combined_text,
                    padding="max_length",
                    truncation=True,
                    max_length=256,  # Increased for more context
                    return_tensors=None
                )
            
            # Tokenize dataset
            logger.info("Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=os.cpu_count() or 1,
                remove_columns=dataset["train"].column_names,
                desc="Processing dataset"
            )
            
            # Split dataset
            train_test_split = tokenized_dataset["train"].train_test_split(
                test_size=0.1, 
                seed=42
            )
            
            # Set format for PyTorch
            train_dataset = train_test_split["train"]
            eval_dataset = train_test_split["test"]
            train_dataset.set_format("torch")
            eval_dataset.set_format("torch")
            
            return {
                "train": train_dataset,
                "eval": eval_dataset
            }
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            raise

    def _create_model(self) -> torch.nn.Module:
        """Create and prepare model for training."""
        try:
            # Quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True
            )

            # Load base model
            logger.info("Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            # Prepare for k-bit training
            model = prepare_model_for_kbit_training(model)

            # LoRA configuration
            lora_config = LoraConfig(
                r=32,              # Increased rank
                lora_alpha=64,     # Increased alpha
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Enable input gradients
            model.enable_input_require_grads()
            
            return model
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            raise

    def train(self):
        """Main training method."""
        try:
            # Initialize wandb for experiment tracking (optional)
            wandb.init(
                project="llama-fine-tuning",
                name="optimized-qlora-finetuning",
                config={
                    "model": self.model_name,
                    "learning_rate": 5e-5,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 4
                }
            )

            # Load datasets
            datasets = self.load_and_preprocess_dataset()
            
            # Create model
            model = self._create_model()

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.fine_tuned_model_path,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
                learning_rate=5e-5,
                fp16=True,
                logging_steps=1000,
                evaluation_strategy="steps",
                eval_steps=5000,
                save_strategy="steps",
                save_steps=5000,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                logging_dir="./logs",
                report_to="wandb",  # Use wandb for tracking
                warmup_ratio=0.1,
                optim="adamw_torch"
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=datasets["train"],
                eval_dataset=datasets["eval"],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )

            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Train the model
            logger.info("Starting training...")
            trainer.train()

            # Save the model and tokenizer
            logger.info("Saving fine-tuned model...")
            trainer.save_model(self.fine_tuned_model_path)
            self.tokenizer.save_pretrained(self.fine_tuned_model_path)

            # Finish wandb run
            wandb.finish()

            logger.info("Training completed successfully!")

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA Out of Memory. Try reducing batch size or max_length.")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    """Main execution method."""
    try:
        fine_tuner = LlamaFineTuner()
        fine_tuner.train()
    except Exception as e:
        logger.error(f"Fine-tuning process failed: {e}")

if __name__ == "__main__":
    main()