import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='grammar_correction_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths to model directories
fine_tuned_model_path = "./fine-tuned-llama-3B(1)"
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load fine-tuned LoRA model
print("Loading fine-tuned model...")
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.eval()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on device: {device}")

# Define a stopping criteria to stop generation after the first corrected sentence
class SingleSentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.encode("\n")[0],
            self.tokenizer.encode("<|endofside|>")[0]
        ]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] in self.stop_tokens:
            return True
        return False

# Function to generate response
def generate_response(prompt, max_length=64, temperature=0, top_p=1.0, top_k=50):
    """
    Generates a single corrected sentence from the model based on the input prompt.
    
    Args:
        prompt (str): The input text.
        max_length (int): Maximum number of tokens in the generated sentence.
        temperature (float): Sampling temperature.
        top_p (float):  cumulative sampling for nucleus sampling
        top_k (int): fixed number of choices for the next words
    
    Returns:
        str: The generated corrected sentence.
    """
    logging.info(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    stopping_criteria = StoppingCriteriaList([SingleSentenceStoppingCriteria(tokenizer)])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,    # Set to 0 for deterministic behavior
            top_p=top_p,               # Disable nucleus sampling by setting top_p=1.0
            top_k=top_k,               # Reduce randomness
            do_sample=False,           # Disable sampling for deterministic results
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,    # Ensure only one sequence is returned
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"Raw Output: {decoded}")
    
    # Post-process the output to extract only the first corrected sentence
    corrected = extract_corrected_sentence(decoded)
    logging.info(f"Corrected Output: {corrected}")
    
    return corrected

# Function to extract the first corrected sentence using regex
def extract_corrected_sentence(decoded_text):
    """
    Extracts the first corrected sentence from the decoded text.
    
    Args:
        decoded_text (str): The raw output from the model.
    
    Returns:
        str: The first corrected sentence.
    """
    # Define patterns that indicate the start of the corrected sentence
    patterns = [
        r'Corrected:\s*(.*)',  # e.g., "Corrected: I love..."
        r'I love.*',            # Specific example based on user input
        r'.*',                  # Fallback to the entire line
    ]
    
    for pattern in patterns:
        match = re.search(pattern, decoded_text, re.IGNORECASE)
        if match:
            corrected = match.group(1).strip()
            # Remove any trailing unwanted tokens
            corrected = re.split(r'<\|endofside\|>|<\|endoftext\|>', corrected)[0].strip()
            # Capitalize the first letter and ensure proper punctuation
            if corrected and not corrected[0].isupper():
                corrected = corrected[0].upper() + corrected[1:]
            if corrected and corrected[-1] not in '.!?':
                corrected += '.'
            return corrected
    
    # Fallback if no pattern matches
    return decoded_text.strip().split('\n')[0].strip()

# Interactive testing
def interactive_test():
    """
    Allows real-time testing of the model with user input.
    """
    print("\n=== Interactive Grammar Correction ===")
    print("Enter a sentence to correct (type 'exit' to quit):\n")
    while True:
        try:
            user_input = input("Input: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive session.")
            break

        if user_input.lower() == "exit":
            print("Exiting interactive session.")
            break

        if not user_input.strip():
            print("Please enter a non-empty sentence.\n" + "-" * 50)
            continue

        # Create prompt for grammar correction
        prompt = f"Please correct the following sentence:\n\n{user_input}\n\nCorrected:"

        # Generate response
        response = generate_response(prompt)

        # Display the corrected sentence
        print(f"Corrected:\n{response}\n{'-' * 50}")

# Run the interactive test
if __name__ == "__main__":
    interactive_test()
