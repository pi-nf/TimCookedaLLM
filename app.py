import torch
import time
import tiktoken
import streamlit as st

from supplementary import GPTModel, generate_text_simple, text_to_token_ids, token_ids_to_text


def main(input_text, max_tokens):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	GPT_CONFIG_124M = {
		"vocab_size": 50257,   # Vocabulary size
		"context_length": 128, # Shortened context length (orig: 1024)
		"emb_dim": 768,        # Embedding dimension
		"n_heads": 12,         # Number of attention heads
		"n_layers": 12,        # Number of layers
		"drop_rate": 0.35,      # Dropout rate
		"qkv_bias": False      # Query-key-value bias
	}

	# torch.manual_seed(123)
	model = GPTModel(GPT_CONFIG_124M)
	model.to(device)

	weights = torch.load("weights.pth", weights_only=True, map_location=device)
	model.load_state_dict(weights)
	
	tokenizer = tiktoken.get_encoding("gpt2")

	input_ids = text_to_token_ids(input_text, tokenizer).to(device)

	# Generate text
	token_ids = generate_text_simple(
		model=model,
		idx=input_ids,
		max_new_tokens=max_tokens,
		context_size=128
	)


	return token_ids_to_text(token_ids, tokenizer), input_ids.size(1) + token_ids.size(1)


if __name__ == "__main__":
	st.title('TimCookedaLLM')

	input_text = st.text_area("Start context")
	max_tokens = st.number_input("Max tokens", min_value=1, max_value=100, value=20, step=1)

	if st.button('Generate'):
		start_time = time.time()
		
		output, token_usage = main(input_text, max_tokens)
		
		end_time = time.time()
		processing_time = end_time - start_time

		st.write("### Response:")
		st.write(output)
		st.markdown(f"<small><strong>Token Usage:</strong> {token_usage} tokens | <strong>Processing Time:</strong> {processing_time:.2f}s</small>", unsafe_allow_html=True)
