# myrepo
my repository

# Author- Seema
print("🤖 Chatbot ready! Type 'quit' to exit.\n")

chat_history_ids = None while True: user_input = input("You: ") if user_input.lower() in ["quit", "exit", "bye"]: print("Bot: Goodbye! 👋") break

# encode the new user input, add end of string token
new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

# append tokens to chat history (if any)
bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

# generate response
chat_history_ids = model.generate(
    bot_input_ids,
    max_length=1000,
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size=3,
    top_k=100,
    top_p=0.9,
    temperature=0.7
)

# decode and print
bot_reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
print(f"Bot: {bot_reply}")
