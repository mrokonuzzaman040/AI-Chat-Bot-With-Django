from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from django.http import JsonResponse

# Load the tokenizer and model
# Use a smaller model or quantized version if necessary
model_name = "mistralai/Mistral-1.3B"  # Smaller model for compatibility with your system
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Move model to CPU
model = model.to("cpu")

def get_response(request):
    if request.method == "GET":
        user_message = request.GET.get('message', '')

        if not user_message:
            return JsonResponse({'error': 'Message parameter is required'}, status=400)

        try:
            # Tokenize the user input
            inputs = tokenizer.encode(user_message, return_tensors="pt").to("cpu")

            # Generate a response
            outputs = model.generate(
                inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode the response
            bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return JsonResponse({'response': bot_response})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)
