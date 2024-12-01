from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from django.http import JsonResponse

# Replace with a public model name
model_name = "gpt2"  # Use a publicly available model
try:
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32  # Optimized for CPU
    )
    model = model.to("cpu")  # Move model to CPU
except Exception as e:
    raise RuntimeError(f"Failed to load the model or tokenizer: {e}")

def get_response(request):
    """
    Handles chatbot response generation.

    Args:
        request: Django HTTP request with a 'message' parameter.

    Returns:
        JsonResponse: Chatbot's response or error message.
    """
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
                max_length=150,  # Adjust for faster responses
                num_return_sequences=1,
                temperature=0.7,  # Controls randomness of responses
                top_p=0.95,  # Nucleus sampling
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode the response
            bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return JsonResponse({'response': bot_response})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)
