# chatbot/views.py

from django.shortcuts import render
from django.http import JsonResponse
import json

def chat_view(request):
    if request.method == 'GET':
        return render(request, 'chatbot/chat.html')
    elif request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')

        # Placeholder response
        response_message = f"You said: {user_message}"

        return JsonResponse({'response': response_message})