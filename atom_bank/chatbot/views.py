# chatbot/views.py
import sys
from django.shortcuts import render
from django.http import JsonResponse
sys.path.append('../')
import autogen_hackathon

import json

def chat_view(request):
    if request.method == 'GET':
        return render(request, 'chatbot/chat.html')
    elif request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')
        resp = autogen_hackathon.autogen_metrics(user_message)[1]
        # Placeholder response
        response_message = """"""
        for i in resp:
            response_message += f"""- {i}\n"""

        

        return JsonResponse({'response': response_message})
