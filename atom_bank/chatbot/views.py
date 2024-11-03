# chatbot/views.py
import sys
from django.shortcuts import render
from django.http import JsonResponse
sys.path.append('../')
import autogen_hackathon
from estimation import predict

import json

def chat_view(request):
    if request.method == 'GET':
        return render(request, 'chatbot/chat.html')
    elif request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')
        back = autogen_hackathon.autogen_metrics(user_message)
        resp = back[1]
        dict_stats = back[0]
        # Placeholder response
        prediction_value = predict(user_message)
        response_message = ""
        for i in resp:
            response_message += f"• {i}<br>\n"  # Using HTML `<br>` tags for new lines in HTML

        
        response_message += f"\n Based on our AI prediction and ML analysis, the predicted value of {user_message} next year is {round(prediction_value,2)}. Predicted statistics:{dict_stats}"

        response_message = response_message.replace("*", "")

        return JsonResponse({'response': response_message})
