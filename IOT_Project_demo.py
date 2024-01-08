from __future__ import unicode_literals
from typing import Text
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from picamera import PiCamera
from time import sleep
import datetime
import time
import datetime
import RPi.GPIO as GPIO
import pyimgur

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
RELAY_PIN = 12
PIR_IN_PIN = 11
PIR_OUT_PIN = 13
NOTPASS_PIN = 36
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(PIR_IN_PIN, GPIO.IN) #Read output from PIR motion sensor
GPIO.setup(PIR_OUT_PIN, GPIO.OUT)
GPIO.setup(NOTPASS_PIN, GPIO.OUT)

camera = PiCamera()
camera.hflip = 0
camera.vflip = 0
camera.resolution = (256,256)

#-----------------
#building 
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to specified size
    transforms.ToTensor(),  # Convert to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

class CNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(128, num_classes)  # Set num_classes based on your problem

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the path to your model file
model_path = './savemodel.pth'
model = CNNModel()  # Replace with the architecture of your model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

#-----------------
app = Flask(__name__)

# LINE 聊天機器人的基本資料
line_bot_api = LineBotApi('RD3mDhoB3i1l0u/zVC/2UM5YaQRGVfZf1597ff/wlVBRCnmZYfzJqbCtr/C6n7yskxoelHx/Vh4j50CEehmhNSk6DKaFkWwNwOyWf/ZJq8LtFgFD62j6ji/JKIRadOlTZdBu3PvjQXF+QZ3l0Xhy7gdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('0039418607e5d52d9a2a8ebf0dd59fe5')

#-----------------
def image_recognition():
    camera.zoom = (0.3,0,0.5,0.6)
    camera.capture('./image_phone.jpg')
    input_image_path = './image_phone.jpg'
    input_image = Image.open(input_image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    # Assuming output is a tensor, you can convert it to probabilities or predicted class
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    # Print the result
    #print(f"Predicted class: {predicted_class}, Probability: {probabilities[predicted_class]:.4f}")
    return predicted_class, probabilities[predicted_class]

#-----------------
def is_valid_24hour_time(input_str):
    try:
        # 嘗試將輸入字串轉換為時間格式
        time_obj = datetime.datetime.strptime(input_str, "%H:%M")

        # 確認時間在合法的範圍內（00:00 到 23:59）
        return 0 <= time_obj.hour < 24 and 0 <= time_obj.minute < 60

    except ValueError:
        # 轉換失敗或者時間不在合法範圍內
        return False

def is_integer(input_str):
    try:
        int_value = int(input_str)
        return True
    except ValueError:
        return False
#-----------------
def line_reply_message(event,reply_text):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

def capture_upload_image(event):
    event_user_id = event.source.user_id
    camera.zoom = (0,0,1,1)
    camera.capture('./image_phone.jpg')
    CLIENT_ID = "d42d9441d1ea8fd"
    PATH = "./image_phone.jpg" #A Filepath to an image on your computer"
    title = "Uploaded with PyImgur"
    im = pyimgur.Imgur(CLIENT_ID)
    uploaded_image = im.upload_image(PATH, title=title)
    line_bot_api.push_message(event_user_id,ImageSendMessage(original_content_url=uploaded_image.link, preview_image_url=uploaded_image.link),)    
    #print(uploaded_image.link)
    
    
#-----------------
def function_a(event,input_2):
    reply_text = ""
    check_num = 0
    alarm_time_t = datetime.datetime.strptime(input_2, "%H:%M")
#     alarm_time_2_hours_before = alarm_time - datetime.timedelta(hours=2)
    alarm_time_2_hours_before_t = alarm_time_t - datetime.timedelta(minutes=2)
    alarm_time = alarm_time_t.strftime("%H:%M")
    alarm_time_2_hours_before = alarm_time_2_hours_before_t.strftime("%H:%M")
    print(alarm_time_2_hours_before)
    print(alarm_time)
    print(datetime.datetime.now())

    print("reset, Relay is off")
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    
    while True:
        current_time = datetime.datetime.now().strftime("%H:%M")       
        
        if(alarm_time_2_hours_before <= current_time <= alarm_time):
            print("start, Relay is on")
            GPIO.output(RELAY_PIN, GPIO.LOW)
            
#         if(check_num%4==0):
#             GPIO.output(RELAY_PIN, GPIO.HIGH)
#             sleep(1)
#             GPIO.output(RELAY_PIN, GPIO.LOW)
#             sleep(15)
        
        predicted_class, probabilities = image_recognition()
        print(f"Predicted class: {predicted_class}, Probability: {probabilities:.4f}")
        if(probabilities>0.998 and predicted_class == 0):
            print("battery full, Relay is off")
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            break
        elif check_num >=24:
            print("runtime exceed, Relay is off")
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            break
        elif(current_time >= alarm_time):
            print("time exceed, Relay is off")
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            break
        else:
            print(f"Current time is {current_time}. Sleeping...")
            check_num+=1
            time.sleep(20)
#             time.sleep(300)  # 每隔5分鐘檢查一次

            
    reply_text = 'finsh a'
    line_reply_message(event,reply_text)

def function_b(event,input_2):
    event_user_id = event.source.user_id
#     sleep(30)
    sleep(5)
    cicle_time = int(input_2)
    print("time start")
    print("Relay is on")
    GPIO.output(RELAY_PIN, GPIO.LOW)
    GPIO.output(NOTPASS_PIN, 1)

  
    for i in range(cicle_time):
        GPIO.output(NOTPASS_PIN, 1)
#         for i in range(25*60):
        for i in range(1*60):
            i=GPIO.input(11)
            if i==0: #When output from motion sensor is LOW
                print("No intruders",i)
                GPIO.output(PIR_OUT_PIN, 0) #Turn OFF LED 
            elif i==1: #When output from motion sensor is HIGH
                print("Intruder detected",i)
                GPIO.output(PIR_OUT_PIN, 1) #Turn ON LED
                line_bot_api.push_message(event_user_id,TextSendMessage(text='Intruder detected'))
                capture_upload_image(event)
            time.sleep(1)
        GPIO.output(NOTPASS_PIN, 0)
#         time.sleep(5*60)
        time.sleep(1*60)

    print('finish b')
    GPIO.output(RELAY_PIN, GPIO.HIGH) 
    reply_text = 'finish b'
    line_reply_message(event,reply_text)

def function_c1(event,input_2):#at the time
    print("doing c1")
    event_user_id = event.source.user_id
    finish_time = input_2
    count_time = 0
    reply_text = ""
    GPIO.output(RELAY_PIN, GPIO.HIGH)
#     sleep(30)
    sleep(5)
    GPIO.output(RELAY_PIN, GPIO.LOW)

    while True:
        GPIO.output(NOTPASS_PIN, 1)
        current_time = datetime.datetime.now().strftime("%H:%M")
        
        i=GPIO.input(11)
        if i==0: #When output from motion sensor is LOW
            print("No intruders",i)
            GPIO.output(PIR_OUT_PIN, 0) #Turn OFF LED 
        elif i==1: #When output from motion sensor is HIGH
            print("Intruder detected",i)
            line_bot_api.push_message(event_user_id,TextSendMessage(text='Intruder detected'))
            capture_upload_image(event)
            GPIO.output(PIR_OUT_PIN, 1) #Turn ON LED
            
#         if(count_time%300 == 0 and count_time != 0):
#             if(count_time%1200 == 0):
#                 GPIO.output(RELAY_PIN, GPIO.HIGH)
#                 sleep(1)
#                 GPIO.output(RELAY_PIN, GPIO.LOW)
#                 sleep(15)
        
        if(count_time%20 == 0 and count_time != 0):
            predicted_class, probabilities = image_recognition()
            print(f"Predicted class: {predicted_class}, Probability: {probabilities:.4f}")
            if(probabilities>0.998 and predicted_class == 0):
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                print('relay off')
            elif count_time >=90*60:
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                print('relay off')        
        if current_time == finish_time:
            break
        
        time.sleep(1)
        count_time+=1
        
    GPIO.output(NOTPASS_PIN, 0)
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    reply_text = 'finish c1'
    line_reply_message(event,reply_text)
    
def function_c2(event,input_2):#remain time
    print('doing c2')
    event_user_id = event.source.user_id
    finish_time_remain = int(input_2)
    count_time = 0
    reply_text = ""
    GPIO.output(RELAY_PIN, GPIO.HIGH)
#     sleep(30)
    sleep(5)
    GPIO.output(RELAY_PIN, GPIO.LOW)

    for i in range(finish_time_remain*60):
        GPIO.output(NOTPASS_PIN, 1)
        i=GPIO.input(11)
        if i==0: #When output from motion sensor is LOW
            print("No intruders",i)
            GPIO.output(PIR_OUT_PIN, 0) #Turn OFF LED 
        elif i==1: #When output from motion sensor is HIGH
            print("Intruder detected",i)
            line_bot_api.push_message(event_user_id,TextSendMessage(text='Intruder detected'))
            capture_upload_image(event)
            GPIO.output(PIR_OUT_PIN, 1) #Turn ON LED
        
#         if(count_time%300 == 0 and count_time != 0):
        if(count_time%20 == 0 and count_time != 0):
#             if(count_time%1200 == 0):
#                 GPIO.output(RELAY_PIN, GPIO.HIGH)
#                 sleep(1)
#                 GPIO.output(RELAY_PIN, GPIO.LOW)
#                 sleep(15)
            predicted_class, probabilities = image_recognition()
            print(f"Predicted class: {predicted_class}, Probability: {probabilities:.4f}")
            if(probabilities>0.998 and predicted_class == 0):
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                print('relay off')
            elif count_time >=90*60:
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                print('relay off')
        
        time.sleep(1)
        count_time+=1
        
    GPIO.output(NOTPASS_PIN, 0)
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    reply_text = 'finish c2'
    line_reply_message(event,reply_text)
#---------------------------
def function_t1(event):
    event_user_id = event.source.user_id
    print(event_user_id)
    camera.capture('./image_phone.jpg')
    CLIENT_ID = "d42d9441d1ea8fd"
    PATH = "./image_phone.jpg" #A Filepath to an image on your computer"
    title = "Uploaded with PyImgur"
    im = pyimgur.Imgur(CLIENT_ID)
    uploaded_image = im.upload_image(PATH, title=title)
    line_bot_api.push_message(event_user_id,ImageSendMessage(original_content_url=uploaded_image.link, preview_image_url=uploaded_image.link),)    
    print(uploaded_image.link)
    line_reply_message(event,'finish_test')
def function_t2(event):
    predicted_class, probabilities = image_recognition()
    print(f"Predicted class: {predicted_class}, Probability: {probabilities:.4f}")
    line_reply_message(event,f"Predicted class: {predicted_class}, Probability: {probabilities:.4f}")
# 接收 LINE 的資訊
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# 接收訊息的路由
@handler.add(MessageEvent, message=TextMessage)
def reply_message(event):
    received_message = event.message.text.lower()  # 將收到的訊息轉換為小寫
    count_semicolon = received_message.count(';')
    
    if received_message == 'test1':
        function_t1(event)
    elif received_message == 'test2':
        function_t2(event)
    
    if(count_semicolon != 1):
        reply_text = '沒有這個指令，請重新輸入:\na. 睡眠時充電(eg.a;15:00) \nb. 番茄時鐘(eg.b;3) \nc. 手機保管箱(eg.c;15:00 or c;200)' 
    else:
        input_1, input_2= received_message.split(';')
        if input_1 == 'a':
            if is_valid_24hour_time(input_2):
                function_a(event,input_2)
            else:
                reply_text = '沒有這個指令，請重新輸入:\na. 睡眠時充電(eg.a;15:00) \nb. 番茄時鐘(eg.b;3) \nc. 手機保管箱(eg.c;15:00 or c;200)' 
        elif input_1 == 'b':
            if is_integer(input_2):
                function_b(event,input_2) 
            else:
                reply_text = '沒有這個指令，請重新輸入:\na. 睡眠時充電(eg.a;15:00) \nb. 番茄時鐘(eg.b;3) \nc. 手機保管箱(eg.c;15:00 or c;200)' 
        elif input_1 == 'c':        
            if is_valid_24hour_time(input_2):
                function_c1(event,input_2)
            elif is_integer(input_2):
                function_c2(event,input_2)
            else:
                reply_text = '沒有這個指令，請重新輸入:\na. 睡眠時充電(eg.a;15:00) \nb. 番茄時鐘(eg.b;3) \nc. 手機保管箱(eg.c;15:00 or c;200)' 
        else:
            print(reply_text)
    
    line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )
    
    

if __name__ == "__main__":
    app.run()
