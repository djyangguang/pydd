import socket
import urllib
import os
import json
import time
import web
import numpy as np
import uuid
from PIL import Image
import requests
ip_port = ('127.0.0.1',8889)    #端口号
BUFSIZE=1024
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  #买手机
s.bind(ip_port) #手机插卡
s.listen(5) #手机待机
#上传的url_suffix为image/data/3/1/1/8/e/9/b/a/7230df0d-75d3-4790-aa76-fbaa65cdf456.jpg对应uuid为：35b30be6-ffc2-4eb5-b2ee-1ff2b8fe4aea
#b'\x01\x00\x00\x00~\x00\x00\x00{"delay":0,"url":"E:/study_and_work/project_data/invoice/image/data/3/1/1/8/e/9/b/a/7230df0d-75d3-4790-aa76-fbaa65cdf456.jpg"}'
while True:                  #新增接收链接循环,可以不停的接电话
    conn,addr=s.accept()    #手机接电话
    print('接到来自%s的票据识别请求' %addr[0])
    while True:                 ##新增通信循环,可以不断的通信,收发消息
        msg=conn.recv(BUFSIZE)  #听消息,听话
        if len(msg) == 0:break  #如果不加,那么正在链接的客户端突然断开,recv便不再阻塞,死循环发生
        t = time.time()
        url = "http://127.0.0.1:8080/ocr"
        data = '{"billModel":"OCR","textAngle":"True","imgString":"data:image/jpeg;base64/9j/4AAQSkZJRgABAQEAYABgAADZ"}'
        # 字符串格式
        res = requests.post(url=url, data=data)
        print(res.text)
        mes =  '{"action_id":6,"msg_id":100, "id":3, "status":0, "url":"xxxxxxx"}'






        print(mes,type(mes))
        conn.send(mes.upper())  #发消息,说话

    conn.close()                #挂电话
s.close()               #手机关机