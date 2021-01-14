from bs4 import BeautifulSoup
from threading import Thread
import urllib
import uuid
import requests 
import json
import socket
import csv
import time

HOST = 'localhost'
PORT = 9999
LOG_DIR = './logs/'
CAPTHCA_DIR = './capcha_images/'

BASE_URL = "https://2captcha.com/captcha3.php?client={0}&"
GET_URL = BASE_URL + "action=get"
SEND_URL = BASE_URL + "action=send_captcha"
CANC_URL = BASE_URL + "action=close_captcha"

CLIENT_ID = 1
USERNAME = 2
PASSWORD = 3

class ConnectionThread:
    def run(self, client_id, username, password):
        header = { "Cookie": "client=" + client_id, "X-Requested-With": "XMLHttpRequest" }
        data = { "captcha_type": "1", "hostname": "2captcha.com" }

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))
            while True:
                res = requests.post(GET_URL.format(client_id), headers=header)
                if len(res.text) > 0:
                    parsed_html = BeautifulSoup(res.text, "html.parser")
                    
                    image = parsed_html.find("img")['src']
                    img_path = self.save_image(image)
                    pred, acc = self.predict(sock, img_path)

                    if acc > 2:
                        captcha_id = parsed_html.find(id="id")['value']
                        data = {"captcha_id": str(captcha_id), "code": pred}
                        requests.post(SEND_URL.format(client_id), data=data, headers=header)
                    else:
                        requests.post(CANC_URL.format(client_id), data=data, headers=header)
                time.sleep(1)

    def save_image(self, image):
        image_path = CAPTHCA_DIR + str(uuid.uuid4()) + ".jpg"
        response = urllib.request.urlopen(image)             
        with open(image_path, 'wb') as f:
            f.write(response.file.read())
        return image_path

    def predict(self, sock, img_path):
        sock.sendall(img_path.encode())
        data = json.loads(sock.recv(1024).decode())
        pred = data.get("pred")
        acc = data.get("acc")
        return (pred, acc)

threads = []
with open('users.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for row in reader:
        t = Thread(target=ConnectionThread().run, args=(row[CLIENT_ID], row[USERNAME], row[PASSWORD]))
        t.start()
        threads.append(t)

for t in threads:
    t.join()