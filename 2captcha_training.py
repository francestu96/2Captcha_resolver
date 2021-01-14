from bs4 import BeautifulSoup
import requests 

username = "eito.maycen@usweek.net"
password = "panciotto8"
client_id = "10980748%3Ab9c10efc98394b893c91ee3e88543eb3"
  
GET_URL = "https://2captcha.com/cabinet/ajax_get_training"
SEND_URL = "https://2captcha.com/cabinet/ajax_send_training"
header = {"Cookie": "client=" + client_id}

codes = ["281737", "29274805", "94yssm", "15725897"]

for i in range(len(codes)):
    res = requests.post(GET_URL, headers=header)

    parsed_html = BeautifulSoup(res.text, "html.parser")
    captchaId = parsed_html.find(id="id")['value']

    data = {"captcha_id": str(captchaId), "code": codes[i]}
    requests.post(SEND_URL, data=data, headers=header)
