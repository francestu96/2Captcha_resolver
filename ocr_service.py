import json
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path 
import socket
from tensorflow import keras
from tensorflow.keras import layers
import threading

class Ocr:
    def __init__(self, model_dir):
        self.img_width = 250
        self.img_height = 80
        self.prediction_model = keras.models.load_model(model_dir, compile=False)   
        characters = [x for x in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']
        self.char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(characters), num_oov_indices=0, mask_token=None)
        self.num_to_char = layers.experimental.preprocessing.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True) 

    def encode_single_sample(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return {"image": img, "label": label}

    def decode_single_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        res, acc = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
        output_text = tf.strings.reduce_join(self.num_to_char(res[0][:,: np.argmax(res[0] == -1)])).numpy().decode("utf-8")
        return (output_text, float(acc[0][0]))
    
    def predict(self, image_path):
        image = np.reshape(self.encode_single_sample(image_path, "unkown")["image"], (1, self.img_width, self.img_height, 1))
        pred = self.prediction_model.predict(image)
        return self.decode_single_prediction(pred)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <host> <port> <model_dir>")
        sys.exit(2)
    else:
        host = sys.argv[1]
        port = int(sys.argv[2])
        ocr =  Ocr(sys.argv[3])
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen(5)
            print(f"Server listening on {host}:{port}")
            while 1:
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    while True:
                        image = conn.recv(1024)
                        if not image: 
                            break
                        pred, acc = ocr.predict(image.decode())
                        data = json.dumps({"pred": pred, "acc": acc})
                        conn.sendall(data.encode())

