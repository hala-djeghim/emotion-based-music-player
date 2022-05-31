import torchvision.transforms as transforms
from tensorflow.keras.utils import img_to_array
import imutils
import cv2
import winsound as ws
import numpy as np
import time
from vgg_model import VGG
import torch

status = True


def happy_song():
    status = False
    ws.PlaySound( "musics\\happy.wav", ws.SND_ASYNC)
    status = True


def neutral_song():
    status = False
    ws.PlaySound( "musics\\neutral.wav", ws.SND_ASYNC)
    status = True


def sad_song():
    status = False
    ws.PlaySound( "musics\\sad.wav", ws.SND_ASYNC)
    status = True


def angry_song():
    status = False
    ws.PlaySound( "musics\\angry.wav", ws.SND_ASYNC)
    status = True


# def disgust_song():
#     status = False
#     ws.PlaySound( "musics\\disgust.wav", ws.SND_ASYNC)
#     status = True


def scared_song():
    status = False
    ws.PlaySound( "musics\\fear.wav", ws.SND_ASYNC)
    status = True


def surprised_song():
    status = False
    ws.PlaySound( "musics\\surprise.wav", ws.SND_ASYNC)
    status = True


detection_model_path = "haarcascade_frontalface_default.xml"
emotion_model_path = "model.pt"

DEVICE = torch.device("cpu")
if torch.cuda.is_available() :
    DEVICE = torch.device("cuda")
net =  VGG('VGG19').to(DEVICE)

face_detection = cv2.CascadeClassifier(detection_model_path)
net.load_state_dict(torch.load(emotion_model_path,  map_location=torch.device('cpu')))
EMOTIONS = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
    ]


class VideoCamera(object):
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.frame_count = 0

    def __del__(self):
        self.camera.release()

    def get_frame(self):
        frame = self.camera.read()[1]
        self.frame_count += 1

        frame = imutils.resize(frame, width=830, height=500)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.faces = face_detection.detectMultiScale(frame)

        frameClone = frame.copy()
        if len(self.faces) > 0:
            self.faces = sorted(self.faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = self.faces

            roi = frame[fY:fY + fH, fX:fX + fW]

            transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((48)),
                                    transforms.Normalize((0.5), (0.5))])

            roi = torch.unsqueeze(transform(roi),0)
            net.eval()
            global output
            with torch.no_grad():
                output = net(roi)
                _, predicted = torch.max(output.data, 1)
                output = output.cpu().detach().numpy()
                output = [element for sub in output for element in sub]
                predicted = predicted.tolist()[0]
                label = EMOTIONS[predicted]

        else:
            _, jpeg = cv2.imencode('.jpg', frameClone)
            return jpeg.tobytes()

        # for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, output)):
        #     if self.frame_count > 10:
        #         if (prob * 100) > 40:
        if (label == "happy") and status:
            print("Playing Happy Song...")
            happy_song()
            time.sleep(10)
            self.frame_count = 0

        elif (label == "neutral") and status:
            print("Playing Neutral Song...")
            neutral_song()
            time.sleep(10)
            self.frame_count = 0

        elif (label == "sad") and status:
            print("Playing Sad Song...")
            sad_song()
            time.sleep(10)
            self.frame_count = 0

        elif (label == "angry") and status:
            print("Playing angry Song...")
            angry_song()
            time.sleep(10)
            self.frame_count = 0

        # elif (label == "disgust") and status:
        #     print("Playing disgust Song...")
        #     disgust_song()
        #     time.sleep(3)
        #     frame_count = 0

        elif (label == "surprise") and status:
            print("Playing surprised Song...")
            surprised_song()
            time.sleep(10)
            self.frame_count = 0

        elif (label == "fear") and status:
            print("Playing scared Song...")
            scared_song()
            time.sleep(10)
            self.frame_count = 0

        cv2.putText(frameClone, label, (fX, fY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                        (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frameClone)
        return jpeg.tobytes()