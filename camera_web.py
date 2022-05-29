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
    ws.PlaySound( "musics\\scared.wav", ws.SND_ASYNC)
    status = True


def surprised_song():
    status = False
    ws.PlaySound( "musics\\surprised.wav", ws.SND_ASYNC)
    status = True


detection_model_path = "haarcascade_frontalface_default.xml"
emotion_model_path = "model.pt"

DEVICE = torch.device("cpu")
if torch.cuda.is_available() :
    DEVICE = torch.device("cuda")
net =  VGG('VGG19').to(DEVICE)

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = net.load_state_dict(torch.load(emotion_model_path,  map_location=torch.device('cpu')))
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.faces = face_detection.detectMultiScale(gray)

        frameClone = frame.copy()
        if len(self.faces) > 0:
            self.faces = sorted(self.faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = self.faces

            roi = gray[fY:fY + fH, fX:fX + fW]
            # roi = cv2.resize(roi, (64, 64))
            # roi = roi.astype("float") / 255.0
            transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((64)),
                                    transforms.Normalize((0.5), (0.5))])
            # Convert the image to PyTorch tensor
            roi = transform(roi)

            size = list(roi.size())
            size.insert(1,3)
            roi = torch.zeros(size, dtype=roi.dtype, device=roi.device)

            # roi = torch.unsqueeze(transform(roi),0)
            # roi = torch.reshape(roi,(1,3,48,48))

            net.eval()
            with torch.no_grad():
                output = net(roi)
                _, predicted = torch.max(output.data, 1)
                label = EMOTIONS[predicted]

            # global preds
            # preds = emotion_classifier.predict(roi)[0]
            # label = EMOTIONS[preds.argmax()]
        else:
            _, jpeg = cv2.imencode('.jpg', frameClone)
            return jpeg.tobytes()

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, output)):
            if self.frame_count > 10:
                if (prob * 100) > 40:
                    if (emotion == "happy") and status:
                        print("Playing Happy Song...")
                        happy_song()
                        time.sleep(3)
                        self.frame_count = 0

                    elif (emotion == "neutral") and status:
                        print("Playing Neutral Song...")
                        neutral_song()
                        time.sleep(3)
                        self.frame_count = 0

                    elif (emotion == "sad") and status:
                        print("Playing Sad Song...")
                        sad_song()
                        time.sleep(3)
                        frame_count = 0

                    elif (emotion == "angry") and status:
                        print("Playing angry Song...")
                        angry_song()
                        time.sleep(3)
                        frame_count = 0

                    # elif (emotion == "disgust") and status:
                    #     print("Playing disgust Song...")
                    #     disgust_song()
                    #     time.sleep(3)
                    #     frame_count = 0

                    elif (emotion == "surprise") and status:
                        print("Playing surprised Song...")
                        surprised_song()
                        time.sleep(3)
                        frame_count = 0

                    elif (emotion == "fear") and status:
                        print("Playing scared Song...")
                        scared_song()
                        time.sleep(3)
                        frame_count = 0

            cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                            (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frameClone)
        return jpeg.tobytes()