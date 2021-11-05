import requests
import json
import os
import wave
from xml.etree import ElementTree

class TextToSpeech(object):
    def __init__(self, text, voice):
        key = '6a8aaf87489d0e1174e3217e0b84111a'
        self.key = key
        self.text = text
        self.voice = voice

    def save_audio(self):
        url="https://kakaoi-newtone-openapi.kakao.com/v1/synthesize"
        headers = {
            "Content-Type" : "application/xml",
            "Authorization" : "KakaoAK " + self.key,
        }
        data = "<speak><voice name=\"" + self.voice + "\">" + self.text.encode('utf-8').decode('latin1') + "</voice></speak>"

        res = requests.post(url, headers=headers, data=data)

        return res.content
