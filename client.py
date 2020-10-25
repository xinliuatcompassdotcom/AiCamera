from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:5000'


request_room_type_url = addr + '/api/get_room_type'
request_description_url = addr + '/api/get_description'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('test.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(request_room_type_url, data=img_encoded.tobytes(), headers=headers)
# decode response
print(response)
print(json.loads(response.text))

response = requests.post(request_description_url, data=img_encoded.tobytes(), headers=headers)
# decode response
print(response)
print(json.loads(response.text))