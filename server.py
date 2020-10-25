
from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import fastai.vision as fv
import torch
import json 

from caption import caption_image_beam_search 


# Initialize the Flask application
app = Flask(__name__)
 

class AIModel:
    def __init__(self):
        # load room type model 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learn = fv.load_learner('./models', file='resnet-roomtype.pkl')
        self.learn.model = self.learn.model.module 

        # Load caption model
        checkpoint = torch.load('BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', map_location=str(device))
        self.decoder = checkpoint['decoder']
        self.decoder = self.decoder.to(device)
        self.decoder.eval()
        self.encoder = checkpoint['encoder']
        self.encoder = self.encoder.to(device)
        self.encoder.eval()

        # Load word map (word2ix)
        with open('WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 'r') as j:
            self.word_map = json.load(j)
        self.rev_word_map = {v: k for k, v in self.word_map.items()}  # ix2word

    def getRoomType(self, img_path): 
        img = fv.open_image(img_path)
        predict_class,predict_idx,predict_values = self.learn.predict(img)
        return predict_class, predict_idx, predict_values 


    def getDescription(self, img_path, beam_size=5): 
        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(self.encoder, self.decoder, img_path, self.word_map, beam_size)
        alphas = torch.FloatTensor(alphas)
        #Final predicted sentence
        words = [self.rev_word_map[ind] for ind in seq]
        return words 



aiModel = AIModel()


# route http posts to this method
@app.route('/api/get_room_type', methods=['POST'])
def get_room_type():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    TEMP_IMAGE_PATH = 'temp.jpg'
    cv2.imwrite(TEMP_IMAGE_PATH, img)

    predict_class, predict_idx, predict_values = aiModel.getRoomType(TEMP_IMAGE_PATH)
    confdience = predict_values[predict_idx.item()].item()
    
    # build a response dict to send back to client
    strClass = f'{predict_class}'
    response = {"class":strClass,"confdience":confdience}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# route http posts to this method
@app.route('/api/get_description', methods=['POST'])
def get_description():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    TEMP_IMAGE_PATH = 'temp.jpg'
    cv2.imwrite(TEMP_IMAGE_PATH, img)

    description = aiModel.getDescription(TEMP_IMAGE_PATH)
    description = description[1:-1]
    description = ' '.join(description)

    # build a response dict to send back to client
    response = {'message': description}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    
    # start flask app
    app.run(debug=True, host="0.0.0.0", port=5000)
