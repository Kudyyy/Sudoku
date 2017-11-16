import numpy as np
from keras.models import load_model

class Model:
    def load(self, model_path):
        self.model = load_model(model_path)
        
    def predict(self, img, debug=False):
        img_vec = Model.__to_vector(img)
        predictions = self.model.predict(np.array([img_vec]))
        if (debug):
            for i, p in enumerate(predictions[0]):
                print(str(i + 1) + ' --> ' + str(p))
        return np.argmax(predictions, axis=1)[0] + 1
    
    @staticmethod
    def __to_vector(image):
        normalize = np.vectorize(lambda x: 1 if x > 128 else 0)
        return normalize(np.array(image)).flatten()

