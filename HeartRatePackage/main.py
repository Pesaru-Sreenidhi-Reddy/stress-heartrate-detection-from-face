from collections import Counter
import cv2
import tensorflow as tf
import numpy as np
import dlib
class DetectFace(object):
    def __init__(self):
        self.kerasmodel = tf.keras.models.load_model('saved_model\\my_model.keras')
        self.stressdict={0:'Not Stressed' ,  1:'Stressed'}
        self.detector = dlib.get_frontal_face_detector()

    def _predict(self, image):
        prediction = self.kerasmodel.predict(np.expand_dims(image, axis=0))
        maxindex = int(prediction[0][0])
        return self.stressdict[maxindex]
    
    def predict_stress(self, frame):
     f_h, f_w, c = frame.shape
     faces = self.detector(frame)
     stressarr=[]
     try:
       if faces:
            for face in faces:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                roi=frame[y1:y2, x1:x2]
                resized_frame = cv2.resize(roi, (200, 200))
                stress = self._predict(resized_frame)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=[255, 0, 137], thickness=2)
                frame = cv2.rectangle(frame, (x1, y1 - int(f_h*0.03125)), (x1 + int(f_w*0.125), y1), color=[255, 0, 137], thickness=-1)
                frame = cv2.putText(frame, text=stress, org=(x1 + 5, y1 - 3), fontFace=cv2.FONT_HERSHEY_PLAIN, color=[255, 255, 255], fontScale=1, thickness=1)
                stressarr.append(stress)
                stresscntr=Counter(stressarr)
            return frame, stresscntr.most_common()[0][0],resized_frame
       else:
                stress = 'No Face'
                resized_frame =cv2.resize(frame, (200, 200))
                return frame, stress,resized_frame
     except Exception as e:
         print("Hold the camera properly")


         
