from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack() #Tracking the detected objects
    
    #0.1 to allow more detections conf 0<=c<=1
    #0.8 means the model is 80% confident in it's prediction
    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size],conf = 0.1) 
            detections+=detection_batch
            break
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames) #detections-ultralytics.results object
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        for frame_num,detection in enumerate(detections):
            class_names = detection.names #0-ball,1-gk,2-player,3-referee
            cls_names_inv = {value:key for key,value in class_names.items()}
            #Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            #Convert gk to player using class_id array replace it with player id according to the  yaml file 
            #before gk:1 player:2 now gk:2 player:2
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_names_inv["player"]
            
            #Track Objects unique track ids are given for each objects
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision) #ByteTrack

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            #No need of tracking the ball since it's count is 1
            #Extract Bounding Box filtering out the necessary objects players,ref,gks
            for frame_detection in detections_with_tracks:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]