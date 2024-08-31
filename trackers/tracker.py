from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
from utils import get_bbox_width,get_center_of_bbox

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
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        #Pickle file is created to reduce run time and avoid calling func multiple times
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects 
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            if stub_path is not None:
                with open(stub_path,'wb') as f:
                    pickle.dump(tracks,f)

        return tracks
    
    #Bounding Box gives us the width, height and center
    #With that, ellipse is drawn in that place on that frame with cv.ellipse
    #So frames are passed
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x2 = int(bbox[2])
        x_center,y_center=get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
    
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=3,
            lineType=cv2.LINE_4)
        
        cv2.putText(frame, f"ID: {track_id}", (x_center - 10, y_center - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    #Construct circle around the player
    #tracks is a dict which has three keys players,referees and ball
    #Values: [{track_id:{bbox:[]}},{},{}..] of all three classes
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            #Drawing ellipses for all frames of a player and then appending
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"],(19,202,235), track_id)
                
            output_video_frames.append(frame)
            
        return output_video_frames