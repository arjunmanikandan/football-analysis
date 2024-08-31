from utils import read_video,save_video
from trackers import Tracker
import os
import pickle

def main():
    video_frames = read_video("input_videos/ars_vs_wolves.mp4")

    # #Initialize Tracker
    tracker = Tracker("models/football_detection.pt")
    
    #Create Pickle file
    tracks = tracker.get_object_tracks(video_frames,
    read_from_stub=True,
    stub_path='stubs/track_stubs.pkl')
    
    #Draw Shapes around players
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    output_video_path = os.path.join("output_videos", "detected.avi")
    save_video(output_video_frames,output_video_path)

if __name__ == "__main__":
    main()