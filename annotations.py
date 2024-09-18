import cv2
import torch
from yolov5.models.experimental import attempt_load
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

classes = ['ball', 'goalkeeper', 'nonimpact', 'player']

# Video
cap = cv2.VideoCapture('dataset/08fd33_0.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    team1_players = []
    team2_players = []

    for det in results.pred[0]:
      class_idx = int(det[5])
      class_name = classes[class_idx]
      bbox = det[:4].int().cpu().numpy()
      if class_name == 'player':
          # Process player detection
          x_center = int((bbox[0] + bbox[2]) / 2)
          y_center = bbox[1]  # Use the top y-coordinate of the bounding box
          if det.tolist() in team1_players:  # Convert tensor to list before comparison
              marker_color = (255, 0, 0)  # Red for team 1
          elif det.tolist() in team2_players:  # Convert tensor to list before comparison
              marker_color = (0, 0, 255)  # Blue for team 2
          else:
              if x_center < width / 2:
                  marker_color = (255, 0, 0)  # Red for team 1
                  team1_players.append(det.tolist())
              else:
                  marker_color = (0, 0, 255)  # Blue for team 2
                  team2_players.append(det.tolist())

          cv2.drawMarker(frame, (x_center, y_center), marker_color, markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=10, thickness=2)

    ball_detections = [det for det in results.pred[0] if classes[int(det[5])] == 'ball']
    if ball_detections:
        highest_confidence_ball = max(ball_detections, key=lambda x: x[4])
        ball_bbox = highest_confidence_ball[:4].int().cpu().numpy()
        ball_x = (ball_bbox[0] + ball_bbox[2]) // 2  # Calculate the x-coordinate of the center of the ball
        line_x = ball_x + 10  # Adjust the value to move the line further in front of the ball
        # Draw the line
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

    if team1_players:
        team1_leftmost = min(team1_players, key=lambda x: x[0])
        team1_rightmost = max(team1_players, key=lambda x: x[0])
        # Get positions of leftmost and rightmost players
        team1_leftmost_x = int((team1_leftmost[0] + team1_leftmost[2]) / 2)
        team1_leftmost_x = team1_leftmost_x + 10
        team1_rightmost_x = int((team1_rightmost[0] + team1_rightmost[2]) / 2)
        team1_rightmost_x = team1_rightmost_x + 10
    else:
        team1_leftmost_x = 0
        team1_rightmost_x = 0

    if team2_players:
        team2_leftmost = min(team2_players, key=lambda x: x[0])
        team2_rightmost = max(team2_players, key=lambda x: x[0])
        # Get positions of leftmost and rightmost players
        team2_leftmost_x = int((team2_leftmost[0] + team2_leftmost[2]) / 2)
        team2_leftmost_x = team2_leftmost_x + 80
        team2_rightmost_x = int((team2_rightmost[0] + team2_rightmost[2]) / 2)
        team2_rightmost_x = team2_rightmost_x - 20
    else:
        # Set default values if no players are detected
        team2_leftmost_x = 0
        team2_rightmost_x = 0

    # Draw lines in front of leftmost and rightmost players of both teams
    cv2.line(frame, (team1_leftmost_x, 0), (team1_leftmost_x, frame.shape[0]), (255, 0, 0), 2)  # Blue line for team 1
    cv2.line(frame, (team1_rightmost_x, 0), (team1_rightmost_x, frame.shape[0]), (255, 0, 0), 2)  # Blue line for team 1
    cv2.line(frame, (team2_leftmost_x, 0), (team2_leftmost_x, frame.shape[0]), (0, 0, 255), 2)  # Red line for team 2
    cv2.line(frame, (team2_rightmost_x, 0), (team2_rightmost_x, frame.shape[0]), (0, 0, 255), 2)  # Red line for team 2

    # Write the processed frame to the output video
    out.write(frame)

# Release video capture and writer
cap.release()
out.release()
