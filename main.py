import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(100, 200, 0))

save_screen = cv2.resize(cv2.imread("save_screen.png"), (1024, 756))

while True:
  with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    frame = cv2.resize(cap.read()[1], (1024, 756))

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
      cv2.destroyWindow("frame")
      while not results.multi_face_landmarks:
        results = face_mesh.process(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
        cv2.imshow("frame", save_screen)
        if cv2.waitKey(10) & 0xFF == 27: break
      cv2.destroyWindow("frame")

    annotated_image = frame.copy()
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)

    cv2.imshow("frame", annotated_image)

  if cv2.waitKey(10) & 0xFF == 27: break

cap.release()
cv2.destoryAllWindows()