import cv2
import mediapipe as mp
import pygame

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pygame.mixer.init()
#lista de sonidos, 
notes=[
    pygame.mixer.Sound("notes/#fa.wav"), #index left #0
    pygame.mixer.Sound("notes/la.wav"), #Middle left #1
    pygame.mixer.Sound("notes/re.wav"), #Ring left #2
    pygame.mixer.Sound("notes/#do.wav"), #Index left #3
    pygame.mixer.Sound("notes/#sol.wav"), #Middle right #4
    pygame.mixer.Sound("notes/si.wav"), #Ring right #5
    #pygame.mixer.Sound("notes/do.wav"), #Index right #6
]
def is_finger_down(landmarks, finger_tip, finger_mcp):
    """Returns True if the given finger is down, False otherwise."""
    return landmarks[finger_tip].y > landmarks[finger_mcp].y

cap = cv2.VideoCapture(0) # / If i pass a webcam
#cap = cv2.VideoCapture(video_path) / if i pass a file video

#Using Content-Manager to free resources
with mp_hands.Hands(
    # Confidence detecction
    min_detection_confidence=0.5,
    #Confidence tracking
    min_tracking_confidence=0.5,
    #number of hands
    max_num_hands=2) as hands:

    # the initial state is false
    finger_state = [False]*6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        #detect fingers in a frame and draw fingers and connections
        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                finger_tips = [8,12,16]
                finger_mep = [5,9,13]
                for i in range(3): #3 fingers
                    finger_index = i+h*3
                    if is_finger_down(hand_landmarks.landmark, finger_tips[i], finger_mep[i]):
                        if not finger_state[finger_index]:
                            notes[finger_index].play()
                            finger_state[finger_index] = True
                    else:
                        finger_state[finger_index] = False
        #Show image
        cv2.imshow('Hand Detecction', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

#Free and close resources
cap.release()
cv2.destroyAllWindows()