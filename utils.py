import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt

core = ov.Core()

model_face = core.read_model(model='models/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model = model_face, device_name='CPU')

input_layer_face =  compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

model_emo = core.read_model(model='models/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model = model_emo, device_name='CPU')

input_layer_emo =  compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

model_ag = core.read_model(model='models/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model = model_ag, device_name='CPU')

input_layer_ag =  compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output

def preprocess(frame, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    resized_frame = cv2.resize(frame, (input_width, input_height))
    transposed_frame = resized_frame.transpose(2, 0, 1)
    input_frame = np.expand_dims(transposed_frame, 0)
    return input_frame, resized_frame

def find_faceboxes(frame, results, confidence_threshold):
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    frame_h, frame_w, frame_channels = frame.shape
    face_boxes = face_boxes * np.array([frame_w, frame_h, frame_w, frame_h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes, scores

def draw_age_gender_emotion(face_boxes, frame):
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    show_frame = frame.copy()
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = frame[ymin:ymax, xmin:xmax]
        
        # Emotion
        input_frame, _ = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_frame])[output_layer_emo].squeeze()
        index = np.argmax(results_emo)
        emotion = EMOTION_NAMES[index]
        
        # Age and Gender
        input_frame_ag, _ = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_frame_ag])
        age, gender = results_ag[1], results_ag[0]
        age = int(np.squeeze(age) * 100)
        gender = np.squeeze(gender)
        gender_text = 'female' if gender[0] >= 0.65 else 'male' if gender[1] >= 0.55 else 'unknown'
        
        # Draw
        text = f"{gender_text} {age} {emotion}"
        cv2.putText(show_frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 2)
        cv2.rectangle(show_frame, (xmin, ymin), (xmax, ymax), (0, 200, 0), 3)
    return show_frame

def predict_image(image, confs_threshold):
    input_image, _ = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, scores = find_faceboxes(image, results, confs_threshold)
    visualise_image = draw_age_gender_emotion(face_boxes, image)
    
    return visualise_image