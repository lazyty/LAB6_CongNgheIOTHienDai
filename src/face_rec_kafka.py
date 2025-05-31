from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64
import json
import time
import requests
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# Face recognition parameters
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = './Models/20180402-114759.pb'

# ThingsBoard API settings
THINGSBOARD_URL = "http://192.168.220.190:9090/api/v1/GLXcK5tqsGt9GsjltDCY/telemetry"
ROOM_ID = "B3.08"

# Kafka settings
KAFKA_BROKER = "192.168.220.195:9092"
KAFKA_TOPIC_RECEIVE = "lab6-image"
KAFKA_TOPIC_RESPONSE = "lab6-response"

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

# Set up TensorFlow and FaceNet model
print('Setting up TensorFlow...')
tf_graph = tf.Graph()
with tf_graph.as_default():
    # Configure GPU if available
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    
    with sess.as_default():
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)
        
        # Get input and output tensors
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        
        # Create MTCNN for face detection
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
        
        # Define the function to send data to ThingsBoard
        def send_to_thingsboard(student_name):
            try:
                # Get current timestamp in the required format
                current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                
                # Prepare payload for ThingsBoard
                payload = {
                    "time": current_time,
                    "student": student_name,
                    "roomid": ROOM_ID
                }
                
                # Send POST request to ThingsBoard
                headers = {"Content-Type": "application/json"}
                response = requests.post(THINGSBOARD_URL, json=payload, headers=headers)
                
                if response.status_code == 200:
                    print(f"Successfully sent data to ThingsBoard: {student_name} at {current_time}")
                else:
                    print(f"Failed to send data to ThingsBoard. Status code: {response.status_code}")
                    
            except Exception as e:
                print(f"Error sending data to ThingsBoard: {str(e)}")
        
        # Define the function to process images and recognize faces
        def process_image(image_data, w, h):
            try:
                # Decode base64 image
                decoded_string = base64.b64decode(image_data)
                frame = np.frombuffer(decoded_string, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)
                
                # Detect faces
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                faces_found = bounding_boxes.shape[0]
                
                name = "Unknown"
                probability = 0.0
                
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        
                        # Use the whole frame since the provided code seems to do this
                        cropped = frame
                        # If you want to crop to the detected face, uncomment the following:
                        # cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        
                        if best_class_probabilities[0] > 0.5:
                            name = class_names[best_class_indices[0]]
                            probability = float(best_class_probabilities[0])
                            print(f"Name: {name}, Probability: {probability}")
                            
                            # Send data to ThingsBoard for successful recognition
                            send_to_thingsboard(name)
                        else:
                            print(f"Unknown face, Probability: {best_class_probabilities[0]}")
                
                return name, probability
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return "Error", 0.0

def main():
    try:
        # Create Kafka consumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC_RECEIVE,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='face_recognition_server',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        # Create Kafka producer for responses
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        print(f"Connected to Kafka broker at {KAFKA_BROKER}")
        print(f"Listening on topic: {KAFKA_TOPIC_RECEIVE}")
        print(f"Will respond on topic: {KAFKA_TOPIC_RESPONSE}")
        
        # Listen for messages
        for message in consumer:
            try:
                start_time = time.time()
                
                # Parse the received message
                payload = message.value
                image_data = payload.get("image", "")
                w = int(payload.get("w", 0))
                h = int(payload.get("h", 0))
                
                print("Received image recognition request")
                
                # Process the image
                name, probability = process_image(image_data, w, h)
                
                # Prepare response
                response = {
                    "name": name,
                    "probability": probability,
                    "processing_time": time.time() - start_time
                }
                
                # Send response via Kafka
                try:
                    producer.send(KAFKA_TOPIC_RESPONSE, value=response)
                    producer.flush()  # Ensure message is sent
                    print(f"Published response to {KAFKA_TOPIC_RESPONSE}")
                except KafkaError as e:
                    print(f"Failed to send response: {str(e)}")
                
            except Exception as e:
                print(f"Error processing message: {str(e)}")
    
    except Exception as e:
        print(f"Failed to connect to Kafka broker: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()