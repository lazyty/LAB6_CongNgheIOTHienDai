import cv2
import base64
import time
import json
import sys
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import threading

# Kafka settings
KAFKA_BROKER = "192.168.220.195:9092"
KAFKA_TOPIC_SEND = "lab6-image"
KAFKA_TOPIC_RESPONSE = "lab6-response"

# Function to encode image to Base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

# Global variables
last_response_time = None
processing = False

# Kafka consumer function (runs in separate thread)
def kafka_consumer_thread():
    global processing, last_response_time
    
    try:
        # Create Kafka consumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC_RESPONSE,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='face_recognition_client',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        print(f"Consumer subscribed to {KAFKA_TOPIC_RESPONSE}")
        
        # Listen for messages
        for message in consumer:
            try:
                response = message.value
                end_time = time.time()
                
                print(f"Response from server: {response['name']}")
                print(f"Confidence: {response['probability']:.3f}")
                print(f"Server processing time: {response['processing_time']:.3f} seconds")
                
                if last_response_time is not None:
                    print(f"Total recognition time: {end_time - last_response_time:.3f} seconds")
                
                processing = False
                
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                processing = False
                
    except Exception as e:
        print(f"Error in consumer thread: {str(e)}")

def main():
    global processing, last_response_time
    
    try:
        # Create Kafka producer
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        print(f"Connected to Kafka broker at {KAFKA_BROKER}")
        
        # Start consumer thread
        consumer_thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
        consumer_thread.start()
        
        # Open webcam (camera 0)
        cap = cv2.VideoCapture(0)
        print("Press SPACE to capture and send image for recognition.")
        print("Press ESC to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read image from camera.")
                break
            
            # Display the image
            cv2.imshow("Camera (Press SPACE to capture)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC to exit
                break
            
            elif key == 32 and not processing:  # Space bar to capture image and send
                processing = True
                print("Capturing and sending image...")
                
                # Encode the image
                encoded_image_data = encode_image_to_base64(frame)
                
                # Get image dimensions
                h, w = frame.shape[:2]
                
                # Prepare the payload
                payload = {
                    "image": encoded_image_data,
                    "w": w,
                    "h": h
                }
                
                # Record start time
                last_response_time = time.time()
                
                # Send message to Kafka
                try:
                    producer.send(KAFKA_TOPIC_SEND, value=payload)
                    producer.flush()  # Ensure message is sent
                    print(f"Image published to {KAFKA_TOPIC_SEND}")
                except KafkaError as e:
                    print(f"Failed to send message: {str(e)}")
                    processing = False
        
        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()
        
        # Close producer
        producer.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()