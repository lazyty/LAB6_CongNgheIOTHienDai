# IoT Lab 6

## Initialize Kafka as Docker Container and create Kafka topics

### Step 1: Running Kafka Broker and Zookeeper container
```bash
cd kafka-compose
docker-compose up -d
```

### Step 2: Execute to Kafka Container
```bash
docker exec -it kafka bin/bash
cd /opt/kafka/bin
```

### Step 3: Create necessary topics
```bash
kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic lab6-image

kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic lab6-response
```

## Initialize ThingsBoard as Docker Container

### Step 1: Create necessary topics
```bash
cd thingsboard-compose
docker-compose up -d
```

### Step 2: Perform some necessary Device config and Dashboard config on ThingsBoard webUI

## Edge Server and Raspberry Pi 3

### Step 1: Install pip packages
```python
pip install -r requirements.txt
```

### Step 2: Data Augmentation
Each person in the group originally has only 5 face images. The `data_augmentation.py` script expands this dataset by applying random transformations such as flipping, brightness/contrast changes, rotation, distortion, noise, and blur. Each original image generates 10 augmented copies, resulting in 55 images per person. With 4 people, a total of 220 images are created for training the SVM model.

```bash
python3 data_augmentation.py
```

### Step 3: Preprocessing Images

This step detects and aligns faces from the augmented dataset using MTCNN and stores the processed thumbnails into the `Dataset/processed` folder. The aligned faces are resized to 160×160 and saved with optional margin and face bounding box data.

```bash
python3 src/align_dataset_mtcnn.py Dataset/raw Dataset/processed --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
```

### Step 4: Train the Model

The `classifier.py` script extracts facial embeddings using the pre-trained FaceNet model, then trains a Support Vector Machine (SVM) classifier to recognize identities. The trained classifier is saved as a `.pkl` file.

```bash
python3 src/classifier.py TRAIN Dataset/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 64
```

### Step 5: Run Kafka Server for Face Recognition

Run the Kafka-based server to handle image input and perform face recognition. This script loads a pre-trained FaceNet model and a custom SVM classifier, listens to image data from a Kafka topic, and returns recognition results through another Kafka topic. It also pushes successful recognitions to ThingsBoard for logging.

```bash
python3 src/face_rec_kafka.py
```

#### What it does:

* Loads the FaceNet model and pre-trained classifier from `Models/`.
* Listens for incoming base64-encoded images via the Kafka topic `lab6-image`.
* Detects and recognizes faces from the images using MTCNN and FaceNet embeddings.
* Sends recognition results (name, confidence, processing time) back through `lab6-response`.
* Logs recognized identities to ThingsBoard (via HTTP) with timestamp and room ID.

### Step 6: Send Image from Raspberry Pi 3

Run the client script on a Raspberry Pi 3 with a camera to automatically capture and send an image to the Kafka server. It also listens for the recognition response and measures total recognition time (from sending to response).

```bash
python3 client_kafka/client_pro.py
```

#### What it does:

* Captures images from the Pi Camera when SPACE is pressed.
* Encodes the image in base64 and sends it (along with width and height) via Kafka to `lab6-image`.
* Starts a background thread to listen for server response from `lab6-response`.
* Prints the recognition result, confidence score, server processing time, and full round-trip time (client send → server response).