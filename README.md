# SMS Spam Detector

A binary text classifier that detects whether an SMS message is **spam** or **ham** (legitimate). Built from scratch using a single-neuron model with a sigmoid activation function trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). Served as a Flask web application.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Latest Release](https://img.shields.io/github/v/release/phantomsage219/sms-spam-detector?label=latest)

---

## How it works

1. Each SMS is cleaned and tokenized into words
2. A Bag of Words feature vector is built from a ~8,600 word vocabulary
3. A single-neuron model with sigmoid activation predicts the spam probability
4. A threshold of 0.5 classifies the message as SPAM or HAM

---

## Development

### Prerequisites

- Python 3.11
- Conda (recommended) or virtualenv

### Setup

```bash
git clone https://github.com/phantomsage219/sms-spam-detector.git
cd sms-spam-detector

# create and activate environment
conda create -n sms-spam python=3.11
conda activate sms-spam

pip install -r requirements.txt
```

### Train the model

Download [spam.csv](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it at the project root, then:

```bash
python train.py
```

This will save `model/weights.npy` and `model/vocab.npy` when training completes.

### Run the server

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Docker

### Pull and run from Docker Hub

```bash
# replace <version> with the tag you want — see available tags at the link below
docker pull phantomsage219/sms-spam-detector:<version>
docker run -p 5000:5000 phantomsage219/sms-spam-detector:<version>
```

Check available tags on [Docker Hub](https://hub.docker.com/r/phantomsage219/sms-spam-detector/tags).

Open [http://localhost:5000](http://localhost:5000)

### Build locally

```bash
docker build -t sms-spam-detector:dev .
docker run -p 5000:5000 sms-spam-detector:dev
```

---

## Project structure

```
sms-spam-detector/
├── app.py              # Flask server
├── model_utils.py      # Shared utilities (cleaning, vectorizing, model math)
├── train.py            # Training script
├── train.ipynb         # Training notebook
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── model/
│   ├── weights.npy     # Trained model weights
│   └── vocab.npy       # Vocabulary (required for inference)
└── templates/
    └── index.html      # Web UI
```

---

## CI/CD

Pushing a version tag to GitHub automatically builds and pushes the Docker image to Docker Hub via GitHub Actions.

```bash
git tag v<version>
git push origin v<version>
```

Image will be published as `phantomsage219/sms-spam-detector:v<version>`.

### Required GitHub secrets

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not your password) |

Add them under **Settings → Secrets and variables → Actions** in your GitHub repo.

---

## API

### `POST /predict`

**Request**
```json
{ "message": "Congratulations! You have won a free prize!" }
```

**Response**
```json
{ "label": "SPAM", "probability": 94.21 }
```
