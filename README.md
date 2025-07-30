# CMMC Chatbot

A Flask-based chatbot application that answers questions about CMMC (Cybersecurity Maturity Model Certification) using document embeddings and semantic search.

## Features

- Question answering based on CMMC documents
- Document embedding using sentence-transformers
- Vector storage with ChromaDB
- RESTful API with Flask

## Deployment on Render.com

### Prerequisites

1. Create a Render.com account
2. Fork or clone this repository to your GitHub account

### Deployment Steps

1. In Render.com, go to Dashboard and click "New Web Service"
2. Connect your GitHub repository
3. Configure the service with the following settings:
   - Name: cmmc-chatbot (or your preferred name)
   - Environment: Python
   - Build Command: `chmod +x build.sh && ./build.sh`
   - Start Command: `python app.py`

4. Add the following environment variables if needed:
   - TOGETHER_API_KEY: Your Together API key
   - Any other environment variables required by your application

5. Click "Create Web Service"

### Troubleshooting

If you encounter issues with the tokenizers package during deployment:

1. The build.sh script installs Rust which is required for tokenizers
2. The requirements.txt file includes a specific version of tokenizers that should work with pre-built wheels
3. Check the build logs for any specific errors

## Local Development

### Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the embedding script: `python embed_documents.py`
4. Start the server: `python app.py`

### API Endpoints

- POST `/chat`: Send a question and get an answer
  - Request body: `{"question": "What is CMMC Level 1?"}`
  - Response: `{"answer": "..."}`

- GET `/health`: Health check endpoint
  - Response: `{"status": "healthy"}`