# Simple FastAPI Server

## Overview
A simple FastAPI server with basic endpoints and automatic API documentation.

## Project Structure
- `main.py` - Main FastAPI application with all endpoints

## Endpoints
- `GET /` - Welcome message
- `GET /health` - Health check endpoint
- `GET /items` - List all items
- `GET /items/{item_id}` - Get a specific item by ID
- `POST /items` - Create a new item

## Running the Server
The server runs on port 5000 with uvicorn.

## API Documentation
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Dependencies
- FastAPI
- Uvicorn
- Pydantic
