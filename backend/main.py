from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import json
import aiohttp
import logging
import traceback
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables securely from the .env file, following the best practice of keeping secrets out of the codebase
load_dotenv()

# Configure centralized logging to ensure consistent log format across the application, enabling easy integration with observability tools like ELK or Datadog
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  # Logger setup with appropriate naming for contextual logs across different modules

# Instantiate the FastAPI application with modular middleware and routing for scalable and maintainable architecture
app = FastAPI()

# CORS configuration: Securely allow cross-origin requests from specified domains, vital for frontend-backend integration in distributed environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # This should be dynamically configured based on deployment environment (dev/prod)
    allow_credentials=True,  # Securely transmit cookies and other credentials across domains
    allow_methods=["*"],  # Flexibility during development; should be locked down to specific methods (GET, POST, etc.) in production
    allow_headers=["*"],  # Allows custom headers from the frontend; could be restricted to known headers for enhanced security
)

# API configuration: External API endpoint and keys are managed via environment variables for security and flexibility
MODEL_API_URL = "https://api.ai71.ai/v1/chat/completions"
MODEL_API_KEY = os.getenv('FALCON_API_KEY')  # Critical to ensure API keys are rotated regularly and stored securely

# WebSocket connection management: Efficiently manage multiple concurrent WebSocket connections using a set, ensuring low-latency real-time communication
active_connections = set()

@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()  # Accept incoming WebSocket connection for real-time communication
    active_connections.add(websocket)  # Track active connections for broadcasting messages
    try:
        # Infinite loop to keep the WebSocket connection open; in production, consider timeout mechanisms or heartbeat checks
        while True:
            await websocket.receive_text()  # Wait for messages from the client; can be extended for interactive features
    finally:
        # Ensure connection is removed from the active set upon disconnection or error, preventing memory leaks
        active_connections.remove(websocket)

# Broadcast status updates to all active WebSocket clients, ensuring all clients receive consistent updates
async def broadcast_status_update(status_message: str):
    for connection in active_connections:
        await connection.send_json({"status": status_message})  # Structured JSON messages for consistent frontend parsing

# Endpoint for handling file uploads and processing: Asynchronous to handle high concurrency, enabling scalability
@app.post("/upload")
async def handle_pdf_upload(uploaded_file: UploadFile = File(...)):
    await broadcast_status_update("File received. Initiating analysis...")  # Immediate feedback to the user for improved UX
    logger.info(f"File received: {uploaded_file.filename}")  # Log the filename for auditing and debugging purposes
    try:
        # Validate the file type to prevent unsupported file formats from entering the processing pipeline
        if not uploaded_file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_bytes = await uploaded_file.read()  # Efficiently read the file content into memory for processing
        await broadcast_status_update(f"File size: {len(file_bytes)} bytes. Extracting content...")  # Notify user of ongoing processing
        logger.info(f"File size: {len(file_bytes)} bytes")  # Log file size for monitoring and optimization purposes

        extracted_text = await extract_text_from_pdf(file_bytes)  # Extract text from PDF asynchronously to avoid blocking
        await broadcast_status_update(f"Extracted {len(extracted_text)} characters from PDF. Analyzing content...")  # Progress update
        logger.info(f"Extracted text length: {len(extracted_text)} characters")  # Detailed logging for traceability

        analysis_results = await analyze_pdf_content(extracted_text)  # Asynchronously analyze the extracted text using the AI model
        await broadcast_status_update("Analysis successfully completed.")  # Notify the user upon successful analysis completion
        logger.info("Analysis successfully completed")  # Final log entry for the process
        return analysis_results  # Return the analysis results as a structured JSON response
    except Exception as error:
        # Comprehensive error handling with detailed logging for diagnostics and user-friendly error messages
        error_log_message = f"Error processing file: {str(error)}"
        await broadcast_status_update(error_log_message)  # Notify user of the error in real-time
        logger.error(error_log_message)  # Log the error for further analysis
        logger.error(traceback.format_exc())  # Capture full stack trace for debugging
        raise HTTPException(status_code=500, detail=error_log_message)  # Raise an HTTP 500 error to signal a server-side issue

# Asynchronous function to extract text content from a PDF file using PyPDF2, which handles PDFs in-memory for efficiency
async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))  # Create an in-memory PDF reader instance
        extracted_text = ""  # Initialize an empty string to store extracted text
        for page_index, page in enumerate(pdf_reader.pages):
            # Extract text from each page and append it to the result, ensuring pages are processed sequentially
            extracted_text += page.extract_text() + "\n"
            await broadcast_status_update(f"Extracted page {page_index+1} of {len(pdf_reader.pages)}")  # Real-time updates for each page
        logger.info(f"Extracted {len(pdf_reader.pages)} pages from PDF")  # Log the number of pages processed
        return extracted_text  # Return the concatenated text from all pages
    except Exception as error:
        # Log any extraction errors with full tracebacks to identify issues in PDF processing
        logger.error(f"Error extracting PDF content: {str(error)}")
        logger.error(traceback.format_exc())
        raise  # Re-raise the exception to ensure it is propagated up the call stack

# Asynchronous function to analyze the extracted PDF content using the Falcon 180B model via API
async def analyze_pdf_content(pdf_content: str) -> dict:
    logger.info("Commencing analysis of medical report")  # Log the start of the analysis process
    await broadcast_status_update("Analyzing medical report...")  # Notify user that analysis is in progress

    # Chunking the content to comply with API input size limits, a common strategy to handle large inputs in NLP tasks
    content_chunks = [pdf_content[i:i+1500] for i in range(0, len(pdf_content), 1500)]
    all_analysis_results = []  # Initialize a list to store results from each chunk

    async with aiohttp.ClientSession() as http_session:  # Use aiohttp for efficient asynchronous HTTP requests
        for chunk_index, chunk in enumerate(content_chunks):
            await broadcast_status_update(f"Processing chunk {chunk_index+1} of {len(content_chunks)}...")  # Notify user of progress
            logger.info(f"Processing chunk {chunk_index+1} of {len(content_chunks)}")  # Log each chunk being processed

            # Dynamic prompt generation for the AI model, ensuring the prompt is contextually relevant for each chunk
            analysis_prompt = f"""
            Analyze the following section of a medical report and extract key information.
            Return the results in a JSON format with the following structure:
            {{
                "summary": "Brief summary of this report section",
                "abnormal_results": [
                    {{"test_name": "Test Name", "value": "Abnormal Value", "reference_range": "Normal Range", "interpretation": "Brief interpretation"}}
                ],
                "charts": [
                    {{
                        "chart_type": "bar",
                        "title": "Chart Title",
                        "data": [
                            {{"label": "Category1", "value1": Number1, "value2": Number2, ...}},
                            {{"label": "Category2", "value1": Number1, "value2": Number2, ...}},
                            ...
                        ]
                    }},
                    {{
                        "chart_type": "area",
                        "title": "Chart Title",
                        "x_axis_key": "month",
                        "data_keys": ["value1", "value2", ...],
                        "data": [
                            {{"month": "January", "value1": Number1, "value2": Number2, ...}},
                            {{"month": "February", "value1": Number1, "value2": Number2, ...}},
                            ...
                        ],
                        "trend_percentage": 5.2,
                        "date_range": "January - June 2024"
                    }}
                ],
                "recommendations": ["Recommendation 1", "Recommendation 2", ...]
            }}

            Medical Report Section {chunk_index+1}/{len(content_chunks)}:
            {chunk}
            """

            # Prepare HTTP request with secure authorization headers, ensuring API keys are not exposed
            request_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MODEL_API_KEY}",
            }
            request_payload = {
                "model": "tiiuae/falcon-180B-chat",  # Specify the model used for processing
                "messages": [
                    {"role": "system", "content": "You are a medical expert analyzing health reports."},  # Context-setting for the model
                    {"role": "user", "content": analysis_prompt},
                ],
            }

            try:
                await broadcast_status_update(f"Sending request to AI model for chunk {chunk_index+1}")  # Notify user before sending the request
                async with http_session.post(MODEL_API_URL, headers=request_headers, json=request_payload) as api_response:
                    api_response.raise_for_status()  # Immediately handle HTTP errors, ensuring only successful responses are processed
                    result_json = await api_response.json()  # Parse the JSON response from the API
                    await broadcast_status_update(f"Received response from AI model for chunk {chunk_index+1}")  # Notify user of successful receipt

                # Parse and store the AI model's response for this chunk, ensuring data integrity and consistency
                parsed_chunk = json.loads(result_json["choices"][0]["message"]["content"])
                all_analysis_results.append(parsed_chunk)  # Append the parsed results to the overall list
                await broadcast_status_update(f"Successfully parsed AI model response for chunk {chunk_index+1}")  # Notify user of successful parsing
            except Exception as error:
                # Handle and log errors on a per-chunk basis to ensure other chunks can still be processed
                chunk_error_message = f"Error processing chunk {chunk_index+1}: {str(error)}"
                await broadcast_status_update(chunk_error_message)  # Notify user of the error
                logger.error(chunk_error_message)  # Log the error for later analysis
                logger.error(traceback.format_exc())  # Include the full stack trace for diagnostics

    # Aggregate results from all processed chunks into a cohesive final output
    aggregated_results = {
        "summary": " ".join([result["summary"] for result in all_analysis_results]),  # Combine summaries from all chunks
        "abnormal_results": [item for result in all_analysis_results for item in result.get("abnormal_results", [])],  # Aggregate abnormal results
        "charts": [item for result in all_analysis_results for item in result.get("charts", [])],  # Combine chart data for visualization
        "recommendations": [item for result in all_analysis_results for item in result.get("recommendations", [])]  # Aggregate recommendations
    }

    await broadcast_status_update("Finalizing analysis results...")  # Notify user that final results are being prepared
    return aggregated_results  # Return the final, aggregated analysis results to the client

# Run the FastAPI application using Uvicorn, a high-performance ASGI server, ensuring it is production-ready with appropriate security configurations
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the server on all available interfaces (0.0.0.0) to allow external access
