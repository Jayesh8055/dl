"""
FastAPI Main Application
Agentic Multimodal Input Processing System
"""

# Import FastAPI and related tools for building the web API
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Allows frontend to call this API
from fastapi.responses import JSONResponse  # Return JSON responses to frontend
from fastapi.staticfiles import StaticFiles  # Serve frontend HTML/CSS/JS files
import logging  # For tracking what the app does (debugging)
import os  # For file system operations
from typing import Optional  # For type hints

# Load environment variables from .env file (like API keys)
from dotenv import load_dotenv
load_dotenv()

# Import extractors - these get text from different file types
from extractors.image_extractor import extract_from_image  # Get text from images using OCR
from extractors.pdf_extractor import extract_from_pdf  # Get text from PDF files
from extractors.audio_extractor import extract_from_audio  # Convert speech to text
from extractors.youtube_extractor import (
    is_youtube_url,  # Check if text is a YouTube URL
    extract_youtube_url,  # Extract YouTube URL from text
    extract_youtube_transcript,  # Get transcript from YouTube video
)

# Import agents - these are the AI brains of the system
from agents.intent_classifier import IntentClassifier  # Understands what user wants to do
from agents.task_router import TaskRouter  # Routes to the right task handler

# Set up logging - this helps us see what's happening and debug issues
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create the FastAPI application - this is our web server
app = FastAPI(
    title="Agentic Multimodal App",
    description="AI-powered system for processing text, images, PDFs, and audio",
    version="1.0.0",
)

# Allow frontend (running on different port) to call this API
# Without this, browser will block requests due to CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (in production, specify exact domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Serve frontend files (HTML, CSS, JavaScript) if frontend folder exists
if os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Create instances of our AI agents - these will be used throughout the app
intent_classifier = IntentClassifier()  # Agent that figures out what user wants
task_router = TaskRouter()  # Agent that routes to the right task handler

# Store conversation context temporarily (for clarification flow)
# In production, use a database or Redis instead of in-memory dictionary
conversation_contexts = {}


@app.get("/")
async def root():
    """
    Root endpoint - shows API is running and lists available endpoints
    This is called when someone visits the base URL
    """
    return {
        "message": "Agentic Multimodal App API",
        "status": "running",
        "endpoints": {
            "process": "/process (POST)",
            "clarify": "/clarify (POST)",
            "health": "/health (GET)",
        },
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint - used to verify the API is working
    Monitoring tools can call this to check if server is alive
    """
    return {
        "status": "healthy",
        "services": {"fastapi": "running", "ollama": "check localhost:11434"},
    }


@app.post("/process")
async def process_input(
    text: Optional[str] = Form(None),  # User can send text directly
    file: Optional[UploadFile] = File(None),  # Or upload a file (image/PDF/audio)
    session_id: Optional[str] = Form("default"),  # Session ID for tracking conversations
):
    """
    Main endpoint - this is where all the magic happens!
    
    This function processes user input through 5 steps:
    1. Extract content from file or text
    2. Classify user intent (what do they want to do?)
    3. Check if we need clarification
    4. Execute the task (summarize, analyze sentiment, etc.)
    5. Return results to user

    Args:
        text: Optional text input (user can type text directly)
        file: Optional file upload (user can upload image, PDF, or audio file)
        session_id: Session identifier (for keeping track of conversations)

    Returns:
        JSON response with extraction results, intent, and task output
    """
    # Initialize variables to store data as we process
    logs = []  # List of messages to show user what's happening
    extracted_content = ""  # The text we extract from file or user input
    extraction_metadata = {}  # Info about extraction (file type, pages, etc.)

    try:
        # Log what we received (for debugging)
        logger.info(f"Processing request - Text: {bool(text)}, File: {bool(file)}")
        logs.append("Processing your request...")

        # ======================
        # STEP 1: Content Extraction
        # Extract text from whatever the user sent (file or text)
        # ======================

        if file:
            # User uploaded a file - read it and figure out what type it is
            file_bytes = await file.read()  # Read file into memory as bytes
            filename = file.filename.lower()  # Convert filename to lowercase for checking
            logs.append(f"Received file: {file.filename}")

            # Check if it's an image file (PNG, JPG, etc.)
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                logs.append("Extracting text from image using OCR...")
                # Use OCR (Optical Character Recognition) to read text from image
                result = extract_from_image(file_bytes)

                # If OCR succeeded, save the extracted text
                if result.get("success"):
                    extracted_content = result["text"]  # The text we found in the image
                    extraction_metadata = {
                        "type": "image",
                        "method": result["method"],  # How we extracted it (OCR)
                        "confidence": result["confidence"],  # How confident we are (0-100%)
                    }
                    logs.append(
                        f"✓ OCR completed (confidence: {result['confidence']}%)"
                    )
                else:
                    # OCR failed - tell user and stop
                    logs.append(f"✗ OCR failed: {result.get('error', 'Unknown error')}")
                    raise HTTPException(
                        status_code=400, detail="Image extraction failed"
                    )

            # Check if it's a PDF file
            elif filename.endswith(".pdf"):
                logs.append("Extracting text from PDF...")
                # Extract text from PDF (tries direct extraction, falls back to OCR if scanned)
                result = extract_from_pdf(file_bytes)

                # If PDF extraction succeeded, save the text
                if result.get("success"):
                    extracted_content = result["text"]  # Text from all pages
                    extraction_metadata = {
                        "type": "pdf",
                        "method": result["method"],  # How we extracted (text or OCR)
                        "pages": result["pages"],  # Number of pages
                    }
                    logs.append(
                        f"✓ PDF extraction completed ({result['pages']} pages, method: {result['method']})"
                    )
                else:
                    # PDF extraction failed
                    logs.append(
                        f"✗ PDF extraction failed: {result.get('error', 'Unknown error')}"
                    )
                    raise HTTPException(status_code=400, detail="PDF extraction failed")

            # Check if it's an audio file (MP3, WAV, etc.)
            elif filename.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
                logs.append("Transcribing audio using Whisper...")
                # Use Whisper AI to convert speech to text
                result = extract_from_audio(file_bytes, filename)

                # If transcription succeeded, save the text
                if result.get("success"):
                    extracted_content = result["text"]  # Transcribed speech
                    extraction_metadata = {
                        "type": "audio",
                        "language": result["language"],  # Detected language (en, es, etc.)
                        "duration": result["duration"],  # Audio length in seconds
                    }
                    logs.append(
                        f"✓ Audio transcribed ({result['duration']}s, language: {result['language']})"
                    )
                else:
                    # Transcription failed
                    logs.append(
                        f"✗ Audio transcription failed: {result.get('error', 'Unknown error')}"
                    )
                    raise HTTPException(
                        status_code=400, detail="Audio transcription failed"
                    )

            else:
                # File type not supported - tell user
                logs.append(f"✗ Unsupported file type: {filename}")
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file type: {filename}"
                )

        elif text:
            # User sent text instead of a file
            # Check if the text is a YouTube URL
            if is_youtube_url(text):
                # User sent a YouTube URL - try to get the video transcript
                logs.append("Detected YouTube URL, fetching transcript...")
                url = extract_youtube_url(text)  # Extract clean URL
                result = extract_youtube_transcript(url)  # Get transcript from video

                if result.get("success"):
                    extracted_content = result["text"]  # Video transcript
                    extraction_metadata = {
                        "type": "youtube",
                        "title": result["title"],  # Video title
                        "duration": result["duration"],  # Video length
                    }
                    logs.append(f"✓ YouTube transcript fetched: {result['title']}")
                else:
                    # Transcript not available - just use the URL as text
                    logs.append(
                        f"ℹ YouTube transcript unavailable: {result.get('error', 'Unknown error')}"
                    )
                    extracted_content = text
                    extraction_metadata = {"type": "text"}
            else:
                # User sent plain text (not a YouTube URL)
                extracted_content = text
                extraction_metadata = {"type": "text"}
                logs.append("Processing text input...")

        else:
            # User didn't send anything - that's an error
            logs.append("✗ No input provided")
            raise HTTPException(
                status_code=400, detail="No input provided (text or file required)"
            )

        # Make sure we actually got some content (at least 5 characters)
        if not extracted_content or len(extracted_content.strip()) < 5:
            logs.append("⚠ Warning: Very little content extracted")
            return JSONResponse(
                {
                    "status": "error",
                    "message": "Could not extract meaningful content from input",
                    "logs": logs,
                }
            )

        # ======================
        # STEP 2: Intent Classification
        # Figure out what the user wants to do (summarize, analyze sentiment, etc.)
        # ======================

        logs.append("Analyzing your intent...")
        # Ask the AI agent to figure out what the user wants
        intent_result = intent_classifier.classify(extracted_content, text)

        # Check if intent classification worked
        if not intent_result.get("success"):
            logs.append("⚠ Intent classification failed, requesting clarification")
        else:
            # Show user what intent we detected
            logs.append(
                f"✓ Intent identified: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})"
            )

        # ======================
        # STEP 3: Check for Clarification
        # If we're not sure what user wants, ask them to clarify
        # ======================

        if intent_result.get("needs_clarification"):
            # We're not sure what the user wants - need to ask them
            logs.append("ℹ Clarification needed from user")

            # Save the extracted content so we can use it when user responds
            # This allows us to continue the conversation
            conversation_contexts[session_id] = {
                "extracted_content": extracted_content,
                "extraction_metadata": extraction_metadata,
            }

            # Return a response asking user to clarify what they want
            return JSONResponse(
                {
                    "status": "needs_clarification",
                    "extracted_content": (
                        extracted_content[:500] + "..."  # Show preview (first 500 chars)
                        if len(extracted_content) > 500
                        else extracted_content
                    ),
                    "extraction_metadata": extraction_metadata,
                    "question": intent_result.get("clarification_question"),  # Question to ask user
                    "reasoning": intent_result.get("reasoning"),  # Why we need clarification
                    "logs": logs,
                    "session_id": session_id,  # So user can respond with same session
                }
            )

        # ======================
        # STEP 4: Execute Task
        # We know what user wants - now do it (summarize, analyze sentiment, etc.)
        # ======================

        logs.append(f"Executing task: {intent_result['intent']}...")
        # Route to the right task handler and execute it
        final_result = task_router.execute(
            intent_result["intent"],  # What task to do (summarization, sentiment, etc.)
            extracted_content,  # The content to process
            text  # Original user query
        )

        # Check if task completed successfully
        if final_result.get("success"):
            logs.append("✓ Task completed successfully")
        else:
            logs.append("⚠ Task completed with warnings")

        # ======================
        # STEP 5: Return Results
        # Send everything back to the user (frontend)
        # ======================

        return JSONResponse(
            {
                "status": "success",
                "extracted_content": (
                    extracted_content[:1000] + "..."  # Show preview (first 1000 chars)
                    if len(extracted_content) > 1000
                    else extracted_content
                ),
                "extraction_metadata": extraction_metadata,  # Info about extraction
                "intent": {
                    "type": intent_result["intent"],  # What we detected (summarization, etc.)
                    "confidence": intent_result["confidence"],  # How confident (0-1)
                    "reasoning": intent_result.get("reasoning"),  # Why we chose this intent
                },
                "result": final_result,  # The actual task result (summary, sentiment, etc.)
                "logs": logs,  # All the log messages for user to see progress
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions (these are expected errors we handle)
        raise
    except Exception as e:
        # Catch any unexpected errors and return a friendly error message
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)  # Log for debugging
        logs.append(f"✗ Error: {str(e)}")
        return JSONResponse(
            {"status": "error", "message": str(e), "logs": logs}, status_code=500
        )


@app.post("/clarify")
async def clarify_intent(
    clarification: str = Form(...),  # User's response to our clarification question
    session_id: str = Form("default")  # Session ID to find the previous conversation
):
    """
    Handle clarification responses from user
    
    When we ask user "What do you want to do?", they respond here.
    We then re-classify their intent and execute the task.

    Args:
        clarification: User's clarification response (e.g., "summarize this")
        session_id: Session identifier to retrieve the previous context

    Returns:
        JSON response with task execution results
    """
    logs = ["Processing your clarification..."]

    try:
        # Get the stored context from the previous request
        # This contains the extracted content we saved earlier
        context = conversation_contexts.get(session_id)

        # Check if we have the previous context (if not, session expired)
        if not context:
            logs.append("✗ Session context not found")
            raise HTTPException(
                status_code=400,
                detail="Session expired or invalid. Please resubmit your input.",
            )

        # Get the content we extracted earlier
        extracted_content = context["extracted_content"]
        extraction_metadata = context["extraction_metadata"]

        # Now that we have user's clarification, re-classify their intent
        logs.append("Re-analyzing intent with your clarification...")
        intent_result = intent_classifier.classify(extracted_content, clarification)

        # Check if we still need more clarification (user's answer wasn't clear enough)
        if intent_result.get("needs_clarification"):
            logs.append("ℹ Still need more clarification")
            return JSONResponse(
                {
                    "status": "needs_clarification",
                    "question": intent_result.get("clarification_question"),  # Ask again
                    "logs": logs,
                }
            )

        # Great! Now we understand what user wants
        logs.append(f"✓ Intent clarified: {intent_result['intent']}")

        # Execute the task with the clarified intent
        logs.append(f"Executing task: {intent_result['intent']}...")
        final_result = task_router.execute(
            intent_result["intent"],  # The clarified intent
            extracted_content,  # The content from before
            clarification  # User's clarification as the query
        )

        if final_result.get("success"):
            logs.append("✓ Task completed successfully")

        # Clean up - delete the stored context since we're done with this conversation
        if session_id in conversation_contexts:
            del conversation_contexts[session_id]

        # Return the final result to user
        return JSONResponse(
            {
                "status": "success",
                "extracted_content": (
                    extracted_content[:1000] + "..."  # Preview of content
                    if len(extracted_content) > 1000
                    else extracted_content
                ),
                "extraction_metadata": extraction_metadata,
                "intent": {
                    "type": intent_result["intent"],  # What we're doing
                    "confidence": intent_result["confidence"],  # How confident
                },
                "result": final_result,  # The actual task result
                "logs": logs,  # Progress messages
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions (expected errors)
        raise
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Clarification error: {str(e)}", exc_info=True)
        logs.append(f"✗ Error: {str(e)}")
        return JSONResponse(
            {"status": "error", "message": str(e), "logs": logs}, status_code=500
        )


# This runs when you execute the file directly (python app.py)
if __name__ == "__main__":
    import uvicorn  # Web server to run FastAPI

    logger.info("Starting Agentic Multimodal App server...")
    # Start the web server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
