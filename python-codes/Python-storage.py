import os
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from bson.objectid import ObjectId
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB environment variables
mongo_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")

# Ensure MongoDB configuration is provided
if not mongo_uri or not database_name or not collection_name:
    raise ValueError("MongoDB URI, Database Name, or Collection Name not found in environment variables.")

# Initialize MongoDB client and connect to the database and collection
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client[database_name]
mongo_collection = mongo_db[collection_name]

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API Key not found in environment variables.")
client = OpenAI(api_key=api_key)

# Initialize FastAPI application
app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_with_openai(question, result):
    if not result:
        return "No"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that verifies answers. Evaluate the answer and respond with 'Correct', 'Partially Correct', or 'Incorrect'."},
                {"role": "user", "content": f"Question: {question}\nAnswer: {result}\nEvaluate this answer:"}
            ],
            max_tokens=50
        )
        evaluation = response.choices[0].message.content.strip().lower()
        if 'correct' in evaluation:
            return "Yes"
        elif 'partially correct' in evaluation:
            return "Partially"
        else:
            return "No"
    except Exception as e:
        print(f"Error verifying with OpenAI: {str(e)}")
        return "Error"

@app.get("/verify/{item_id}")
async def verify_answers(item_id: str):
    try:
        document = mongo_collection.find_one({"_id": ObjectId(item_id)})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        questions = document.get("questions", [])
        results = document.get("results", [])

        if len(questions) != len(results):
            raise HTTPException(status_code=400, detail="Questions and results array length mismatch")

        verification_results = []
        yes_count = 0
        partial_count = 0
        total_questions = len(questions)

        for question, result in zip(questions, results):
            verification = verify_with_openai(question, result)
            verification_results.append(verification)
            if verification == "Yes":
                yes_count += 1
            elif verification == "Partially":
                partial_count += 0.5  # Count partially correct as half correct

        # Calculate the ratio
        total_correct = yes_count + partial_count
        ratio = f"{total_correct}/{total_questions}"

        return JSONResponse(content={
            "status": "success",
            "verification_results": verification_results,
            "Final Score": ratio
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-mongo")
async def test_mongo():
    try:
        collections = mongo_db.list_collection_names()
        return JSONResponse(content={"status": "success", "collections": collections})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)