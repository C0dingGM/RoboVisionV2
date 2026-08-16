import os
from base64 import b64encode
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

gemini_key = os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    raise ValueError(
        f"GOOGLE_API_KEY not found. Add it to {ENV_PATH} "
        "(one line, no spaces around =): GOOGLE_API_KEY=your-key-here"
    )


llm = init_chat_model(
    model='gemini-2.5-flash', 
    model_provider='google_genai', 
    google_api_key=gemini_key,
    temperature=0.0
)


class ChessBoardOutput(BaseModel):
    step_by_step_scan: str = Field(
        description="First determine board orientation (which side is White/Black). "
                    "Then scan rank by rank (8 down to 1) and file by file (a to h), "
                    "listing every detected piece and its exact square coordinate."
    )
    ascii_diagram: str = Field(
        description="A clean, 8x8 ASCII text grid representation of the board "
                    "with row and column labels (a-h and 1-8). Use standard piece letters "
                    "(P, N, B, R, Q, K for White; p, n, b, r, q, k for Black; . for empty)."
    )

IMAGE_PATH = Path(__file__).resolve().parent / "img" / "sample1.jpg"
with open(IMAGE_PATH, 'rb') as image_file:
    encoded_image = b64encode(image_file.read()).decode('utf-8')


message = HumanMessage(
    content=[
        {
            "type": "text", 
            "text": (
                "Analyze the chessboard image carefully. "
                "1. Identify board orientation and scan each square from rank 8 down to 1, files a to h. "
                "2. Double check piece identities (e.g., distinguishing Bishops vs Pawns or Kings vs Queens). "
                "3. Produce a precise ASCII text diagram of the board."
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        }
    ]
)


structured_llm = llm.with_structured_output(ChessBoardOutput)


result: ChessBoardOutput = structured_llm.invoke([message])

print("=== Step-by-Step Scan & Verification ===")
print(result.step_by_step_scan)
print("\n=== Accurate ASCII Chess Board ===")
print(result.ascii_diagram)