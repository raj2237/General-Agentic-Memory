from dotenv import load_dotenv
load_dotenv()
import os 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)