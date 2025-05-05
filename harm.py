from langchain_google_genai import ChatGoogleGenerativeAI
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    raise ImportError("Please install google-generativeai package: pip install google-generativeai")

# Configure safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# Initialize the chat model
chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key="AIzaSyA0HdQsa0OekSYA1n08BKM49xdIF6ru4VI",
    safety_settings=safety_settings,
    temperature=0.7
)
