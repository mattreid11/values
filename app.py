import os
import time
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Collectible Valuer",
    page_icon="🪙",
    layout="wide"
)

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Please set GEMINI_API_KEY in Streamlit Secrets or Environment Variables.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
class IdentificationResult(BaseModel):
    category: str = Field(description="Category: Sports Card, TCG Card, Coin, Stamp, or Other")
    item_title: str = Field(description="Full item identification title")
    year: str = Field(description="Year of issue or release")
    set_or_mint: str = Field(description="Mint mark, card set name, or country")
    variant_details: str = Field(description="Edition, parallel, or double die errors")
    estimated_condition: str = Field(description="Estimated visual condition grade")
    search_keywords: str = Field(description="Optimized search string for pricing lookups")


# ---------------------------------------------------------------------------
# Helper: Retry Wrapper for 503 / 429 Errors
# ---------------------------------------------------------------------------
def call_gemini_with_retry(func, max_retries=3, initial_delay=2):
    """Executes a Gemini API call with exponential backoff on transient errors."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e


# ---------------------------------------------------------------------------
# Agent Functions
# ---------------------------------------------------------------------------
def identify_collectible(image: Image.Image) -> IdentificationResult:
    prompt = """
    Examine the provided image of a collectible item (card, coin, stamp, etc.).
    Extract all identifiable details into structured metadata.
    """
    
    def _call():
        return client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IdentificationResult,
                temperature=0.2,
            ),
        )
    
    response = call_gemini_with_retry(_call)
    return IdentificationResult.model_validate_json(response.text)


def analyze_market_value(item_info: IdentificationResult) -> str:
    pricing_prompt = f"""
    You are an expert appraiser for rare collectibles.
    
    Item Details:
    - Category: {item_info.category}
    - Title: {item_info.item_title}
    - Year: {item_info.year}
    - Mint/Set: {item_info.set_or_mint}
    - Details: {item_info.variant_details}
    - Estimated Condition: {item_info.estimated_condition}
    - Search Query: "{item_info.search_keywords}"

    Search for recent completed sales on platforms like eBay, PriceCharting, or PCGS.
    Provide estimated values for Raw vs Graded conditions in clear Markdown.
    """

    def _call():
        return client.models.generate_content(
            model='gemini-3.6-flash',
            contents=pricing_prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.3,
            ),
        )
    
    response = call_gemini_with_retry(_call)
    return response.text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🔍 AI Collectible Identifier & Valuation Agent")
st.markdown("Upload a photo of a **sports card**, **trading card**, **coin**, or **stamp**.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Collectible", use_container_width=True)
        analyze_btn = st.button("🚀 Analyze Item & Find Value", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Results")
    
    if uploaded_file and 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Analyzing image... (retrying automatically if servers are busy)"):
            try:
                item_details = identify_collectible(image)
            except Exception as e:
                st.error(f"Server is currently high-demand. Please wait 10 seconds and try clicking analyze again. Error: {e}")
                st.stop()
        
        st.success("Item Identified!")
        st.markdown(f"### **{item_details.item_title}**")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Category", item_details.category)
        m2.metric("Year", item_details.year)
        m3.metric("Est. Condition", item_details.estimated_condition)
        
        st.divider()
        
        with st.spinner("Searching market prices..."):
            try:
                valuation_report = analyze_market_value(item_details)
                st.markdown(valuation_report)
            except Exception as e:
                st.error(f"Error fetching market pricing: {e}")