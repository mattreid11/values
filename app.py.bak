import os
import json
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

# Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Please set the `GEMINI_API_KEY` environment variable.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------------------
# Structured Data Model for Identification
# ---------------------------------------------------------------------------
class IdentificationResult(BaseModel):
    category: str = Field(description="Category: Sports Card, TCG Card, Coin, Stamp, or Other")
    item_title: str = Field(description="Full item identification title (e.g., '1909-S VDB Lincoln Wheat Cent')")
    year: str = Field(description="Year of issue or release")
    set_or_mint: str = Field(description="Mint mark, card set name, or country of origin")
    variant_details: str = Field(description="Edition, parallel, serial numbering, or double die errors")
    estimated_condition: str = Field(description="Estimated visual raw condition grade (e.g., VF-20, Near Mint, Gem Mint)")
    search_keywords: str = Field(description="Optimized search string for pricing lookups")


# ---------------------------------------------------------------------------
# Agent Functions
# ---------------------------------------------------------------------------
def identify_collectible(image: Image.Image) -> IdentificationResult:
    """Uses Gemini 1.5 Pro to visually inspect and structure metadata."""
    prompt = """
    Examine the provided image of a collectible item (card, coin, stamp, etc.).
    Extract all identifiable details including:
    - Item category
    - Exact item title / subject
    - Year of release / minting
    - Set name, mint mark, or issuing authority
    - Key varieties, errors, card numbers, or special details
    - Estimated physical condition grade based on visual cues
    - An optimized marketplace search query string to find recent sales.
    """
    
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IdentificationResult,
            temperature=0.2,
        ),
    )
    
    return IdentificationResult.model_validate_json(response.text)


def analyze_market_value(item_info: IdentificationResult) -> str:
    """Uses Gemini 1.5 Pro with Google Search Grounding to evaluate market pricing."""
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

    Task:
    1. Search for recent completed sales and current fair market valuations on platforms like eBay, PriceCharting, PCGS, or specialized auction houses for this exact item.
    2. Provide a estimated value range for Ungraded / Raw condition.
    3. If applicable, provide estimated graded values (e.g., PSA 9/10, NGC MS65).
    4. Provide 2-3 key sales factors (e.g., condition sensitivity, recent market demand, rare varieties).
    5. Present the valuation clearly in Markdown format.
    """

    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=pricing_prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],  # Enable real-time Google Search Grounding
            temperature=0.3,
        ),
    )
    
    return response.text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🔍 AI Collectible Identifier & Valuation Agent")
st.markdown("Upload a photo of a **sports card**, **trading card**, **coin**, or **stamp** to identify it and estimate its market value.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Collectible", use_container_width=True)
        
        analyze_btn = st.button("🚀 Analyze Item & Find Value", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Identification & Valuation Results")
    
    if uploaded_file and 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Step 1/2: Visually inspecting item metadata..."):
            try:
                item_details = identify_collectible(image)
            except Exception as e:
                st.error(f"Error during visual identification: {e}")
                st.stop()
        
        # Display extracted metadata
        st.success("Item Identified!")
        st.markdown(f"### **{item_details.item_title}**")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Category", item_details.category)
        m2.metric("Year", item_details.year)
        m3.metric("Est. Condition", item_details.estimated_condition)
        
        with st.expander("Technical Metadata Details"):
            st.write(f"**Set / Mint:** {item_details.set_or_mint}")
            st.write(f"**Variant / Details:** {item_details.variant_details}")
            st.write(f"**Search Query Used:** `{item_details.search_keywords}`")
            
        st.divider()
        
        # Step 2: Market Search & Pricing Synthesis
        with st.spinner("Step 2/2: Searching live sales data for current market values..."):
            try:
                valuation_report = analyze_market_value(item_details)
                st.markdown(valuation_report)
            except Exception as e:
                st.error(f"Error fetching market pricing: {e}")