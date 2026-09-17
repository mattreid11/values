import os
import json
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
    page_title="AI Collectible Valuer & Catalog",
    page_icon="🪙",
    layout="wide"
)

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Please set GEMINI_API_KEY in Streamlit Secrets or Environment Variables.")
    st.stop()

client = genai.Client(api_key=api_key)

# Initialize Collection Database in Session State
if "collection" not in st.session_state:
    st.session_state.collection = []

# ---------------------------------------------------------------------------
# High-Precision Data Model for Exact Valuation
# ---------------------------------------------------------------------------
class ComprehensiveIdentification(BaseModel):
    category: str = Field(description="Category: Sports Card, TCG Card, Coin, Stamp, or Other")
    item_name: str = Field(description="Exact subject/player/design name (e.g., 'Michael Jordan', 'Lincoln Wheat Cent', 'Pikachu')")
    year_issued: str = Field(description="Year of manufacture or minting")
    brand_or_publisher: str = Field(description="Manufacturer/Set/Country (e.g., Fleer, Panini, Topps, US Mint, Pokémon Base Set, USPS)")
    card_or_catalog_number: str = Field(description="Card number (#57), Scott Stamp #, or KM Coin Catalog #")
    mint_mark_or_printing: str = Field(description="Mint mark (e.g., 'S', 'D', 'CC'), 1st Edition logo, Shadowless, or Perforation size")
    key_varieties_and_errors: str = Field(description="Sub-types, parallels, refractor colors, die cracks, or double strikes (e.g., VDB, Red Refractor /5, 1st Edition)")
    autograph_or_serial_number: str = Field(description="Is it autographed or numbered? (e.g., 'On-card Auto 04/10', 'None')")
    grading_status: str = Field(description="Graded (PSA, BGS, NGC, PCGS + Cert # if visible) vs Raw")
    estimated_grade_condition: str = Field(description="Visual grade estimate (e.g., PSA 10, Near Mint 7, MS-63, Used/Hinged)")
    critical_wear_flaws: str = Field(description="Key wear points: surface scratches, corner wear, centering off-ratio, creases, or pinholes")
    exact_search_queries: list[str] = Field(description="2-3 highly specific marketplace search strings to look up exact completed comps on eBay, PriceCharting, or PCGS")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def call_gemini_with_retry(func, max_retries=3, initial_delay=2):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e


def identify_collectible_exact(image: Image.Image) -> ComprehensiveIdentification:
    prompt = """
    Analyze the provided image of a collectible item with maximum precision.
    Extract every critical parameter required by professional appraisers to determine exact fair market value:
    
    1. Identify exact Subject/Player/Design, Year, Manufacturer/Set, and Card Number/Catalog ID.
    2. Inspect closely for key varieties (e.g., 1st Edition, Shadowless, Refractors, Mint Marks like 'S' or 'CC', Double Die errors).
    3. Check for signatures, patch cards, or serial numbers (e.g., 05/25).
    4. Assess grading status (is it encapsulated by PSA, BGS, NGC, PCGS? If raw, estimate the numerical/adjective condition grade).
    5. Note specific physical condition factors (corner wear, centering, surface scratches, creases, toning, or hinge marks).
    6. Formulate precise search terms for fetching exact sold comps on eBay or specialized pricing guides.
    """
    
    def _call():
        return client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ComprehensiveIdentification,
                temperature=0.1,  # Low temperature for precise factual extraction
            ),
        )
    
    response = call_gemini_with_retry(_call)
    return ComprehensiveIdentification.model_validate_json(response.text)


def analyze_market_value(item: ComprehensiveIdentification) -> str:
    search_terms_str = " OR ".join([f'"{q}"' for q in item.exact_search_queries])
    
    pricing_prompt = f"""
    You are an expert appraiser evaluating market pricing for a collectible.
    
    Item Specification:
    - Subject/Title: {item.item_name}
    - Year & Publisher: {item.year_issued} {item.brand_or_publisher}
    - Card/Catalog #: {item.card_or_catalog_number}
    - Mint Mark / Edition: {item.mint_mark_or_printing}
    - Varieties/Serial #: {item.key_varieties_and_errors} | {item.autograph_or_serial_number}
    - Status & Grade: {item.grading_status} (Est. Grade: {item.estimated_grade_condition})
    - Visible Condition Notes: {item.critical_wear_flaws}
    
    Recommended Search Strings: {search_terms_str}

    Instructions:
    1. Perform a real-time web search for recent completed sales and current market benchmarks on platforms like eBay, PriceCharting, PCGS CoinFacts, or TCGplayer.
    2. Provide an **Exact Valuation Breakdown**:
       - **Raw/Ungraded Estimated Value Range**
       - **Graded Value Benchmarks** (PSA 8, 9, 10 or NGC/PCGS MS60, MS63, MS65+ where applicable)
    3. Highlight the **Top 3 Value Drivers** for this specific item (e.g., centering, specific key variety premium, grade multiplier).
    4. Provide direct advice on whether sending this item to a professional grading service (PSA/BGS/NGC/PCGS) would yield a positive ROI.
    """

    try:
        response = call_gemini_with_retry(
            lambda: client.models.generate_content(
                model='gemini-3.6-flash',
                contents=pricing_prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.2,
                ),
            )
        )
        return response.text
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            st.warning("⚠️ Live search quota reached. Falling back to offline AI valuation estimation.")
            response = call_gemini_with_retry(
                lambda: client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=pricing_prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
            )
            return response.text
        else:
            raise e


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🪙 AI Precision Collectible Valuer & Catalog")

tabs = st.tabs(["🔍 Precision Identification & Valuation", "📚 Collection Database"])

# --- TAB 1: IDENTIFY & VALUE ---
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload High-Res Image")
        uploaded_file = st.file_uploader("Upload clear photo (front/back)...", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Collectible", use_container_width=True)
            analyze_btn = st.button("🚀 Identify & Estimate Exact Value", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Detailed Appraiser Report")
        
        if uploaded_file and 'analyze_btn' in locals() and analyze_btn:
            with st.spinner("Extracting mint marks, catalog numbers, and grading cues..."):
                try:
                    item_details = identify_collectible_exact(image)
                except Exception as e:
                    st.error(f"Error during visual identification: {e}")
                    st.stop()
            
            st.success("Item Identified!")
            st.markdown(f"### **{item_details.year_issued} {item_details.brand_or_publisher} {item_details.item_name}**")
            
            # Key Metadata Metrics Display
            m1, m2, m3 = st.columns(3)
            m1.metric("Category", item_details.category)
            m2.metric("Catalog / Card #", item_details.card_or_catalog_number)
            m3.metric("Est. Grade", item_details.estimated_grade_condition)
            
            # Granular Attributes Table
            with st.expander("📋 Full Identification Specifications", expanded=True):
                st.write(f"**Mint Mark / Printing:** {item_details.mint_mark_or_printing}")
                st.write(f"**Varieties & Errors:** {item_details.key_varieties_and_errors}")
                st.write(f"**Autograph / Serial #:** {item_details.autograph_or_serial_number}")
                st.write(f"**Grading Status:** {item_details.grading_status}")
                st.write(f"**Physical Condition Notes:** {item_details.critical_wear_flaws}")
                st.write(f"**Target Search String:** `{item_details.exact_search_queries[0] if item_details.exact_search_queries else ''}`")

            st.divider()
            
            # Market Valuation Step
            with st.spinner("Searching auction databases and live completed listings..."):
                try:
                    valuation_report = analyze_market_value(item_details)
                    st.markdown(valuation_report)
                except Exception as e:
                    st.error(f"Error fetching market pricing: {e}")
                    valuation_report = "Valuation analysis unavailable."

            # Record detailed payload in session collection
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "item_name": item_details.item_name,
                "category": item_details.category,
                "year": item_details.year_issued,
                "brand_set": item_details.brand_or_publisher,
                "card_catalog_no": item_details.card_or_catalog_number,
                "mint_printing": item_details.mint_mark_or_printing,
                "varieties": item_details.key_varieties_and_errors,
                "autograph_serial": item_details.autograph_or_serial_number,
                "grading_status": item_details.grading_status,
                "estimated_grade": item_details.estimated_grade_condition,
                "condition_flaws": item_details.critical_wear_flaws,
                "search_queries": item_details.exact_search_queries,
                "valuation_report": valuation_report
            }
            st.session_state.collection.append(record)
            st.toast("Saved detailed record to collection catalog!", icon="💾")


# --- TAB 2: MY COLLECTION DATABASE ---
with tabs[1]:
    st.subheader("📁 Complete Scanned Inventory")
    
    if not st.session_state.collection:
        st.info("No items scanned yet. Scan an item in the first tab to build your inventory.")
    else:
        st.write(f"**Total Collectibles in Database:** {len(st.session_state.collection)}")
        
        # Display inventory table
        st.dataframe(st.session_state.collection, use_container_width=True)
        
        # Convert full list to JSON
        json_data = json.dumps(st.session_state.collection, indent=2)
        
        # Download button
        st.download_button(
            label="📥 Download Full collection.json Database",
            data=json_data,
            file_name="collection_full_specs.json",
            mime="application/json",
            type="primary"
        )