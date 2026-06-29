import os
import streamlit as st
import pandas as pd
import requests
import io
from typing import Dict
from fpdf import FPDF

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. ENTERPRISE UI STYLING (GREEN THEME)
# ==========================================
st.set_page_config(page_title="NCS Duty Calculator", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {background-color: #f8f9fa;}
        
        /* Typography & Headings */
        h1, h2, h3 {color: #1a2b4c; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
        
        /* Custom Button Styling - Green Theme */
        .stButton>button {
            background-color: #0f5132;
            color: white;
            border-radius: 6px;
            width: 100%;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #146c43;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Right Column Summary Card styling hook */
        .summary-box {
            background-color: #0f5132;
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .summary-box h4 { color: #e8f5e9; font-size: 14px; margin-bottom: 5px; font-weight: normal; }
        .summary-box h2 { color: white; font-size: 36px; margin-top: 0; margin-bottom: 20px; }
        
        /* Breakdown Table Styling */
        .breakdown-row {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
            color: #333;
        }
        .breakdown-row.total {
            font-weight: bold;
            font-size: 16px;
            border-bottom: none;
            border-top: 2px solid #0f5132;
            margin-top: 10px;
            padding-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA SCHEMA & FINANCIAL ENGINE
# ==========================================
class CustomsClassification(BaseModel):
    hs_code_8_digit: str = Field(description="The exact 8-digit ECOWAS CET HS code extracted from the database")
    item_description: str = Field(description="Official description of the item as found in the tariff book")
    import_duty_rate: float = Field(description="Import duty percentage (ID) extracted from the database")
    requires_nafdac: bool = Field(description="True if item requires NAFDAC clearance")
    requires_soncap: bool = Field(description="True if item requires SON clearance")
    confidence_rationale: str = Field(description="Brief explanation of why this HS code was selected")

def calculate_duties_ngn(fob_usd: float, freight_usd: float, insurance_usd: float, duty_rate_pct: float, ex_rate: float) -> Dict[str, float]:
    """Deterministic calculation of Nigerian Customs Levies converted strictly to NGN."""
    # Convert base USD values to NGN using the Customs Form M Exchange Rate
    fob_ngn = fob_usd * ex_rate
    freight_ngn = freight_usd * ex_rate
    insurance_ngn = insurance_usd * ex_rate
    cif_ngn = fob_ngn + freight_ngn + insurance_ngn
    
    # Official Customs Formula applied to NGN values
    import_duty = cif_ngn * (duty_rate_pct / 100)
    surcharge = import_duty * 0.07
    ciss = fob_ngn * 0.01  # 1% of FOB
    etls = cif_ngn * 0.005 # 0.5% of CIF
    vatable_base = cif_ngn + import_duty + surcharge + ciss + etls
    vat = vatable_base * 0.075
    
    total_customs_cost = import_duty + surcharge + ciss + etls + vat
    
    return {
        "FOB (NGN)": fob_ngn,
        "Freight (NGN)": freight_ngn,
        "CIF (NGN)": cif_ngn,
        "Import Duty": import_duty,
        "Surcharge (7%)": surcharge,
        "CISS (1%)": ciss,
        "ETLS (0.5%)": etls,
        "VAT (7.5%)": vat,
        "Total Payable Duty": total_customs_cost
    }

# ==========================================
# 3. KNOWLEDGE BASE INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_system():
    db_path = "./chroma_cet_live"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(db_path):
        return Chroma(persist_directory=db_path, embedding_function=embeddings)
        
    st.info("System Initializing: Downloading and indexing ECOWAS CET Tariff Book... This only happens once.")
    
    excel_url = "https://customs.gov.ng/wp-content/uploads/2022/06/CET_tariff_v2.xls"
    local_xls = "CET_tariff_v2.xls"
    if not os.path.exists(local_xls):
        response = requests.get(excel_url)
        with open(local_xls, "wb") as f:
            f.write(response.content)
            
    df = pd.read_excel(local_xls, engine="xlrd")
    df = df.dropna(how='all') 
    
    documents = []
    for index, row in df.iterrows():
        row_dict = {str(k).strip(): str(v).strip() for k, v in row.items() if pd.notna(v)}
        content = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
        if len(content) > 10:
            documents.append(Document(page_content=content))
            
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_path)
    return vectorstore

# ==========================================
# 4. LLM INFERENCE PIPELINE
# ==========================================
def analyze_shipment(description: str, vectorstore) -> CustomsClassification:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(description)
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=CustomsClassification)
    
    prompt = PromptTemplate(
        template="""You are a strict Nigerian Customs Classification Engine.
        Find the exact matching HS code for the product from the Knowledge Base Context provided.
        
        Rules:
        1. ONLY use data explicitly found in the context.
        2. Extract the Import Duty (ID) percentage. If not found, use 0.0.
        3. Identify if the product normally requires NAFDAC or SONCAP based on standard Nigerian regulations.
        
        Product: {description}
        
        Knowledge Base Context (Official CET Book):
        {context}
        
        {format_instructions}""",
        input_variables=["description", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({"description": description, "context": context})
    return CustomsClassification(**result)

# ==========================================
# 5. PDF GENERATOR (FIXED FOR UNICODE)
# ==========================================
def create_pdf(classification: CustomsClassification, financials: dict, ex_rate: float) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(15, 81, 50) # Dark Green
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, txt=" NIGERIA CUSTOMS DUTY ESTIMATION REPORT", ln=True, align='L', fill=True)
    pdf.ln(10)
    
    # Classification Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="1. CLASSIFICATION & REGULATORY PROFILE", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=f"Item Description: {classification.item_description}")
    pdf.cell(0, 8, txt=f"HS Code (ECOWAS CET): {classification.hs_code_8_digit}", ln=True)
    pdf.cell(0, 8, txt=f"Base Import Duty Rate: {classification.import_duty_rate}%", ln=True)
    # CHANGED: Replaced ₦ with NGN to prevent Helvetica character map crashes
    pdf.cell(0, 8, txt=f"Exchange Rate Applied: NGN {ex_rate:,.2f} / USD", ln=True) 
    pdf.ln(5)
    
    # Regulatory Section
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, txt="Mandatory Clearances:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, txt=f"- NAFDAC: {'YES' if classification.requires_nafdac else 'NO'}", ln=True)
    pdf.cell(0, 8, txt=f"- SONCAP: {'YES' if classification.requires_soncap else 'NO'}", ln=True)
    pdf.ln(10)
    
    # Financial Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="2. FINANCIAL BREAKDOWN (NGN)", ln=True)
    
    for key, value in financials.items():
        if key in ["Total Payable Duty", "CIF (NGN)"]:
            pdf.set_font("Arial", 'B', 11)
        else:
            pdf.set_font("Arial", '', 11)
        pdf.cell(100, 8, txt=key)
        pdf.cell(0, 8, txt=f"NGN {value:,.2f}", align='R', ln=True)
            
    return bytes(pdf.output())

# ==========================================
# 6. STREAMLIT APPLICATION LAYOUT
# ==========================================
def main():
    st.title("Duty Calculator")
    st.markdown("<p style='color: #666;'>Estimate import duty, levy, and VAT for vehicles and general goods using AI-powered CET Classification.</p>", unsafe_allow_html=True)
    st.write("") # Spacer

    # Initialize RAG silently
    with st.spinner("Connecting to Local CET Database..."):
        vectorstore = initialize_system()

    # Create the modern Left (Inputs) / Right (Results) layout
    col_left, col_right = st.columns([1.3, 1], gap="large")

    # ================= LEFT COLUMN =================
    with col_left:
        with st.container(border=True):
            st.markdown("#### Shipment Parameters")
            st.divider()
            
            # Exchange Rate block
            st.markdown("<p style='font-size: 12px; font-weight: bold; color: #555;'>CURRENCIES & RATE</p>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Invoice Currency", value="USD", disabled=True)
            with c2:
                ex_rate = st.number_input("Customs Exchange Rate (₦)", value=1450.50, step=10.0, format="%.2f")
            
            st.write("") # Spacer
            
            # Item Container
            with st.container(border=True):
                st.markdown("<div style='background-color: #0f5132; color: white; padding: 10px; border-radius: 4px 4px 0 0;'><b> Item 1 Details</b></div>", unsafe_allow_html=True)
                st.write("")
                
                product_desc = st.text_area("Product Specifications / Invoice Line Item", height=100, placeholder="e.g. Matured Atlantic and Pacific bluefin tunas...")
                
                # Values block
                st.write("")
                v1, v2, v3 = st.columns(3)
                with v1:
                    fob_val = st.number_input("FOB Value (USD)", min_value=0.0, value=0.0, step=100.0)
                with v2:
                    freight_val = st.number_input("Freight (USD)", min_value=0.0, value=0.0, step=100.0)
                with v3:
                    ins_val = st.number_input("Insurance (USD)", min_value=0.0, value=0.0, step=50.0)

            st.write("")
            process_btn = st.button("Calculate Duty")

    # ================= RIGHT COLUMN =================
    with col_right:
        if process_btn:
            if not product_desc:
                st.error("Please enter a product description.")
                return
            if fob_val == 0:
                st.warning("Please enter an FOB value greater than 0.")
                return

            with st.spinner("Classifying HS Code and computing tariffs..."):
                try:
                    # 1. Run AI Classification
                    classification = analyze_shipment(product_desc, vectorstore)
                    
                    # 2. Run Financial Engine (Output in NGN)
                    financials = calculate_duties_ngn(fob_val, freight_val, ins_val, classification.import_duty_rate, ex_rate)
                    
                    # --- RENDER SUMMARY CARD ---
                    total_duty = financials['Total Payable Duty']
                    fob_ngn = financials['FOB (NGN)']
                    cif_ngn = financials['CIF (NGN)']
                    
                    st.markdown(f"""
                        <div class="summary-box">
                            <h4>Total Payable Duty (All Items)</h4>
                            <h2>₦{total_duty:,.2f}</h2>
                            <p style="font-size: 12px; border-bottom: 1px solid #ffffff40; padding-bottom: 5px;">HS Code: <b>{classification.hs_code_8_digit}</b> | Duty Rate: <b>{classification.import_duty_rate}%</b></p>
                            <div style="display: flex; justify-content: space-between; font-size: 14px; margin-top: 10px;">
                                <span>Total FOB</span>
                                <b>₦{fob_ngn:,.2f}</b>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 14px; margin-top: 5px;">
                                <span>Total CIF</span>
                                <b>₦{cif_ngn:,.2f}</b>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # --- PDF DOWNLOAD BUTTON ---
                    pdf_bytes = create_pdf(classification, financials, ex_rate)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"Duty_Estimation_{classification.hs_code_8_digit}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.write("")
                    
                    # --- RENDER BREAKDOWN TAB ---
                    with st.container(border=True):
                        st.markdown("##### Overall Breakdown")
                        st.markdown("<hr style='margin: 0; margin-bottom: 15px;'>", unsafe_allow_html=True)
                        
                        # Custom HTML table layout for clean lines
                        def make_row(label, amount, is_total=False):
                            cls = "breakdown-row total" if is_total else "breakdown-row"
                            return f"<div class='{cls}'><span>{label}</span><span>₦{amount:,.2f}</span></div>"

                        html = ""
                        html += make_row("CISS / FCS (1%)", financials['CISS (1%)'])
                        html += make_row("Total ETLS (0.5%)", financials['ETLS (0.5%)'])
                        html += make_row("Total Import Duty", financials['Import Duty'])
                        html += make_row("Total Surcharge (7%)", financials['Surcharge (7%)'])
                        html += make_row("Total VAT (7.5%)", financials['VAT (7.5%)'])
                        html += make_row("TOTAL PAYABLE DUTY", financials['Total Payable Duty'], is_total=True)
                        
                        st.markdown(html, unsafe_allow_html=True)
                        
                        # Add Regulatory warnings if triggered
                        if classification.requires_nafdac or classification.requires_soncap:
                            st.write("")
                            st.warning(f"**Regulatory Notice:** \n"
                                       f"{'• NAFDAC Permit Required' if classification.requires_nafdac else ''}\n"
                                       f"{'• SONCAP Certificate Required' if classification.requires_soncap else ''}")

                except Exception as e:
                    st.error(f"Engine Error: {str(e)}")
        else:
            # Placeholder State when app loads
            st.markdown("""
                <div style="background-color: white; border: 1px dashed #ccc; border-radius: 8px; padding: 40px; text-align: center; color: #888;">
                    <h3 style="color: #ccc;">Awaiting Data</h3>
                    <p>Enter shipment parameters on the left and click Calculate Duty to view the breakdown.</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
