# Agentic RAG Model Pipeline

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline that automates data gathering, cleaning, analysis, visualization, and retrieval.

---

## Pipeline Overview

1. **Data Collection**
   - Scrape raw data from the web.
   - Output: `fondue_scraping.csv`.

2. **Data Cleaning**
   - Inputs: `teacher.csv`, `fondue_scraping.csv`.
   - Processing: Clean and merge using a Pandas pipeline.
   - Output: Updated `fondue_scraping.csv`.

3. **Model & Statistical Processing**
   - Inputs: `cleaned_data.csv`, `test.csv`, `RAG.docx`.
   - Processing: Generate statistics and prepare data for retrieval.
   - Outputs:
     - `district_stats.csv`
     - `org_stats.csv`
     - `type_stats.csv`

4. **Visualization & Retrieval**
   - Visualize key metrics via **Streamlit**.
   - Retrieve data using **LangGraph** RAG flow.

---
