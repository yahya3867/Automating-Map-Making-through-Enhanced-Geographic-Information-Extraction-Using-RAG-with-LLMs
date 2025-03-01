# Automating Map-Making through Enhanced Geographic Information Extraction Using Retrieval-Augmented Generation with Large Language Models 🌍

<p align="center">
  📄 <a href="XXX" target="_blank">Paper</a> &nbsp;  &nbsp;
</p>

<p align="center"> <img src=".static/images/AutomappingWorkflow.png" style="width: 70%;" id="title-icon"> </p>

## Updates

## Table of Contents 📌
- [Overview](#overview) 🔍
- [Key Results](#key-results) 📊
- [Mapping Workflow](#mapping-workflow) 🗺️
- [Model Zoo](#model-zoo) 🏛️
- [Quick Start](#quick-start) 🚀
- [Demo](#demo) 🎬
- [License](#license) 📜
- [Citation](#citation) 🔖

## Overview
This study introduces an automated mapping workflow that leverages LLMs with RAG to extract geographic information from unstructured text. The multi-agent system (MAS) includes 7 agents:

- **Article Relevance Agent**
- **Location Extraction Agent**
- **Classification Agent**
- **Validation Agent**
- **Text Summarization Agent**
- **Geocoding Agent**
- **ArcPy Agent**

The framework transforms extracted information into spatiotemporal visualizations, supporting applications such as conflict analysis and humanitarian planning.

## Key Results
| Use Case | Processing Time |
|----------|------------------|
| **Case 1** (English Article) | [X mins] |
| **Case 2** (English Article) | [X mins] | 
| **Case 3** (Arabic Article)  |[X mins] |
| **Case 4** (Synthetic Article) |[X mins] | 

## Mapping Workflow

### **Multi-Agent System Overview:**

When a user inputs a news article or textual report:

- **Article Relevance Agent:** Evaluates whether the content is related to the conflict.

- **Location Extraction Agent:** Identifies geographic locations associated with each incident and extracts corresponding incident dates.

- **Classification Agent:** Assigns each incident an event category based on predefined criteria.

- **Text Summarization Agent:** Generates descriptive map titles and structured filenames.

- **Validation Agent:** Conducts a quality control check by reviewing the extracted locations, classifications, and summaries for consistency and accuracy.

- **Geocoding Agent:** Converts textual location data into geographic coordinates using a Geocoding API.

- **ArcPy Agent:** Automates the visualization process by leveraging ArcPy, a Python library within ArcGIS Pro, by producing the final, high-quality geospatial visualizations.

## Model Zoo

| Model Type | Models Used |
|------------|--------------|
| **LLMs**   | `Gemma2-9b` |
| **Geocoding** | `OpenCage API` |
| **GIS Processing** | `ArcPy via ArcGIS Pro` |

## Quick Start

### **Prerequisites**
- **ArcGIS Pro** must be installed.
- **Anaconda Navigator** installed for environment management.
- **VSCode** installed and accessible from Anaconda Navigator.

### **1️⃣ ArcGIS Pro Environment Setup**
1. Open **ArcGIS Pro**.
2. Go to **Settings** → **Python** → **Package Manager**.
3. Next to the active environment, click the **gear icon**.
4. Locate the default environment and click **Clone**.
5. Set the destination and environment name (e.g., `AutomappingEnv`).
6. **Activate** the newly created environment.
7. Create a new project named `BlankTemplate`, save it, and close ArcGIS Pro. This template allows ArcPy to reference a default project for further map creation.

### **2️⃣ Conda Environment and Repository Setup**

1. Open **Anaconda Navigator**.  
2. Select the `AutomappingEnv` environment.  
3. Launch **VSCode** from Anaconda Navigator.  
4. Clone the repository:  
   ```bash
   git clone https://github.com/yahya3867/Automating-Map-Making-through-Enhanced-Geographic-Information-Extraction-Using-RAG-with-LLMs.git
   cd Automating-Map-Making-through-Enhanced-Geographic-Information-Extraction-Using-RAG-with-LLMs
   ```

5. Install dependencies using one of the following options:

##### **Option A: Using `environment.yml`**  
```bash
conda activate AutomappingEnv
conda env update --file environment.yml --prune
```


##### **Option B: Using `requirements.txt`**  
Ensure the `AutomappingEnv` Conda environment is activated before running the following command:  
```bash
pip install -r requirements.txt
```


### **3️⃣ Configure LM Studio and Select Models**
1. **Download and install** [LM Studio](https://lmstudio.ai).
2. **Open LM Studio** and go to the **Discover** tab.
3. **Install the LLM**
   - `Gemma2-9b`
4. **If not pre-installed**, download an embedding model such as:
   - `nomic-ai/nomic-embed-text-v1.5-GGUF`

5. **Load the models:**
   - Navigate to the **Developer** tab.
   - Click **"Select a model to load"** and choose both:
     - The **LLM**
     - The **Embedding Model**
      
6. **Verify model status:**
   - Ensure **Status: Running** before proceeding.
     
### **4️⃣ Running the Application**
```bash
streamlit run Automap.py
```
This will launch the **Streamlit UI** in your browser. You can insert articles directly, and the generated maps will be downloadable to your `Downloads` directory. Alternatively, access the ArcGIS Pro project for further modifications.

## Demo Video

[![Watch the Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

📺 Click the image above or [watch the full demo here](https://www.youtube.com/watch?v=YOUR_VIDEO_ID).

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## Citation
```bibtex
@article{Wang2025automapping,
  title     = {Automating Map-Making through Enhanced Geographic Information Extraction Using Retrieval-Augmented Generation with Large Language Models},
  author    = {Zifu Wang and Yahya Masri and Anusha Srirenganathan Malarvizhi and et al.},
  year      = {2025},
  note      = {Manuscript in preparation, not yet submitted}
}
