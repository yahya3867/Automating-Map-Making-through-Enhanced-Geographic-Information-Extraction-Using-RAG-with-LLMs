import os
import shutil
import pandas as pd
import openai
from openai import OpenAI
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext
import faiss
import requests
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
import arcpy
import numpy as np
import streamlit as st

from PIL import Image

# Disables the pixel limit, removing the DecompressionBombWarning.
Image.MAX_IMAGE_PIXELS = None

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = 'lm-studio'
os.environ['OPENAI_API_BASE'] = 'http://localhost:1234/v1'
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

# Initialize OpenCage Geocoder
opencage_api_key = os.getenv("OPENCAGE_API_KEY")
geocoder = OpenCageGeocode(opencage_api_key)

# Streamlit App Title and Description
st.title("Automapping Tool")
st.write("Generate a thematic map from your article text.")

# Streamlit Input for Article Text
article_input = st.text_area(
    "Article Text",
    height=300,
    placeholder="Enter your article here..."
)

# Streamlit Button to Trigger Processing
if st.button("Generate Map"):
    if not article_input.strip():
        st.error("Error: The input cannot be empty.")
    else:
        stop_process = st.button("Stop Process")
        if stop_process:
            st.warning("Process interrupted by user.")
            st.stop()
            
        # Spinner for Content Relevance Check
        relevance_result = None
        relevance_error = None
        with st.spinner('Checking content relevance...'):
            try:
                # Use LLM to check if the input article is relevant to conflict
                st.write("Validating article relevance using LLM...")

                d = 768
                faiss_index = faiss.IndexFlatL2(d)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents([Document(text=article_input)], storage_context=storage_context)

                # Query LLM to check if the article is relevant to conflict
                relevance_query = """
                Determine if the given article is about an ongoing armed conflict, war, or political violence in Sudan.
                Articles should explicitly mention Sudan or Sudan Conflict related dynamics.
                If it is relevant, return "RELEVANT". If not, return "IRRELEVANT".
                
                Article: {article}
                """.format(article=article_input)

                relevance_response = index.as_query_engine(similarity_top_k=1).query(relevance_query)
                relevance_result = relevance_response.response.strip()
                
            except Exception as e:
                relevance_error = str(e)

        if relevance_result == "IRRELEVANT":
            st.error("Error: The input article is not related to conflict events. Please provide a relevant article.")
            st.stop()                
                
        st.success("Article is relevant. Proceeding with processing...")

        with st.spinner("Processing article....This may take a few moments..."):
            try:                
                    # Paths for project, layer, and template
                    new_project_dir = r"E:/Yahya Masri/ArcGIS_Projects/test_st1"
                    new_aprx_path = r"E:/Yahya Masri/ArcGIS_Projects/test_st1/test_st1.aprx"
                    layer_path = "E:/Yahya Masri/automapping_files/sdn_adm_cbs_nic_ssa_20200831_shp/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp"
                    template_project_path = r"E:/Yahya Masri/ArcGIS_Projects/BlankTemplate/BlankTemplate.aprx"
                    # Ensure the directory for the new project exists
                    new_project_dir = os.path.dirname(new_aprx_path)
                    os.makedirs(new_project_dir, exist_ok=True)  # Create the directory if it doesn't exist

                    # Create a new project by copying the template
                    aprx = arcpy.mp.ArcGISProject(template_project_path)
                    aprx.saveACopy(new_aprx_path)

                    # Open the newly created project
                    new_aprx = arcpy.mp.ArcGISProject(new_aprx_path)

                    # Create a new file geodatabase in the .aprx directory
                    gdb_name = "test_st1.gdb"
                    gdb_path = os.path.join(new_project_dir, gdb_name)
                    if not os.path.exists(gdb_path):  # Check if the GDB already exists
                        arcpy.CreateFileGDB_management(new_project_dir, gdb_name)
                        #st.write(f"File geodatabase created at: {gdb_path}")
                    else:
                        st.write(f"File geodatabase already exists at: {gdb_path}")

                    # Add the new .gdb to the project
                    new_aprx.defaultGeodatabase = gdb_path
                    #st.write(f"Default geodatabase set to: {gdb_path}")

                    # Get the first map from the project and add layer
                    try:
                        first_map = new_aprx.listMaps()[0]
                        sudan_shape = first_map.addDataFromPath(layer_path)
                        #st.write(f"Layer '{layer_path}' added to map in new project at: {new_aprx_path}")
                    except IndexError:
                        st.error("No maps found in the template project. Could not add layer.")

                    try:
                        for lyr in first_map.listLayers():
                            if lyr.name == sudan_shape.name:  # Ensure working with the correct layer   
                                sym = lyr.symbology
                                if sym.renderer.type == "SimpleRenderer":
                                    sym.renderer.symbol.symbolType = "esriSFS"
                                    sym.renderer.symbol.color = {'RGB': [255, 255, 255, 0]}
                                    sym.renderer.symbol.outlineColor = {'RGB': [0, 0, 0, 255]}
                                    sym.renderer.symbol.outlineWidth = 1.5
                                    lyr.symbology = sym 
                                    lyr.showLabels = True  
                                    lblClass = lyr.listLabelClasses()[0] 
                                    lblClass.expression = "$feature.ADM1_EN"  
                                    lblClass.showClassLabels = True                          
                                    lblClass.visible = True  
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")


                    ### **Save Article to CSV**
                    csv_name = "Auto.csv"
                    csv_file_path = os.path.join(new_project_dir, csv_name).replace("\\", "/")  # Path for the new CSV file

                    # Article text
                    article_text = article_input

                    # Create a new DataFrame and save it to a CSV file
                    data = {'Text': [article_text]}
                    df = pd.DataFrame(data)
                    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

                    df['Extracted_Meta_data'] = None

                    text = article_text
                    text_list = [text]
                    st.write("Processing text for metadata extraction...")
                    documents = [Document(text=t) for t in text_list]
                    # Set up the FAISS vector store
                    d = 768
                    faiss_index = faiss.IndexFlatL2(d)
                    vector_store = FaissVectorStore(faiss_index=faiss_index)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
                    st.write("Extracting incident locations...")
                    st.write("Classifying incident types...")
                    query = """Identify the geographic location(s) and the time(s) where the incident(s) occurred
                    in the format of "neighborhood, state, country | mm/dd/yyyy". 
                    If there are multiple incidents, list each incident with its corresponding location and date on a separate
                    line without any explanation.

                    Furthermore, your task is to also classify each news incident into one of the following categories. The output should only contain the category name without any explanation.
                    The category should be placed in the format of "incident location | incident date | incident category" : 
            
                    Unlawful detention- Refers to refers to the act of detaining or confining an individual without legal justification or due process. For example, if protesters are arrested and detained without legal basis during peaceful demonstrations, with no access to legal representation, this would be considered unlawful detention. 
                    Human trafficking- Refers to the act of recruiting, transporting, transferring, harboring, or receiving individuals through force, fraud, coercion, or other forms of deception for the purpose of exploitation. Exploitation can take many forms, including forced labor, sexual exploitation, slavery, servitude, or the removal of organs. It is considered a severe violation of human rights and is illegal under international and domestic laws. If an incident is considered Human trafficking it would also be considered as a War crime. 
                    Enslavement- refers to the act of exercising ownership or control over another person, treating them as property, and depriving them of their freedom. It often involves forcing individuals to perform labor or services under coercion, violence, or the threat of punishment. If an incident is considered Enslavement, it would also be considered as a War crime. 
                    Willful killing of civilians- Refers to the intentional killing of civilians who are not directly participating in hostilities, with full knowledge of their noncombatant status. This includes acts like massacres, executions, or deliberate bombings of civilian sites such as homes, schools, or hospitals, where the clear intent is to cause death. For example, a military unit massacring the residents of a village. 
                    Mass execution- Refers to the deliberate killing of a large scale number of individuals, often carried out by state or non-state actors as part of systematic persecution, acts of war, or punitive measures. The victims are typically selected based on political, ethnic, religious, or social affiliations, and the killings are often premeditated and organized. If an incident is considered Mass execution, it would also be considered as a War crime. 
                    Kidnapping- Refers to the unlawful and intentional abduction, confinement, or holding of an individual against their will, often for a specific purpose such as extortion, ransom, political leverage, forced labor, or exploitation. It is a serious crime and violates the individual's right to freedom and security. 
                    Extrajudicial killing- Refers to the killing of a person without any legal process, such as arrest, trial, or sentencing. It is carried out outside the law, often by state agents or with their approval. 
                    Forced disappearance- Refers the act of abducting or detaining a person against their will, followed by a refusal to disclose their fate or whereabouts. This leaves the victim outside the protection of the law and often causes anguish to their family and community.                         Damage or destruction of civilian critical infrastructure- Refers to the reckless harm, sabotage, or destruction of essential facilities, systems, or services necessary for the well-being, safety, and survival of civilian populations. This includes infrastructure such as hospitals, water supplies, power grids, schools, transportation systems, and communication networks. 
                    Damage or destruction, looting, or theft of cultural heritage- Refers to the harm, removal, or appropriation of culturally significant sites, objects, or artifacts during conflicts, disasters, or other destabilizing events. These acts violate international laws that protect cultural heritage as part of humanity's shared history and identity. Furthermore, this also refers to looting incidents. 
                    Military operations (battle, shelling)- Refers to actions explicitly conducted between opposing armed forces, such as the RSF and SAF, during a conflict or war. These actions involve the use of weapons, strategies, and tactics to achieve military objectives, focusing on direct engagements or operations targeting enemy positions. Narratives mentioning attacks on civilian areas or indiscriminate shelling are not included in this category, even if long-range weapons or artillery are used. 
                    Gender-based or other conflict-related sexual violence- Refers to acts of sexual violence committed during or as a result of armed conflict, often targeting individuals based on their gender, identity, or perceived vulnerability. Incidents such as rape or sexual harassment are considered Gender-based or other conflict-related sexual violence. 
                    Violent crackdowns on protesters/opponents/civil rights abuse- Refers to the use of excessive or unlawful force suppress dissent, silence opposition. These acts often involve targeting individuals or groups engaging in protests, political opposition, or advocacy for civil rights. 
                    Indiscriminate use of weapons- Refers to the use of weapons, such as shelling or bombing, in a manner that impacts buildings, neighborhoods, or areas without clear differentiation between combatants and civilians, or military and civilian infrastructure. This category applies only to incidents involving the use of explosives or long-range weapons that cause widespread harm or destruction, regardless of whether brute force or a massacre is involved, unless explicitly mentioned. 
                    Torture or indications of torture- Refers to the infliction of severe physical or psychological pain and suffering on a person, typically to punish, intimidate, extract information, or coerce. 
                    Persecution based on political, racial, ethnic, gender, or sexual orientation- Refers to the systematic mistreatment, harassment, or oppression of individuals or groups due to their political beliefs, race, ethnicity, gender identity, or sexual orientation. 
                    Movement of military, paramilitary, or other troops and equipment- Refers to the deployment, transfer, or relocation of armed forces, armed groups, or their equipment as part of strategic or operational objectives. This movement may occur during preparation for conflict, active military operations, or in maintaining a presence in certain areas. 
                    
                    Classify each incident into one of the above categories. Note that a single incident can be classified into only one category based on the most prominent theme. If an incident fits into multiple categories, select the one that best describes the primary issue.
                    """
                    response = index.as_query_engine(similarity_top_k=1).query(query)
                    #st.write("LLM Response:", response.response.strip())
                    lines = response.response.strip().split('\n')
                    
                    if not lines:
                        st.error("Error: LLM response is empty. Ensure the model is running and returning valid outputs.")
                        st.stop()

                    article_text = df.loc[0, "Text"]
                    # Build a list of dicts, each dict = one new row
                    new_rows = []
                    for line in lines:
                        line = line.strip()
                        if "|" in line:  # Ensure expected format
                            new_rows.append({
                                "Text": article_input,
                                "Extracted_Meta_data": line
                            })

                    # Only create DataFrame if new_rows is not empty
                    if new_rows:
                        df = pd.DataFrame(new_rows)
                    else:
                        st.error("Error: No extracted metadata. LLM might not be returning expected format.")
                        st.stop()

                    if "Extracted_Meta_data" not in df.columns or df["Extracted_Meta_data"].isna().all():
                        st.error("Error: 'Extracted_Meta_data' is missing or empty. Unable to extract metadata.")
                        st.stop()
                    #st.write("Extracted Metadata Preview:", df["Extracted_Meta_data"].head())

                     # Clean up whitespace
                    df['Geo_Meta_data'] = None
                    df['IncidentType'] = None

                    if "Geo_Meta_data" not in df.columns:
                        st.error("Geo_Meta_data columns are missing.")
                        st.stop()

                    if "IncidentType" not in df.columns:
                        st.error("IncidentType columns are missing.")
                        st.stop()                        

                    #FUNCTION TO SPLIT LOCATION AND DATE FROM CLASSIFICATION
                    def split_metadata(row):
                        try:
                            parts = row.rsplit("|", 1)  # Splits only at the last "|"
                            if len(parts) == 2:
                                geo_meta = parts[0].strip()
                                incident_type = parts[1].strip()
                                return geo_meta, incident_type
                            else:
                                return None, None
                        except Exception as e:
                            st.error(f"Error processing metadata row '{row}': {e}")
                            return None, None

                        
                    df[['Geo_Meta_data', 'IncidentType']] = df['Extracted_Meta_data'].apply(lambda x: pd.Series(split_metadata(x)))

                    st.write("Validating extracted geographic metadata...")
                    for i, row in df.iterrows():
                        validation_query = f"""
                        Ensure that the following line adheres to the format:
                        1. "neighborhood, state, country | mm/dd/yyyy"
                        2. Incident Location and Incident Date is split by "|"
                        3. No additional text or explanation.

                        If the line is already valid, return it as is. If not, fix it.

                        Line: {row['Geo_Meta_data']}
                        """
                        validation_response = index.as_query_engine(similarity_top_k=1).query(validation_query)
                        validated_line = validation_response.response.strip()
                        df.at[i, 'Geo_Meta_data'] = validated_line

                    #SPLIT GEO_META_DATA
                    def split_geo_meta_data(df):
                        # Ensure Geo_Meta_data exists and is properly formatted
                        if 'Geo_Meta_data' in df.columns:
                            # Split Geo_Meta_data into Incident Location and Incident Date
                            df[['Incident Location', 'Incident Date']] = df['Geo_Meta_data'].str.split('|', expand=True)
                            # Strip whitespace
                            df['Incident Location'] = df['Incident Location'].str.strip()
                            df['Incident Date'] = df['Incident Date'].str.strip()
                            # Drop the original Geo_Meta_data column
                            df.drop(columns=['Geo_Meta_data'], inplace=True)
                        return df

                    # Apply the function to the DataFrame
                    df = split_geo_meta_data(df)
                    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

                    st.write("Validating article category classification...")
                    for i, row in df.iterrows():
                        categories = [
                            "Unlawful detention",
                            "Human trafficking",
                            "Enslavement",
                            "Willful killing of civilians",
                            "Mass execution",
                            "Kidnapping",
                            "Extrajudicial killing",
                            "Forced disappearance",
                            "Damage or destruction of civilian critical infrastructure",
                            "Damage or destruction, looting, or theft of cultural heritage",
                            "Military operations (battle, shelling)",
                            "Gender-based or other conflict-related sexual violence",
                            "Violent crackdowns on protesters/opponents/civil rights abuse",
                            "Indiscriminate use of weapons",
                            "Torture or indications of torture",
                            "Persecution based on political, racial, ethnic, gender, or sexual orientation",
                            "Movement of military, paramilitary, or other troops and equipment"
                        ]



                        if row['IncidentType'] not in categories:
                            validation_query = f"""
                            Check if the following classification is valid:
                                - The category must exactly match one of the 17 categories provided in the list below.
                                - If the classification is valid, return it as is.
                                - If invalid, select the most appropriate category from the list and return it. Do not return anything other than a valid category.
                            Categories: {categories}
                            IncidentType
                            """
                            validation_response = index.as_query_engine(similarity_top_k=1).query(validation_query)
                            validated_classification = validation_response.response.strip()
                            if validated_classification in categories:
                                df.at[i, 'IncidentType'] = validated_classification   

                    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')


                    ### **Geocoding**
                    st.write("Geocoding incident locations...")
                    df = pd.read_csv(csv_file_path)

                    # Define function to get coordinates
                    def opencage_geocode(location):
                        try:
                            result = geocoder.geocode(location)
                            if result:
                                return result[0]['geometry']['lat'], result[0]['geometry']['lng']
                            else:
                                return None, None
                        except Exception as e:
                            st.error(f"Geocoding Error: {e}")
                            return None, None

                    # Apply geocoding function to each location
                    df['Latitude'], df['Longitude'] = zip(*df['Incident Location'].apply(lambda loc: opencage_geocode(loc)))

                    # Save the new DataFrame to CSV
                    df.to_csv(csv_file_path, index=False)

                    ### **Extract State from Location**
                    #st.write("Extracting state information from incident locations...")
                    def extract_state_from_sudan_format(df):
                        if 'Incident Location' in df.columns:
                            def extract_state(location):
                                try:
                                    # Check if 'Sudan' is in the location string
                                    if 'Sudan' in location:
                                        # Split the string at 'Sudan' and take the part before it
                                        before_sudan = location.rsplit('Sudan', 1)[0]
                                        # Split again at the last comma to isolate the state
                                        state = before_sudan.rsplit(',', 1)[-1].strip()
                                        return state
                                    return None  # If 'Sudan' not in location, return None
                                except Exception as e:
                                    st.error(f"Error processing location '{location}': {e}")
                                    return None
                            
                            # Apply the extraction logic to the column
                            df['State'] = df['Incident Location'].apply(extract_state)
                        else:
                            st.error("Error: 'Incident Location' column is not in the DataFrame.")
                        return df

                    # Apply the function to extract the state
                    df = extract_state_from_sudan_format(df)

                    # Save the updated DataFrame back to the CSV
                    df.to_csv(csv_file_path, index=False)

                    ### **Map Title Generation**
                    st.write("Generating map title...")
                    client = OpenAI(base_url=openai.api_base, api_key=openai.api_key)
                    
                    messages = [
                        {"role": "system", "content": """Create ONE clear, concise, and accurate map title. Follow these rules:
                        1. The title must reflect the map's primary purpose and topic.
                        2. Include the geographic location and relevant details without extraneous or misleading elements.
                        3. Titles should remain brief, fitting within limited space, while fully conveying the map’s intent.
                        4. Avoid redundancy, such as the phrase "A Map of…" or unnecessary abbreviations.
                        5. Adapt the emphasis of the title based on context:
                        - For series maps, highlight distinguishing features like the location.
                        - For standalone thematic maps, emphasize the theme or key aspect.
                        6. Ensure the title reflects the intended audience and avoids ambiguity.
                        7. If a date is relevant, include it appropriately within the title.

                        Furthermore, do not produce markdown formatting nor introductory text; only provide the title without an explanation.
                        """ },
                        {"role": "user", "content": article_text},
                    ]

                    completion = client.chat.completions.create(
                        model="model-identifier", 
                        messages=messages,
                        temperature=0,
                        stream=False,
                    )

                    map_title = completion.choices[0].message.content.strip()
                    #st.write("Map Title:", map_title)

                    ### **PNG Keyphrase Generation**
                    st.write("Generating PNG naming convention...")
                    messages = [
                        {"role": "system", "content": """Generate a very short, concise keyphrase to name a Map PNG based on the given article. 
                        Ensure the output is formatted as a valid PNG filename using only lowercase letters, numbers, and underscores. 
                        Do not include spaces, special characters, markdown formatting, or any explanation of the chosen name."""},
                        {"role": "user", "content": article_text},
                    ]

                    completion = client.chat.completions.create(
                        model="model-identifier", 
                        messages=messages,
                        temperature=0,
                        stream=False,
                    )
                    import re
                    png_keyphrase = completion.choices[0].message.content.strip()
                    png_keyphrase = re.sub(r'[^a-zA-Z0-9_]', '_', png_keyphrase)
                    
                    #st.write("PNG Keyphrase:", png_keyphrase)

                    ### **Define Symbology Layers Paths**
                    #st.write("Defining symbology layers...")
                    damage_or_destruction_looting_or_theft_of_cultural_heritage_symbology_layer = "E:/Yahya Masri/automapping_files/damage_or_destruction_looting_or_theft_of_cultural_heritage.lyrx"
                    damage_or_destruction_of_civilian_critical_infrastructure_symbology_layer = "E:/Yahya Masri/automapping_files/damage_or_destruction_of_civilian_critical_infrastructure.lyrx"
                    enslavement_symbology_layer = "E:/Yahya Masri/automapping_files/enslavement.lyrx"
                    extrajudicial_killing_symbology_layer = "E:/Yahya Masri/automapping_files/extrajudicial_killing.lyrx"
                    forced_disappearance_symbology_layer = "E:/Yahya Masri/automapping_files/forced_disappearance.lyrx"
                    gender_based_violence_symbology_layer = "E:/Yahya Masri/automapping_files/gender_based_violence.lyrx"
                    human_trafficking_symbology_layer = "E:/Yahya Masri/automapping_files/human_trafficking.lyrx"
                    indiscriminate_use_of_weapons_symbology_layer = "E:/Yahya Masri/automapping_files/indiscriminate_use_of_weapons.lyrx"
                    kidnapping_symbology_layer = "E:/Yahya Masri/automapping_files/kidnapping.lyrx"
                    mass_excecution_symbology_layer = "E:/Yahya Masri/automapping_files/mass_excecution.lyrx"
                    military_operations_symbology_layer = "E:/Yahya Masri/automapping_files/military_operations.lyrx"
                    movement_of_military_symbology_layer = "E:/Yahya Masri/automapping_files/movement_of_military.lyrx"
                    persecution_symbology_layer = "E:/Yahya Masri/automapping_files/persecution.lyrx"
                    torture_symbology_layer = "E:/Yahya Masri/automapping_files/torture.lyrx"
                    unlawful_detention_symbology_layer = "E:/Yahya Masri/automapping_files/unlawful_detention.lyrx"
                    violent_crackdowns_on_protesters_symbology_layer = "E:/Yahya Masri/automapping_files/violent_crackdowns_on_protesters.lyrx"
                    willful_killing_of_civilians_symbology_layer = "E:/Yahya Masri/automapping_files/willful_killing_of_civilians.lyrx"
                    
                    ### **Add Incidents to Geodatabase**
                    st.write("Adding incidents to geodatabase...")
                    arcpy.env.workspace = new_aprx.defaultGeodatabase
                    in_file = csv_file_path
                    out_feature_class = 'Incidents'
                    x_coords = "Longitude"
                    y_coords = "Latitude"
                    z_field = ''
                    arcpy.management.XYTableToPoint(in_file, out_feature_class, x_coords, y_coords, z_field, arcpy.SpatialReference(4326))
                    fields = ['State', 'Incident_Location']

                    with arcpy.da.UpdateCursor(out_feature_class, fields) as cursor:
                        for row in cursor:
                            if not row[1]:
                                row[1] = row[0]
                                cursor.updateRow(row)

                    #st.write(f"Feature class '{out_feature_class}' created with incident locations.")

                    # Delete the existing layer if it exists
                    if arcpy.Exists("AllPointsLayer"):
                        arcpy.Delete_management("AllPointsLayer")
                        #st.write("Existing 'AllPointsLayer' deleted.")

                    # Add the feature class to the map
                    try:
                        incidents_layer_path = os.path.join(gdb_path, out_feature_class)
                        first_map.addDataFromPath(incidents_layer_path)  # Add the feature class to the map
                        #st.write(f"Feature class '{out_feature_class}' added to the map.")
                    except Exception as e:
                        st.error(f"Error adding feature class to the map: {e}")

                    # Save the project with all changes
                    new_aprx.save()
                    #st.write(f"Project saved with all updates at: {new_aprx_path}")

                    ### **Apply Symbology Layers**
                    #st.write("Applying symbology layers...")
                
                    # 1
                    try:
                        damage_or_destruction_looting_or_theft_of_cultural_heritage_name = "Damage or destruction, looting, or theft of cultural heritage"
                        damage_or_destruction_looting_or_theft_of_cultural_heritage_query = "IncidentType ='Damage or destruction, looting, or theft of cultural heritage'"            
                        damage_or_destruction_looting_or_theft_of_cultural_heritage_layer = arcpy.management.MakeFeatureLayer(out_feature_class, damage_or_destruction_looting_or_theft_of_cultural_heritage_name, damage_or_destruction_looting_or_theft_of_cultural_heritage_query)
                        damage_or_destruction_looting_or_theft_of_cultural_heritage_layer_obj = damage_or_destruction_looting_or_theft_of_cultural_heritage_layer.getOutput(0) 
                        first_map.addLayer(damage_or_destruction_looting_or_theft_of_cultural_heritage_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], damage_or_destruction_looting_or_theft_of_cultural_heritage_symbology_layer)
                    except:
                        pass

                    # 2
                    try:
                        damage_or_destruction_of_civilian_critical_infrastructure_name = "Damage or destruction of civilian critical infrastructure"
                        damage_or_destruction_of_civilian_critical_infrastructure_query = "IncidentType = 'Damage or destruction of civilian critical infrastructure'"
                        damage_or_destruction_of_civilian_critical_infrastructure_layer = arcpy.management.MakeFeatureLayer(out_feature_class, damage_or_destruction_of_civilian_critical_infrastructure_name, damage_or_destruction_of_civilian_critical_infrastructure_query) 
                        damage_or_destruction_of_civilian_critical_infrastructure_layer_obj = damage_or_destruction_of_civilian_critical_infrastructure_layer.getOutput(0)
                        first_map.addLayer(damage_or_destruction_of_civilian_critical_infrastructure_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], damage_or_destruction_of_civilian_critical_infrastructure_symbology_layer)
                    except:
                        pass

                    # 3
                    try:
                        enslavement_name = "Enslavement"
                        enslavement_query = "IncidentType = 'Enslavement'" 
                        enslavement_layer = arcpy.management.MakeFeatureLayer(out_feature_class, enslavement_name, enslavement_query) 
                        enslavement_layer_obj = enslavement_layer.getOutput(0)
                        first_map.addLayer(enslavement_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], enslavement_symbology_layer)
                    except:
                        pass

                    # 4
                    try:
                        extrajudicial_killing_name = "Extrajudicial killing"
                        extrajudicial_killing_query = "IncidentType = 'Extrajudicial killing'"
                        extrajudicial_killing_layer = arcpy.management.MakeFeatureLayer(out_feature_class, extrajudicial_killing_name, extrajudicial_killing_query)
                        extrajudicial_killing_layer_obj = extrajudicial_killing_layer.getOutput(0)
                        first_map.addLayer(extrajudicial_killing_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], extrajudicial_killing_symbology_layer)
                    except:
                        pass

                    # 5
                    try:
                        forced_disappearance_name = "Forced disappearance"
                        forced_disappearance_query = "IncidentType = 'Forced disappearance'"
                        forced_disappearance_layer = arcpy.management.MakeFeatureLayer(out_feature_class, forced_disappearance_name, forced_disappearance_query)
                        forced_disappearance_layer_obj = forced_disappearance_layer.getOutput(0)
                        first_map.addLayer(forced_disappearance_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], forced_disappearance_symbology_layer)
                    except:
                        pass

                    # 6
                    try:
                        gender_based_violence_name = "Gender-based or other conflict-related sexual violence"
                        gender_based_violence_query = "IncidentType = 'Gender-based or other conflict-related sexual violence'"
                        gender_based_violence_layer = arcpy.management.MakeFeatureLayer(out_feature_class, gender_based_violence_name, gender_based_violence_query)
                        gender_based_violence_layer_obj = gender_based_violence_layer.getOutput(0)
                        first_map.addLayer(gender_based_violence_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], gender_based_violence_symbology_layer)
                    except:
                        pass

                    # 7
                    try:
                        human_trafficking_name = "Human trafficking"
                        human_trafficking_query = "IncidentType = 'Human trafficking'"
                        human_trafficking_layer = arcpy.management.MakeFeatureLayer(out_feature_class, human_trafficking_name, human_trafficking_query)
                        human_trafficking_layer_obj = human_trafficking_layer.getOutput(0)
                        first_map.addLayer(human_trafficking_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], human_trafficking_symbology_layer)
                    except:
                        pass

                    # 8
                    try:
                        indiscriminate_use_of_weapons_name = "Indiscriminate use of weapons"
                        indiscriminate_use_of_weapons_query = "IncidentType = 'Indiscriminate use of weapons'"
                        indiscriminate_use_of_weapons_layer = arcpy.management.MakeFeatureLayer(out_feature_class, indiscriminate_use_of_weapons_name, indiscriminate_use_of_weapons_query)
                        indiscriminate_use_of_weapons_layer_obj = indiscriminate_use_of_weapons_layer.getOutput(0)
                        first_map.addLayer(indiscriminate_use_of_weapons_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], indiscriminate_use_of_weapons_symbology_layer)
                    except:
                        pass

                    # 9
                    try:
                        kidnapping_name = "Kidnapping"
                        kidnapping_query = "IncidentType = 'Kidnapping'"
                        kidnapping_layer = arcpy.management.MakeFeatureLayer(out_feature_class, kidnapping_name, kidnapping_query)
                        kidnapping_layer_obj = kidnapping_layer.getOutput(0)
                        first_map.addLayer(kidnapping_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], kidnapping_symbology_layer)
                    except:
                        pass

                    # 10
                    try:
                        mass_excecution_name = "Mass execution"
                        mass_excecution_query = "IncidentType = 'Mass execution'"
                        mass_excecution_layer = arcpy.management.MakeFeatureLayer(out_feature_class, mass_excecution_name, mass_excecution_query)
                        mass_excecution_layer_obj = mass_excecution_layer.getOutput(0)
                        first_map.addLayer(mass_excecution_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], mass_excecution_symbology_layer)
                    except:
                        pass

                    # 11
                    try:
                        military_operations_name = "Military operations (battle, shelling)"
                        military_operations_query = "IncidentType = 'Military operations (battle, shelling)'"
                        military_operations_layer = arcpy.management.MakeFeatureLayer(out_feature_class, military_operations_name, military_operations_query)
                        military_operations_layer_obj = military_operations_layer.getOutput(0)
                        first_map.addLayer(military_operations_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], military_operations_symbology_layer)
                    except:
                        pass

                    # 12
                    try:
                        movement_of_military_name = "Movement of military, paramilitary, or other troops and equipment"
                        movement_of_military_query = "IncidentType = 'Movement of military, paramilitary, or other troops and equipment'"
                        movement_of_military_layer = arcpy.management.MakeFeatureLayer(out_feature_class, movement_of_military_name, movement_of_military_query)
                        movement_of_military_layer_obj = movement_of_military_layer.getOutput(0)
                        first_map.addLayer(movement_of_military_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], movement_of_military_symbology_layer)
                    except:
                        pass

                    # 13
                    try:
                        persecution_name = "Persecution based on political, racial, ethnic, gender, or sexual orientation"
                        persecution_query = "IncidentType = 'Persecution based on political, racial, ethnic, gender, or sexual orientation'"
                        persecution_layer = arcpy.management.MakeFeatureLayer(out_feature_class, persecution_name, persecution_query)
                        persecution_layer_obj = persecution_layer.getOutput(0)
                        first_map.addLayer(persecution_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], persecution_symbology_layer)
                    except:
                        pass

                    # 14
                    try:
                        torture_name = "Torture or indications of torture"
                        torture_query = "IncidentType = 'Torture or indications of torture'"
                        torture_layer = arcpy.management.MakeFeatureLayer(out_feature_class, torture_name, torture_query)
                        torture_layer_obj = torture_layer.getOutput(0)
                        first_map.addLayer(torture_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], torture_symbology_layer)
                    except:
                        pass

                    # 15
                    try:
                        unlawful_detention_name = "Unlawful detention"
                        unlawful_detention_query = "IncidentType = 'Unlawful detention'"
                        unlawful_detention_layer = arcpy.management.MakeFeatureLayer(out_feature_class, unlawful_detention_name, unlawful_detention_query)
                        unlawful_detention_layer_obj = unlawful_detention_layer.getOutput(0)
                        first_map.addLayer(unlawful_detention_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], unlawful_detention_symbology_layer)
                    except:
                        pass

                    # 16
                    try:
                        violent_crackdowns_on_protesters_name = "Violent crackdowns on protesters/opponents/civil rights abuse"
                        violent_crackdowns_on_protesters_query = "IncidentType = 'Violent crackdowns on protesters/opponents/civil rights abuse'"
                        violent_crackdowns_on_protesters_layer = arcpy.management.MakeFeatureLayer(out_feature_class, violent_crackdowns_on_protesters_name, violent_crackdowns_on_protesters_query)
                        violent_crackdowns_on_protesters_layer_obj = violent_crackdowns_on_protesters_layer.getOutput(0)
                        first_map.addLayer(violent_crackdowns_on_protesters_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], violent_crackdowns_on_protesters_symbology_layer)
                    except:
                        pass

                    # 17
                    try:
                        willful_killing_of_civilians_name = "Willful killing of civilians"
                        willful_killing_of_civilians_query = "IncidentType = 'Willful killing of civilians'"
                        willful_killing_of_civilians_layer = arcpy.management.MakeFeatureLayer(out_feature_class, willful_killing_of_civilians_name, willful_killing_of_civilians_query)
                        willful_killing_of_civilians_layer_obj = willful_killing_of_civilians_layer.getOutput(0)
                        first_map.addLayer(willful_killing_of_civilians_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], willful_killing_of_civilians_symbology_layer)
                    except:
                        pass

                    new_aprx.save()



                    ### **Label Configuration**
                    #st.write("Configuring labels...")
                    try:
                        lyr = first_map.listLayers("Incidents")[0]
                        lyr.showLabels = True
                        lbl_classes = lyr.listLabelClasses()
                        if not lbl_classes:
                            st.warning("No label classes found in the 'Incidents' layer.")
                        else:
                            for lblClass in lbl_classes:
                                lblClass.expression = "$feature.Incident_Location"
                                lblClass.showClassLabels = True  # Ensure labels are actually enabled for this class
                        l_cim = lyr.getDefinition("V2")
                        
                        for lc in l_cim.labelClasses:
                            lc.textSymbol.symbol.color = 'white'
                            lc.textSymbol.symbol.fontSize = 3
                            lc.textSymbol.symbol.haloColor = 'black'
                            lc.textSymbol.symbol.haloSize = 1

                        lyr.setDefinition(l_cim)
                        lyr.showLabels = True
                        #st.write("Labels configured successfully.")
                    except Exception as e:
                        st.warning(f"Label configuration failed: {e}")

                    arcpy.FeatureClassToFeatureClass_conversion(
                        "Incidents", new_aprx.defaultGeodatabase, "Incidents_Copy"
                    )

                    incidents_copy_layer_path = os.path.join(gdb_path, "Incidents_Copy")

                    ### **Density-Based Clustering**
                    st.write("Checking for density-based clusters...")
                    in_feature_class = "Incidents_Copy"  # Feature class has to be consistent with the rest of the code
                    method = "DBSCAN"
                    search_distance = "10000 Meters"  # 10km
                    min_features_cluster = 2

                    # Add a new field to indicate cluster presence
                    try:
                        field_name = "showMap"
                        field_type = "SHORT"
                        arcpy.AddField_management(in_feature_class, field_name, field_type)
                    except Exception as e: st.warning(e)

                    new_aprx.save()

                    try:
                        # Run the DensityBasedClustering tool and capture the output feature class
                        output_clusters = arcpy.stats.DensityBasedClustering(
                            in_features=in_feature_class,
                            cluster_method=method,
                            search_distance=search_distance,
                            min_features_cluster=min_features_cluster,
                        )
                        st.success("Density-based clustering completed successfully.")

                        arcpy.env.workspace = new_aprx.defaultGeodatabase

                        # Initialize the new field with default value of 0
                        with arcpy.da.UpdateCursor(in_feature_class, ["showMap"]) as cursor:
                            for row in cursor:
                                row[0] = 0  # Default value is 0
                                cursor.updateRow(row)

                        #st.write(f"'showMap' field initialized with default value '0'.")

                        # Update 'showMap' field based on the presence of clusters
                        with arcpy.da.UpdateCursor(in_feature_class, ["OBJECTID", "showMap"]) as cursor:
                            cluster_ids = [f[0] for f in arcpy.da.SearchCursor(output_clusters, ["OBJECTID"])]
                            for row in cursor:
                                if row[0] in cluster_ids:
                                    row[1] = 1  # Set showMap to 1 if the point is part of a cluster
                                    cursor.updateRow(row)

                        #st.write("'showMap' field updated successfully.")

                        # Identify unique clusters
                        output_feature_class = output_clusters[0]
                        cluster_field = "CLUSTER_ID"
                        unique_clusters = set(
                            [row[0] for row in arcpy.da.SearchCursor(output_feature_class, [cluster_field]) if row[0] != -1]
                        )
                        st.write(f"Unique Clusters Identified: {unique_clusters}")                                 

                    except arcpy.ExecuteError as e:
                        st.warning("No clusters found. Skipping further cluster processing...")
                        output_clusters = None  # Handle gracefully if clustering fails
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        output_clusters = None


                    # ADD THIS CHECK
                    if not unique_clusters:
                        # If no valid clusters (all are -1), treat as “no cluster found”
                        st.warning("No valid clusters found. Skipping cluster inset layout.")
                        output_clusters = None  # so the if-block for insets is skipped

                    ### **Layout and PNG Export**
                    if output_clusters:
                        st.write("Generating map...")
                        st.write("inset map generation...")
                        #st.write("Creating layout with insets...")
                        def MakeRec_LL(llx, lly, w, h):
                            xyRecList = [[llx, lly], [llx, lly + h], [llx + w, lly + h], [llx + w, lly], [llx, lly]]
                            array = arcpy.Array([arcpy.Point(*coords) for coords in xyRecList])
                            rec = arcpy.Polygon(array)
                            return rec

                        def _circle(radius, xc, yc, theta=1, clockwise=True):
                            """Produce a circle/ellipse depending on parameters."""
                            angles = np.deg2rad(np.arange(180.0, -180.0 - theta, step=-theta)) if clockwise else np.deg2rad(
                                np.arange(-180.0, 180.0 + theta, step=theta))
                            x_s = radius * np.cos(angles)
                            y_s = radius * np.sin(angles)
                            pnts = np.c_[x_s, y_s] + [xc, yc]
                            return [arcpy.Point(*coords) for coords in pnts]

                        def create_layout_with_insets(aprx, map_name, map_title, unique_clusters, full_feature_class_path, out_feature_class):
                            p = aprx
                            lyt = p.createLayout(17, 11, 'INCH')
                            m = p.listMaps(map_name)[0]

                            map_frame_width, map_frame_height = 14, 8
                            map_frame_x = (17 - map_frame_width) / 2
                            map_frame_y = (11 - map_frame_height) / 2

                            mf = lyt.createMapFrame(
                                MakeRec_LL(map_frame_x, map_frame_y, map_frame_width, map_frame_height),
                                m,
                                "New Map Frame"
                            )

                            # Scale bar
                            scale_bar_x = map_frame_x + (map_frame_width - 7) / 2
                            scale_bar_y = map_frame_y - 1
                            sbEnv = MakeRec_LL(scale_bar_x, scale_bar_y, 2.5, 0.3)
                            sbName = 'Scale Line 1 Metric'
                            sbStyle = p.listStyleItems('ArcGIS 2D', 'Scale_bar', sbName)[0]
                            sb = lyt.createMapSurroundElement(sbEnv, 'Scale_bar', mf, sbStyle, 'New Scale Bar')
                            sb.elementWidth = 3.5  # Adjust the width to make the scale bar larger

                            # North arrow
                            north_arrow_x = 17 - 0.3  # Adjusting for the larger map size
                            north_arrow_y = 11 - 0.5  # Slightly lowering the compass for visual balance
                            naStyle = p.listStyleItems('ArcGIS 2D', 'North_Arrow', 'ArcGIS North 2')[0]
                            na = lyt.createMapSurroundElement(arcpy.Point(north_arrow_x, north_arrow_y), 'North_Arrow', mf, naStyle, "ArcGIS North Arrow")
                            na.elementWidth = 0.3
                            """                          
                            #textbox
                            text_box_string = "Esri, NASA, NGA, USGS, Esri, © OpenStreetMap contributors, TomTom, Garmin, Foursquare, METI/NASA, USGS, Esri, © OpenStreetMap contributors, TomTom, Garmin, Foursquare, FAO, METI/NASA, USGS, Esri, USGS, Esri, © OpenStreetMap contributors, TomTom, Garmin, FAO, NOAA, USGS"
                            text_box_x = 0.2
                            text_box_y = 0.15
                            tbox = p.createTextElement(
                                lyt,
                                arcpy.Point(text_box_x, text_box_y),
                                'POINT',
                                text_box_string,
                                6,
                                style_item=txtStyleItem
                            )
                            tbox.setAnchor('BOTTOM_LEFT_CORNER')
                            tbox.textSize = 14  # bigger font
                            """                      
                            # Title
                            txtStyleItem = p.listStyleItems('ArcGIS 2D', 'TEXT', 'Title (Serif)')[0]
                            ptTxt = p.createTextElement(
                                lyt,
                                arcpy.Point(8.5, 10),
                                'POINT',
                                map_title,
                                6,
                                style_item=txtStyleItem
                            )
                            # Centering the title above the map
                            ptTxt.setAnchor('Center_Point')
                            ptTxt.elementPositionX = 8.5  # Center of the layout
                            ptTxt.elementPositionY = 10  # Adjusted to be above the larger map frame
                            ptTxt.textSize = 26  # Adjust as necessary for visibility

                            # Inset logic
                            inset_radius = 1.4
                            inset_x = 15
                            inset_y = 4

                            for cluster_id in unique_clusters:
                                cluster_layer_name = f"Cluster_{cluster_id}_Layer"
                                cluster_query = f"CLUSTER_ID = {cluster_id}"
                                cluster_layer = arcpy.management.MakeFeatureLayer(full_feature_class_path, cluster_layer_name, cluster_query).getOutput(0)

                                inset_map_name = f"Inset_Map_{cluster_id}"
                                inset_map = aprx.createMap(inset_map_name)
                                inset_map.addLayer(cluster_layer)
                                # 1
                                try:
                                    damage_or_destruction_looting_or_theft_of_cultural_heritage_inset = arcpy.management.MakeFeatureLayer(out_feature_class, damage_or_destruction_looting_or_theft_of_cultural_heritage_layer, damage_or_destruction_looting_or_theft_of_cultural_heritage_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(damage_or_destruction_looting_or_theft_of_cultural_heritage_inset, damage_or_destruction_looting_or_theft_of_cultural_heritage_symbology_layer)
                                    inset_map.addLayer(damage_or_destruction_looting_or_theft_of_cultural_heritage_inset)    
                                except:
                                    pass

                                # 2
                                try:
                                    damage_or_destruction_of_civilian_critical_infrastructure_inset = arcpy.management.MakeFeatureLayer(out_feature_class, damage_or_destruction_of_civilian_critical_infrastructure_layer, damage_or_destruction_of_civilian_critical_infrastructure_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(damage_or_destruction_of_civilian_critical_infrastructure_inset, damage_or_destruction_of_civilian_critical_infrastructure_symbology_layer)
                                    inset_map.addLayer(damage_or_destruction_of_civilian_critical_infrastructure_inset)     
                                except:
                                    pass

                                # 3
                                try:
                                    enslavement_inset = arcpy.management.MakeFeatureLayer(out_feature_class, enslavement_layer, enslavement_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(enslavement_inset, enslavement_symbology_layer)
                                    inset_map.addLayer(enslavement_inset)
                                except:
                                    pass

                                # 4
                                try:
                                    extrajudicial_killing_inset = arcpy.management.MakeFeatureLayer(out_feature_class, extrajudicial_killing_layer, extrajudicial_killing_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(extrajudicial_killing_inset, extrajudicial_killing_symbology_layer)
                                    inset_map.addLayer(extrajudicial_killing_inset)
                                except:
                                    pass

                                # 5
                                try:
                                    forced_disappearance_inset = arcpy.management.MakeFeatureLayer(out_feature_class, forced_disappearance_layer, forced_disappearance_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(forced_disappearance_inset, forced_disappearance_symbology_layer)
                                    inset_map.addLayer(forced_disappearance_inset) 
                                except:
                                    pass

                                # 6
                                try:
                                    gender_based_violence_inset = arcpy.management.MakeFeatureLayer(out_feature_class, gender_based_violence_layer, gender_based_violence_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(gender_based_violence_inset, gender_based_violence_symbology_layer)
                                    inset_map.addLayer(gender_based_violence_inset) 
                                except:
                                    pass

                                # 7
                                try:
                                    human_trafficking_inset = arcpy.management.MakeFeatureLayer(out_feature_class, human_trafficking_layer, human_trafficking_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(human_trafficking_inset, human_trafficking_symbology_layer)
                                    inset_map.addLayer(human_trafficking_inset)   
                                except:
                                    pass

                                # 8
                                try:
                                    indiscriminate_use_of_weapons_inset = arcpy.management.MakeFeatureLayer(out_feature_class, indiscriminate_use_of_weapons_layer, indiscriminate_use_of_weapons_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(indiscriminate_use_of_weapons_inset, indiscriminate_use_of_weapons_symbology_layer)
                                    inset_map.addLayer(indiscriminate_use_of_weapons_inset)    
                                except:
                                    pass

                                # 9
                                try:
                                    kidnapping_inset = arcpy.management.MakeFeatureLayer(out_feature_class, kidnapping_layer, kidnapping_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(kidnapping_inset, kidnapping_symbology_layer)
                                    inset_map.addLayer(kidnapping_inset)      
                                except:
                                    pass

                                # 10
                                try:
                                    mass_excecution_inset = arcpy.management.MakeFeatureLayer(out_feature_class, mass_excecution_layer, mass_excecution_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(mass_excecution_inset, mass_excecution_symbology_layer)
                                    inset_map.addLayer(mass_excecution_inset)      
                                except:
                                    pass

                                # 11
                                try:
                                    military_operations_inset = arcpy.management.MakeFeatureLayer(out_feature_class, military_operations_layer, military_operations_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(military_operations_inset, military_operations_symbology_layer)
                                    inset_map.addLayer(military_operations_inset)      
                                except:
                                    pass

                                # 12
                                try:
                                    movement_of_military_inset = arcpy.management.MakeFeatureLayer(out_feature_class, movement_of_military_layer, movement_of_military_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(movement_of_military_inset, movement_of_military_symbology_layer)
                                    inset_map.addLayer(movement_of_military_inset)     
                                except:
                                    pass

                                # 13
                                try:
                                    persecution_inset = arcpy.management.MakeFeatureLayer(out_feature_class, persecution_layer, persecution_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(persecution_inset, persecution_symbology_layer)
                                    inset_map.addLayer(persecution_inset)      
                                except:
                                    pass

                                # 14
                                try:
                                    torture_inset = arcpy.management.MakeFeatureLayer(out_feature_class, torture_layer, torture_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(torture_inset, torture_symbology_layer)
                                    inset_map.addLayer(torture_inset)      
                                except:
                                    pass

                                # 15
                                try:
                                    unlawful_detention_inset = arcpy.management.MakeFeatureLayer(out_feature_class, unlawful_detention_layer, unlawful_detention_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(unlawful_detention_inset, unlawful_detention_symbology_layer)
                                    inset_map.addLayer(unlawful_detention_inset)     
                                except:
                                    pass

                                # 16
                                try:
                                    violent_crackdowns_on_protesters_inset = arcpy.management.MakeFeatureLayer(out_feature_class, violent_crackdowns_on_protesters_layer, violent_crackdowns_on_protesters_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(violent_crackdowns_on_protesters_inset, violent_crackdowns_on_protesters_symbology_layer)
                                    inset_map.addLayer(violent_crackdowns_on_protesters_inset)      
                                except:
                                    pass

                                # 17
                                try:
                                    willful_killing_of_civilians_inset = arcpy.management.MakeFeatureLayer(out_feature_class, willful_killing_of_civilians_layer, willful_killing_of_civilians_query).getOutput(0)
                                    arcpy.ApplySymbologyFromLayer_management(willful_killing_of_civilians_inset, willful_killing_of_civilians_symbology_layer)
                                    inset_map.addLayer(willful_killing_of_civilians_inset)       
                                except:
                                    pass         

                                circle_coords = _circle(inset_radius, inset_x, inset_y)
                                inset_polygon = arcpy.Polygon(arcpy.Array(circle_coords))
                                inset_map_frame = lyt.createMapFrame(inset_polygon, inset_map, f"Cluster_{cluster_id}_MapFrame")

                                inset_y += 2 * inset_radius + 0.5
                                if inset_y + 2 * inset_radius > 17:
                                    inset_y = 1
                                    inset_x += 2 * inset_radius + 0.5

                                new_aprx.save()

                            lyt.openView()
                            return lyt

                        # Define parameters and call the function
                        lyt = create_layout_with_insets(new_aprx, first_map.name, map_title, unique_clusters, incidents_layer_path, out_feature_class)

                        new_aprx.save()

                        # Export layout to PNG
                        output_png = os.path.join(new_project_dir, f"{png_keyphrase}.png")
                        lyt.exportToPNG(output_png, resolution=200)
                        st.success(f"Layout exported to PNG: {output_png}")
                        st.image(output_png, caption="Generated Map")

                    else:
                        #st.warning("Skipping layout creation as clustering did not complete successfully.")
                        st.write("Generating map...")
                        st.write("no inset map generation...")
                        def mapLayout(map_title):
                            def MakeRec_LL(llx, lly, w, h):
                                xyRecList = [
                                    [llx, lly],
                                    [llx, lly + h],
                                    [llx + w, lly + h],
                                    [llx + w, lly],
                                    [llx, lly]
                                ]
                                array = arcpy.Array([arcpy.Point(*coords) for coords in xyRecList])
                                rec = arcpy.Polygon(array)
                                return rec

                            p = arcpy.mp.ArcGISProject(new_aprx_path)
                            lyt = p.createLayout(17, 11, 'INCH')
                            m = p.listMaps("Map")[0]

                            # Adjust the map frame to be larger
                            map_frame_width, map_frame_height = 14, 8  # Adjusted to make the map larger
                            map_frame_x = (17 - map_frame_width) / 2
                            map_frame_y = (11 - map_frame_height) / 2   # Adjusted for the larger map size

                            mf = lyt.createMapFrame(
                                MakeRec_LL(map_frame_x, map_frame_y, map_frame_width, map_frame_height),
                                m,
                                "New Map Frame"
                            )

                            #SCALEBAR
                            scale_bar_width = 2.5
                            scale_bar_height = 0.3
                            scale_bar_x = map_frame_x + 0.2
                            scale_bar_y = map_frame_y - 0.6
                            sbEnv = MakeRec_LL(scale_bar_x, scale_bar_y, scale_bar_width, scale_bar_height)

                            sbName = 'Scale Line 1 Metric'
                            sbStyle = p.listStyleItems('ArcGIS 2D', 'Scale_bar', sbName)[0]
                            sb = lyt.createMapSurroundElement(sbEnv, 'Scale_bar', mf, sbStyle, 'My Scale Bar')
                            sb.elementWidth = 3.5


                            # North arrow
                            north_arrow_x = 17 - 0.3  # Adjusting for the larger map size
                            north_arrow_y = 11 - 0.5  # Slightly lowering the compass for visual balance
                            naStyle = p.listStyleItems('ArcGIS 2D', 'North_Arrow', 'ArcGIS North 2')[0]
                            na = lyt.createMapSurroundElement(arcpy.Point(north_arrow_x, north_arrow_y), 'North_Arrow', mf, naStyle, "ArcGIS North Arrow")
                            na.elementWidth = 0.3

                            # Create a title, ensuring it's centered above the map
                            txtStyleItem = p.listStyleItems('ArcGIS 2D', 'TEXT', 'Title (Serif)')[0]
                            ptTxt = p.createTextElement(
                                lyt,
                                arcpy.Point(8.5, 10),
                                'POINT',
                                map_title,
                                6,
                                style_item=txtStyleItem
                            )
                            # Centering the title above the map
                            ptTxt.setAnchor('Center_Point')
                            ptTxt.elementPositionX = 8.5  # Center of the layout
                            ptTxt.elementPositionY = 10  # Adjusted to be above the larger map frame
                            ptTxt.textSize = 26  # Adjust as necessary for visibility

                            """
                            text_box_string = "Esri, NASA, NGA, USGS, Esri, © OpenStreetMap contributors, TomTom, Garmin, Foursquare, METI/NASA, USGS, Esri, © OpenStreetMap contributors, TomTom, Garmin, Foursquare, FAO, METI/NASA, USGS, Esri, USGS, Esri, © OpenStreetMap contributors, TomTom, Garmin, FAO, NOAA, USGS"
                            text_box_x = 0.2
                            text_box_y = 0.15
                            tbox = p.createTextElement(
                                lyt,
                                arcpy.Point(text_box_x, text_box_y),
                                'POINT',
                                text_box_string,
                                6,
                                style_item=txtStyleItem
                            )
                            tbox.setAnchor('BOTTOM_LEFT_CORNER')
                            tbox.textSize = 10  # bigger font
                            """
                            # Export to PNG
                            output_png = new_project_dir + "/" + png_keyphrase + ".png"
                            #output_png = os.path.join(new_project_dir, f"{png_keyphrase}.png")
                            lyt.exportToPNG(output_png, resolution=900)
                            return output_png

                        output_png = mapLayout(map_title)
                        st.success("Map generated successfully!")
                        st.image(output_png, caption="Generated Map")
                        
                    # Provide the correct file for download
                    with open(output_png, "rb") as file:
                        st.download_button(
                            label="Download Map",
                            data=file,
                            file_name=os.path.basename(output_png),
                            mime="image/png",
                        )
            except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
