# Standard Library Imports
import os
import re

# Third-Party Imports
import pandas as pd
import openai
from openai import OpenAI
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext
import faiss
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
import arcpy
import numpy as np
import streamlit as st
from PIL import Image

# Configuration
Image.MAX_IMAGE_PIXELS = None
load_dotenv()

os.environ["OPENAI_API_KEY"] = 'lm-studio'
os.environ['OPENAI_API_BASE'] = 'http://localhost:1234/v1'
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

opencage_api_key = os.getenv("OPENCAGE_API_KEY")
geocoder = OpenCageGeocode(opencage_api_key)

# Streamlit App
st.title("Automapping Tool")
st.write("Generate a thematic map from your article text.")
article_input = st.text_area("Article Text", height=300, placeholder="Enter your article here...")

if st.button("Generate Map"):
    if not article_input.strip():
        st.error("Error: The input cannot be empty.")
    else:
        stop_process = st.button("Stop Process")
        if stop_process:
            st.warning("Process interrupted by user.")
            st.stop()

        relevance_result = None
        relevance_error = None

        with st.spinner('Checking content relevance...'):
            try:
                # Create a small FAISS index and use LLM to check if input is relevant
                d = 768
                faiss_index = faiss.IndexFlatL2(d)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents([Document(text=article_input)], storage_context=storage_context)

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
                # Project paths
                new_project_dir = r"E:/Yahya Masri/ArcGIS_Projects/test_st1"
                new_aprx_path = r"E:/Yahya Masri/ArcGIS_Projects/test_st1/test_st1.aprx"
                layer_path = "E:/Yahya Masri/automapping_files/sdn_adm_cbs_nic_ssa_20200831_shp/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp"
                template_project_path = r"E:/Yahya Masri/ArcGIS_Projects/BlankTemplate/BlankTemplate.aprx"
                os.makedirs(os.path.dirname(new_aprx_path), exist_ok=True)

                # Create new project from template
                aprx = arcpy.mp.ArcGISProject(template_project_path)
                aprx.saveACopy(new_aprx_path)
                new_aprx = arcpy.mp.ArcGISProject(new_aprx_path)

                # Create or use existing .gdb
                gdb_name = "test_st1.gdb"
                gdb_path = os.path.join(new_project_dir, gdb_name)
                if not os.path.exists(gdb_path):
                    arcpy.CreateFileGDB_management(new_project_dir, gdb_name)
                new_aprx.defaultGeodatabase = gdb_path

                # Add the shapefile layer to map
                try:
                    first_map = new_aprx.listMaps()[0]
                    sudan_shape = first_map.addDataFromPath(layer_path)
                except IndexError:
                    st.error("No maps found in the template project. Could not add layer.")

                # Symbolize and label the shapefile
                try:
                    for lyr in first_map.listLayers():
                        if lyr.name == sudan_shape.name:
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

                # Save article to CSV
                csv_name = "Auto.csv"
                csv_file_path = os.path.join(new_project_dir, csv_name).replace("\\", "/")
                article_text = article_input
                data = {'Text': [article_text]}
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

                df['Extracted_Meta_data'] = None
                text_list = [article_text]
                documents = [Document(text=t) for t in text_list]

                # Metadata extraction
                d = 768
                faiss_index = faiss.IndexFlatL2(d)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

                query = """
                Identify the geographic location(s) and the time(s) where the incident(s) occurred
                in the format of "neighborhood, state, country | mm/dd/yyyy". 
                If there are multiple incidents, list each incident with its corresponding location and date on a separate
                line without any explanation.

                Furthermore, your task is to also classify each news incident into one of the following categories. The output should only contain the category name without any explanation.
                The category should be placed in the format of "incident location | incident date | incident category":

                Unlawful detention- Refers to refers to the act of detaining or confining an individual without legal justification or due process. For example, if protesters are arrested and detained without legal basis during peaceful demonstrations, with no access to legal representation, this would be considered unlawful detention. 
                Human trafficking- Refers to the act of recruiting, transporting, transferring, harboring, or receiving individuals through force, fraud, coercion, or other forms of deception for the purpose of exploitation. Exploitation can take many forms, including forced labor, sexual exploitation, slavery, servitude, or the removal of organs. It is considered a severe violation of human rights and is illegal under international and domestic laws. If an incident is considered Human trafficking it would also be considered as a War crime. 
                Enslavement- refers to the act of exercising ownership or control over another person, treating them as property, and depriving them of their freedom. It often involves forcing individuals to perform labor or services under coercion, violence, or the threat of punishment. If an incident is considered Enslavement, it would also be considered as a War crime. 
                Willful killing of civilians- Refers to the intentional killing of civilians who are not directly participating in hostilities, with full knowledge of their noncombatant status. This includes acts like massacres, executions, or deliberate bombings of civilian sites such as homes, schools, or hospitals, where the clear intent is to cause death. For example, a military unit massacring the residents of a village. 
                Mass execution- Refers to the deliberate killing of a large scale number of individuals, often carried out by state or non-state actors as part of systematic persecution, acts of war, or punitive measures. The victims are typically selected based on political, ethnic, religious, or social affiliations, and the killings are often premeditated and organized. If an incident is considered Mass execution, it would also be considered as a War crime. 
                Kidnapping- Refers to the unlawful and intentional abduction, confinement, or holding of an individual against their will, often for a specific purpose such as extortion, ransom, political leverage, forced labor, or exploitation. It is a serious crime and violates the individual's right to freedom and security. 
                Extrajudicial killing- Refers to the killing of a person without any legal process, such as arrest, trial, or sentencing. It is carried out outside the law, often by state agents or with their approval. 
                Forced disappearance- Refers the act of abducting or detaining a person against their will, followed by a refusal to disclose their fate or whereabouts. This leaves the victim outside the protection of the law and often causes anguish to their family and community.
                Damage or destruction of civilian critical infrastructure- Refers to the reckless harm, sabotage, or destruction of essential facilities, systems, or services necessary for the well-being, safety, and survival of civilian populations. This includes infrastructure such as hospitals, water supplies, power grids, schools, transportation systems, and communication networks. 
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
                lines = response.response.strip().split('\n')

                if not lines:
                    st.error("Error: LLM response is empty. Ensure the model is running and returning valid outputs.")
                    st.stop()

                article_text = df.loc[0, "Text"]
                new_rows = []
                for line in lines:
                    line = line.strip()
                    if "|" in line:
                        new_rows.append({
                            "Text": article_input,
                            "Extracted_Meta_data": line
                        })
                if new_rows:
                    df = pd.DataFrame(new_rows)
                else:
                    st.error("Error: No extracted metadata. LLM might not be returning expected format.")
                    st.stop()

                if "Extracted_Meta_data" not in df.columns or df["Extracted_Meta_data"].isna().all():
                    st.error("Error: 'Extracted_Meta_data' is missing or empty.")
                    st.stop()

                df['Geo_Meta_data'] = None
                df['IncidentType'] = None

                def split_metadata(row):
                    try:
                        parts = row.rsplit("|", 1)
                        if len(parts) == 2:
                            return parts[0].strip(), parts[1].strip()
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

                def split_geo_meta_data(df_in):
                    if 'Geo_Meta_data' in df_in.columns:
                        df_in[['Incident Location', 'Incident Date']] = df_in['Geo_Meta_data'].str.split('|', expand=True)
                        df_in['Incident Location'] = df_in['Incident Location'].str.strip()
                        df_in['Incident Date'] = df_in['Incident Date'].str.strip()
                        df_in.drop(columns=['Geo_Meta_data'], inplace=True)
                    return df_in

                df = split_geo_meta_data(df)
                df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

                st.write("Validating article category classification...")
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

                for i, row in df.iterrows():
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

                # Geocoding
                st.write("Geocoding incident locations...")
                df = pd.read_csv(csv_file_path)

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

                df['Latitude'], df['Longitude'] = zip(*df['Incident Location'].apply(lambda loc: opencage_geocode(loc)))
                df.to_csv(csv_file_path, index=False)

                # Extract State from location
                def extract_state_from_sudan_format(df_in):
                    if 'Incident Location' in df_in.columns:
                        def extract_state(location):
                            try:
                                if 'Sudan' in location:
                                    before_sudan = location.rsplit('Sudan', 1)[0]
                                    return before_sudan.rsplit(',', 1)[-1].strip()
                                return None
                            except Exception as e:
                                st.error(f"Error processing location '{location}': {e}")
                                return None
                        df_in['State'] = df_in['Incident Location'].apply(extract_state)
                    return df_in

                df = extract_state_from_sudan_format(df)
                df.to_csv(csv_file_path, index=False)

                # Generate map title
                st.write("Generating map title...")
                client = OpenAI(base_url=openai.api_base, api_key=openai.api_key)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Create ONE clear, concise, and accurate map title. "
                            "Avoid redundancy and keep it brief. If a date is relevant, include it. "
                            "Do not include any extra explanation."
                        ),
                    },
                    {"role": "user", "content": article_text},
                ]
                completion = client.chat.completions.create(
                    model="model-identifier",
                    messages=messages,
                    temperature=0,
                    stream=False,
                )
                map_title = completion.choices[0].message.content.strip()

                # Generate PNG naming convention
                st.write("Generating PNG naming convention...")
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Generate a short, concise keyphrase for a Map PNG filename, "
                            "using only lowercase letters, numbers, and underscores. "
                            "No spaces, special characters, or explanations."
                        ),
                    },
                    {"role": "user", "content": article_text},
                ]
                completion = client.chat.completions.create(
                    model="model-identifier",
                    messages=messages,
                    temperature=0,
                    stream=False,
                )
                png_keyphrase = completion.choices[0].message.content.strip()
                png_keyphrase = re.sub(r'[^a-zA-Z0-9_]', '_', png_keyphrase)

                # Define symbology layer paths
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
                
                # Create feature class for incidents
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

                # Add feature class to map
                if arcpy.Exists("AllPointsLayer"):
                    arcpy.Delete_management("AllPointsLayer")

                try:
                    incidents_layer_path = os.path.join(gdb_path, out_feature_class)
                    first_map.addDataFromPath(incidents_layer_path)
                except Exception as e:
                    st.error(f"Error adding feature class to the map: {e}")

                new_aprx.save()

                # 1
                try:
                    damage_or_destruction_looting_or_theft_of_cultural_heritage_name = "Damage or destruction, looting, or theft of cultural heritage"
                    damage_or_destruction_looting_or_theft_of_cultural_heritage_query = "IncidentType ='Damage or destruction, looting, or theft of cultural heritage'"            
                    damage_or_destruction_looting_or_theft_of_cultural_heritage_layer = arcpy.management.MakeFeatureLayer(out_feature_class, damage_or_destruction_looting_or_theft_of_cultural_heritage_name, damage_or_destruction_looting_or_theft_of_cultural_heritage_query)
                    damage_or_destruction_looting_or_theft_of_cultural_heritage_layer_obj = damage_or_destruction_looting_or_theft_of_cultural_heritage_layer.getOutput(0)
                    damage_or_destruction_looting_or_theft_of_cultural_heritage_result = arcpy.GetCount_management(damage_or_destruction_looting_or_theft_of_cultural_heritage_layer)
                    damage_or_destruction_looting_or_theft_of_cultural_heritage_count = int(damage_or_destruction_looting_or_theft_of_cultural_heritage_result.getOutput(0))
                    if damage_or_destruction_looting_or_theft_of_cultural_heritage_count > 0:
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
                    damage_or_destruction_of_civilian_critical_infrastructure_result = arcpy.GetCount_management(damage_or_destruction_of_civilian_critical_infrastructure_layer)
                    damage_or_destruction_of_civilian_critical_infrastructure_count = int(damage_or_destruction_of_civilian_critical_infrastructure_result.getOutput(0))
                    if damage_or_destruction_of_civilian_critical_infrastructure_count > 0:
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
                    enslavement_result = arcpy.GetCount_management(enslavement_layer)
                    enslavement_count = int(enslavement_result.getOutput(0))
                    if enslavement_count > 0:
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
                    extrajudicial_killing_result = arcpy.GetCount_management(extrajudicial_killing_layer)
                    extrajudicial_killing_count = int(extrajudicial_killing_result.getOutput(0))
                    if extrajudicial_killing_count > 0:
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
                    forced_disappearance_result = arcpy.GetCount_management(forced_disappearance_layer)
                    forced_disappearance_count = int(forced_disappearance_result.getOutput(0))
                    if forced_disappearance_count > 0:
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
                    gender_based_violence_result = arcpy.GetCount_management(gender_based_violence_layer)
                    gender_based_violence_count = int(gender_based_violence_result.getOutput(0))
                    if gender_based_violence_count > 0:
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
                    human_trafficking_result = arcpy.GetCount_management(human_trafficking_layer)
                    human_trafficking_count = int(human_trafficking_result.getOutput(0))
                    if human_trafficking_count > 0:
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
                    indiscriminate_use_of_weapons_result = arcpy.GetCount_management(indiscriminate_use_of_weapons_layer)
                    indiscriminate_use_of_weapons_count = int(indiscriminate_use_of_weapons_result.getOutput(0))
                    if indiscriminate_use_of_weapons_count > 0:
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
                    kidnapping_result = arcpy.GetCount_management(kidnapping_layer)
                    kidnapping_count = int(kidnapping_result.getOutput(0))
                    if kidnapping_count > 0:
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
                    mass_excecution_result = arcpy.GetCount_management(mass_excecution_layer)
                    mass_excecution_count = int(mass_excecution_result.getOutput(0))
                    if mass_excecution_count > 0:
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
                    military_operations_result = arcpy.GetCount_management(military_operations_layer)
                    military_operations_count = int(military_operations_result.getOutput(0))
                    if military_operations_count > 0:
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
                    movement_of_military_result = arcpy.GetCount_management(movement_of_military_layer)
                    movement_of_military_count = int(movement_of_military_result.getOutput(0))
                    if movement_of_military_count > 0:
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
                    persecution_result = arcpy.GetCount_management(persecution_layer)
                    persecution_count = int(persecution_result.getOutput(0))
                    if persecution_count > 0:
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
                    torture_result = arcpy.GetCount_management(torture_layer)
                    torture_count = int(torture_result.getOutput(0))
                    if torture_count > 0:
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
                    unlawful_detention_result = arcpy.GetCount_management(unlawful_detention_layer)
                    unlawful_detention_count = int(unlawful_detention_result.getOutput(0))
                    if unlawful_detention_count > 0:
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
                    violent_crackdowns_on_protesters_result = arcpy.GetCount_management(violent_crackdowns_on_protesters_layer)
                    violent_crackdowns_on_protesters_count = int(violent_crackdowns_on_protesters_result.getOutput(0))
                    if violent_crackdowns_on_protesters_count > 0:
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
                    willful_killing_of_civilians_result = arcpy.GetCount_management(willful_killing_of_civilians_layer)
                    willful_killing_of_civilians_count = int(willful_killing_of_civilians_result.getOutput(0))
                    if willful_killing_of_civilians_count > 0:
                        first_map.addLayer(willful_killing_of_civilians_layer_obj)
                        arcpy.ApplySymbologyFromLayer_management(first_map.listLayers()[0], willful_killing_of_civilians_symbology_layer)
                except:
                    pass

                new_aprx.save()

                # Configure labels
                try:
                    lyr = first_map.listLayers("Incidents")[0]
                    lyr.showLabels = True
                    lbl_classes = lyr.listLabelClasses()
                    if lbl_classes:
                        for lblClass in lbl_classes:
                            lblClass.expression = "$feature.Incident_Location"
                            lblClass.showClassLabels = True
                    l_cim = lyr.getDefinition("V2")
                    for lc in l_cim.labelClasses:
                        lc.textSymbol.symbol.color = 'white'
                        lc.textSymbol.symbol.fontSize = 3
                        lc.textSymbol.symbol.haloColor = 'black'
                        lc.textSymbol.symbol.haloSize = 1
                    lyr.setDefinition(l_cim)
                    lyr.showLabels = True
                except Exception as e:
                    st.warning(f"Label configuration failed: {e}")

                # Create copy for clustering
                arcpy.FeatureClassToFeatureClass_conversion("Incidents", new_aprx.defaultGeodatabase, "Incidents_Copy")
                incidents_copy_layer_path = os.path.join(gdb_path, "Incidents_Copy")

                # Density-based clustering
                st.write("Checking for density-based clusters...")
                in_feature_class = "Incidents_Copy"
                method = "DBSCAN"
                search_distance = "10000 Meters"
                min_features_cluster = 2

                try:
                    arcpy.AddField_management(in_feature_class, "showMap", "SHORT")
                except Exception as e:
                    st.warning(e)

                new_aprx.save()
                output_clusters = None
                unique_clusters = None

                try:
                    output_clusters = arcpy.stats.DensityBasedClustering(
                        in_features=in_feature_class,
                        cluster_method=method,
                        search_distance=search_distance,
                        min_features_cluster=min_features_cluster,
                    )
                    st.success("Density-based clustering completed successfully.")
                    arcpy.env.workspace = new_aprx.defaultGeodatabase

                    with arcpy.da.UpdateCursor(in_feature_class, ["showMap"]) as cursor:
                        for row in cursor:
                            row[0] = 0
                            cursor.updateRow(row)

                    cluster_ids = [f[0] for f in arcpy.da.SearchCursor(output_clusters, ["OBJECTID"])]

                    with arcpy.da.UpdateCursor(in_feature_class, ["OBJECTID", "showMap"]) as cursor:
                        for row in cursor:
                            if row[0] in cluster_ids:
                                row[1] = 1
                                cursor.updateRow(row)

                    output_feature_class = output_clusters[0]
                    cluster_field = "CLUSTER_ID"
                    unique_clusters = set(
                        row[0] for row in arcpy.da.SearchCursor(output_feature_class, [cluster_field]) if row[0] != -1
                    )
                    st.write(f"Unique Clusters Identified: {unique_clusters}")
                except arcpy.ExecuteError:
                    st.warning("No clusters found. Skipping further cluster processing...")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                if not unique_clusters:
                    # No valid clusters found
                    st.write("No valid clusters found. Skipping cluster inset layout.")
                    output_clusters = None

                # Layout and PNG export
                if output_clusters:
                    st.write("Generating map...")

                    def MakeRec_LL(llx, lly, w, h):
                        xyRecList = [[llx, lly], [llx, lly + h], [llx + w, lly + h], [llx + w, lly], [llx, lly]]
                        array = arcpy.Array([arcpy.Point(*coords) for coords in xyRecList])
                        return arcpy.Polygon(array)

                    def _circle(radius, xc, yc, theta=1, clockwise=True):
                        angles = (
                            np.deg2rad(np.arange(180.0, -180.0 - theta, step=-theta))
                            if clockwise
                            else np.deg2rad(np.arange(-180.0, 180.0 + theta, step=theta))
                        )
                        x_s = radius * np.cos(angles)
                        y_s = radius * np.sin(angles)
                        pnts = np.c_[x_s, y_s] + [xc, yc]
                        return [arcpy.Point(*coords) for coords in pnts]

                    def create_layout_with_insets(aprx_obj, map_name, title_str, clusters, full_fc_path, fc_name):
                        p = aprx_obj
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

                        # Adjust map extent and scale
                        if mf.getLayerExtent():
                            print("Adjusting map extent for clustering...")
                            mf.camera.setExtent(mf.getLayerExtent())  # Fit to all incident points
                            mf.camera.scale = max(mf.camera.scale, 2500000)  # Ensure it's zoomed out enough
                            print(f"Map scale adjusted to: {mf.camera.scale}")
                        else:
                            print("Warning: No features found in the map frame.")

                        # Scale bar
                        scale_bar_x = map_frame_x + (map_frame_width - 7) / 2
                        scale_bar_y = map_frame_y - 1
                        sbEnv = MakeRec_LL(scale_bar_x, scale_bar_y, 2.5, 0.3)
                        sbName = 'Scale Line 1 Metric'
                        sbStyle = p.listStyleItems('ArcGIS 2D', 'Scale_bar', sbName)[0]
                        sb = lyt.createMapSurroundElement(sbEnv, 'Scale_bar', mf, sbStyle, 'New Scale Bar')
                        sb.elementWidth = 3.5

                        # North arrow
                        north_arrow_x = 17 - 0.3
                        north_arrow_y = 11 - 0.5
                        naStyle = p.listStyleItems('ArcGIS 2D', 'North_Arrow', 'ArcGIS North 2')[0]
                        na = lyt.createMapSurroundElement(arcpy.Point(north_arrow_x, north_arrow_y), 'North_Arrow', mf, naStyle, "ArcGIS North Arrow")
                        na.elementWidth = 0.3

                        # Fix Scale Bar Issue
                        scalebars = lyt.listElements("MAPSURROUND_ELEMENT") 
                        sb_fixed = None
                        for s in scalebars:
                            if s.name == "New Scale Bar":  # Ensure it's the same scale bar
                                sb_fixed = s
                                break

                        if sb_fixed:
                            print(f"Scalebar Found: {sb_fixed.name}")

                            # Ensure the scalebar is linked to the correct map frame
                            if sb_fixed.mapFrame.name != mf.name:
                                sb_fixed.mapFrame = mf
                                print(f"Scalebar linked to map frame: {mf.name}")

                            # Force an update to the scale bar
                            sb_fixed.elementWidth = sb_fixed.elementWidth  # This forces an update on the scale bar

                            # Check and update the spatial reference of the map
                            if mf.map.spatialReference is None or mf.map.spatialReference.name == "Unknown":
                                print("Spatial reference is missing, setting it to WGS 1984")
                                mf.map.spatialReference = arcpy.SpatialReference(4326)  # WGS 1984

                            print("Scale bar updated successfully!")
                        else:
                            print("Error: Scale bar not found.")

                        # Title
                        txtStyleItem = p.listStyleItems('ArcGIS 2D', 'TEXT', 'Title (Serif)')[0]
                        ptTxt = p.createTextElement(
                            lyt,
                            arcpy.Point(8.5, 10),
                            'POINT',
                            title_str,
                            6,
                            style_item=txtStyleItem
                        )
                        ptTxt.setAnchor('Center_Point')
                        ptTxt.elementPositionX = 8.5
                        ptTxt.elementPositionY = 10
                        ptTxt.textSize = 26

                        # Inset maps for clusters
                        inset_radius = 1.4
                        inset_x = 15
                        inset_y = 4

                        for cluster_id in clusters:
                            cluster_layer_name = f"Cluster_{cluster_id}_Layer"
                            cluster_query = f"CLUSTER_ID = {cluster_id}"
                            cluster_layer = arcpy.management.MakeFeatureLayer(full_fc_path, cluster_layer_name, cluster_query).getOutput(0)
                            inset_map_name = f"Inset_Map_{cluster_id}"
                            inset_map = aprx_obj.createMap(inset_map_name)
                            inset_map.addLayer(cluster_layer)

                            # Additional symbology applies would go here if needed for each inset map.

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


                    lyt = create_layout_with_insets(new_aprx, first_map.name, map_title, unique_clusters, incidents_layer_path, out_feature_class)
                    new_aprx.save()
                    output_png = os.path.join(new_project_dir, f"{png_keyphrase}.png")
                    lyt.exportToPNG(output_png, resolution=200)
                    st.success(f"Layout exported to PNG: {output_png}")
                    st.image(output_png, caption="Generated Map")

                else:
                    # If no clustering
                    st.write("Generating map...")

                    def MakeRec_LL(llx, lly, w, h):
                        xyRecList = [
                            [llx, lly],
                            [llx, lly + h],
                            [llx + w, lly + h],
                            [llx + w, lly],
                            [llx, lly]
                        ]
                        array = arcpy.Array([arcpy.Point(*coords) for coords in xyRecList])
                        return arcpy.Polygon(array)

                    def mapLayout(m_title):
                        p = arcpy.mp.ArcGISProject(new_aprx_path)
                        lyt = p.createLayout(17, 11, 'INCH')
                        m = p.listMaps("Map")[0]
                        map_frame_width, map_frame_height = 13, 7.5
                        map_frame_x = ((17 - map_frame_width) / 2) - 1.5
                        map_frame_y = (11 - map_frame_height) / 2
                        mf = lyt.createMapFrame(
                            MakeRec_LL(map_frame_x, map_frame_y, map_frame_width, map_frame_height),
                            m,
                            "New Map Frame"
                        )

                        lyt = p.listLayouts('Layout')[0]
                        mf = lyt.listElements('MAPFRAME_ELEMENT', 'New Map Frame')[0]  # Ensure you reference the correct map frame

                        # Define legend position and dimensions
                        legend_x = 13.8
                        legend_y = 5
                        legend_width = 8  # Legend width
                        legend_height = 30 # Legend height

                        # Define the legend style
                        legend_style = p.listStyleItems('ArcGIS 2D', 'LEGEND', 'Legend 1')[0]

                        # Create the legend
                        legend = lyt.createMapSurroundElement(
                            arcpy.Point(legend_x, legend_y), 'LEGEND', mf, legend_style, 'Map Legend'
                        )

                        # Configure legend appearance and behavior
                        legend.elementWidth = legend_width
                        legend.elementHeight = legend_height
                        legend.fittingStrategy = 'AdjustColumnsAndFont'  
                        #legend.columnCount = 4  # Arrange items in # columns
                        legend.showTitle = False  # Hide the legend title

                        # Use CIM to refine the legend further
                        lyt_cim = lyt.getDefinition('V3')
                        for elm in lyt_cim.elements:
                            if elm.name == "Map Legend":  # Replace "Map Legend" with the actual legend name
                                # Access the legend CIM definition
                                legend_cim = elm

                                # Remove the background frame/outline by setting the frame to None
                                legend_cim.frame = None

                                # Hide subheaders and duplicate category names
                                for item in legend_cim.items:
                                    item.showHeading = False  # Remove subheaders
                                    item.showLabels = False  # Hide duplicate category names

                                # Update the definition back to the layout
                                lyt.setDefinition(lyt_cim)

                        # Filter legend items to display only relevant categories on the map
                        valid_categories_in_map = [
                            "Damage or destruction, looting, or theft of cultural heritage",
                            "Damage or destruction of civilian critical infrastructure",
                            "Enslavement",
                            "Extrajudicial killing",
                            "Forced disappearance",
                            "Gender-based or other conflict-related sexual violence",
                            "Human trafficking",
                            "Indiscriminate use of weapons",
                            "Kidnapping",
                            "Mass execution",
                            "Military operations (battle, shelling)",
                            "Movement of military, paramilitary, or other troops and equipment",
                            "Persecution based on political, racial, ethnic, gender, or sexual orientation",
                            "Torture or indications of torture",
                            "Unlawful detention",
                            "Violent crackdowns on protesters/opponents/civil rights abuse",
                            "Willful killing of civilians"
                        ]

                        # Filter and remove items not relevant to the map
                        for item in legend.items:
                            print(f"Legend Item: {item.name}")  # Debugging output
                            if item.name not in valid_categories_in_map:
                                legend.removeItem(item)

                        # Position the legend properly in the layout
                        legend.elementPositionX = legend_x  # Set the X position
                        legend.elementPositionY = legend_y  # Set the Y position

                        #scale bar
                        scale_bar_width = 2.5
                        scale_bar_height = 0.3
                        scale_bar_x = map_frame_x + 0.2
                        scale_bar_y = map_frame_y - 0.6
                        sbEnv = MakeRec_LL(scale_bar_x, scale_bar_y, scale_bar_width, scale_bar_height)
                        sbName = 'Scale Line 1 Metric'
                        sbStyle = p.listStyleItems('ArcGIS 2D', 'Scale_bar', sbName)[0]
                        sb = lyt.createMapSurroundElement(sbEnv, 'Scale_bar', mf, sbStyle, 'My Scale Bar')
                        sb.elementWidth = 3.5
                        
                        #north arrow
                        north_arrow_x = 17 - 0.3
                        north_arrow_y = 11 - 0.5
                        naStyle = p.listStyleItems('ArcGIS 2D', 'North_Arrow', 'ArcGIS North 2')[0]
                        na = lyt.createMapSurroundElement(arcpy.Point(north_arrow_x, north_arrow_y), 'North_Arrow', mf, naStyle, "ArcGIS North Arrow")
                        na.elementWidth = 0.3

                        txtStyleItem = p.listStyleItems('ArcGIS 2D', 'TEXT', 'Title (Serif)')[0]
                        ptTxt = p.createTextElement(
                            lyt,
                            arcpy.Point(8.5, 10),
                            'POINT',
                            m_title,
                            6,
                            style_item=txtStyleItem
                        )
                        ptTxt.setAnchor('Center_Point')
                        ptTxt.elementPositionX = map_frame_x + (map_frame_width / 2)
                        ptTxt.elementPositionY = 10
                        ptTxt.textSize = 26

                        # Fix Scale Bar Issue
                        scalebars = lyt.listElements("MAPSURROUND_ELEMENT") 
                        sb_fixed = None
                        for s in scalebars:
                            if s.name == "My Scale Bar":  # Ensure it's the same scale bar
                                sb_fixed = s
                                break

                        if sb_fixed:
                            print(f"Scalebar Found: {sb_fixed.name}")

                            # Ensure the scalebar is linked to the correct map frame
                            if sb_fixed.mapFrame.name != mf.name:
                                sb_fixed.mapFrame = mf
                                print(f"Scalebar linked to map frame: {mf.name}")

                            # Force an update to the scale bar
                            sb_fixed.elementWidth = sb_fixed.elementWidth  # This forces an update on the scale bar

                            # Check and update the spatial reference of the map
                            if mf.map.spatialReference is None or mf.map.spatialReference.name == "Unknown":
                                print("Spatial reference is missing, setting it to WGS 1984")
                                mf.map.spatialReference = arcpy.SpatialReference(4326)  # WGS 1984

                            # Ensure the map scale is set to a reasonable value
                            mf.camera.setExtent(mf.map.defaultCamera.getExtent())
                            mf.camera.scale = 9000000  # Adjust the scale as needed
                            print(f"Map scale set to: {mf.camera.scale}")

                            print("Scale bar updated successfully!")
                        else:
                            print("Error: Scale bar not found.")

                        output_png_file = os.path.join(new_project_dir, f"{png_keyphrase}.png")
                        lyt.exportToPNG(output_png_file, resolution=900)
                        return output_png_file

                    output_png = mapLayout(map_title)
                    st.success("Map generated successfully!")
                    st.image(output_png, caption="Generated Map")

                with open(output_png, "rb") as file:
                    st.download_button(
                        label="Download Map",
                        data=file,
                        file_name=os.path.basename(output_png),
                        mime="image/png",
                    )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
