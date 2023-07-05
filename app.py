import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import base64
from utils.functions import generate_data


st.set_page_config(page_title="Classifier Playground", page_icon="", layout="wide")

hide_st_style = """
                <style>
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__1S137 {display: none !important;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = pathlib.Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header = """
    <style>
        :root {{
            --base-font-size: 1vw;  /* Define your base font size here */
        }}

        .header {{
            font-family:sans-serif; 
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-image: url('data:image/png;base64,{}');
            background-repeat: no-repeat;
            background-size: cover;
            background-position: center;
            filter: brightness(0.9) saturate(0.8);
            opacity: 1;
            color: #FAFAFA;
            text-align: left;
            padding: 0.4em;  /* Convert 10px to em units */
            z-index: 1;
            display: flex;
            align-items: center;
        }}
        .middle-column {{
            display: flex;
            align-items: center;
            justify-content: center;
            float: center;            
            width: 100%;
            padding: 0em;  /* Convert 10px to em units */
        }}
        .middle-column img {{
            max-width: 200%;
            display: inline-block;
            vertical-align: middle;
        }}
        .button-div {{
            display: flex;
            justify-content: flex-start;
            align-items: center;
            width: 40%;
            gap: 2em;  /* Convert 20px to em units */
        }}
        .button {{
            background-color: #002147;
            border-color: #FAFAFA;
            border-width: 0.1875em;  /* Convert 3px to em units */
            border-radius: 0.1875em;  /* Convert 3px to em units */
            color: #FAFAFA;
            padding: 0em 0em;  /* Adjust padding as per your preference */
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 0.8em;  /* Convert 16px to em units */
            margin: 0em;  /* Adjust margin as per your preference */
            cursor: pointer;
            width: 6.6em;
            height: 2em;
        }}
        .button:hover {{
            background-color: #76787A;
            border-color: #FAFAFA;
            color: #FAFAFA;
        }}
        .clear {{
            clear: both;
        }}
        body {{
            margin-top: 1px;
            font-size: var(--base-font-size);  /* Set the base font size */
        }}
        @media screen and (max-width: 1024px) {{
        .header {{
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 0.8em;  /* Adjust padding for smaller screens */
       }}

        .middle-column {{
            width: 100%;  /* Set width to 100% for full width on smaller screens */
            justify-content: center;
            text-align: center;
            display: flex;
            align-items: center;
            float: left;
            margin-bottom: -1em;  /* Adjust margin for smaller screens */
        }}
        .middle-column img {{
            width: 30%;
            display: flex;
            align-items: center;
            justify-content: center;
            float: left;
          }}
        .button-div {{
            width: 50%;  /* Set width to 100% for full width on smaller screens */
            justify-content: center;  /* Center align items on smaller screens */
            gap: 2em;  /* Convert 20px to em units */
            margin-top: 0em;  /* Add top margin */
        }}
        .button {{
            border-width: 0.1875em;  /* Convert 3px to em units */
            border-radius: 0.1875em;  /* Convert 3px to em units */
            font-size: 1.6em;  /* Convert 16px to em units */
            width: 5em;
            height: 2em;
        }}
    }}
    </style>
    <div class="header">
        <div class="middle-column">
            <img src="data:image/png;base64,{}" class="img-fluid" alt="comrate_logo" width="20%">
        </div>
        <div class="clear"></div>
            <div class="button-div">
                <button class="button">Playground</button>
            </div>
        </div>
        <div class="clear"></div>
    </div>
"""

# Replace `image_file_path` with the actual path to your image file
image_file_path = "images/oxbrain_header_background.jpg"
with open(image_file_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

st.markdown(header.format(encoded_string, img_to_bytes("images/oxbrain_logo_trans.png")),
            unsafe_allow_html=True)







st.markdown(
    """
    <style>
     div.stButton > button:first-child {
        background-color: #002147;
        color: #FAFAFA;
        border-color: #FAFAFA;
        border-width: 3px;
        width: 5.4em;
        height: 1.8em;
        margin-top: 1.5em;
    }
    div.stButton > button:hover {
        background-color: #76787A;
        color: #002147;
        border-color: #002147;
    }
    @media (max-width: 1024px) {
    div.stButton > button {
        width: 100% !important;
        height: 10em !important;
        margin-top: -3em;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
  )
st.markdown("""
  <style>
  /* The input itself */
  div[data-baseweb="select"] > div,
  input[type=number] {
  margin-top: -3em;
  color: #FAFAFA;
  background-color: #4F5254;
  border: 0.25em solid #002147;
  font-size: 0.8em;
  height: 3em;
  }
  /* Hover effect */
  div[data-baseweb="select"] > div:hover,
  input[type=number]:hover {
  background-color: #76787A;
  }
  span.st-bj.st-cf.st-ce.st-f3.st-f4.st-af {
  font-size: 0.6em;
  }
  @media (max-width: 1024px) {
    span.st-bj.st-cf.st-ce.st-f3.st-f4.st-af {
    font-size: 0.8em;
    }
  }
  
  /* Media query for small screens */
  @media (max-width: 1024px) {
  div[data-baseweb="select"] > div,
  input[type=number] {
    font-size: 0.8em;
    height: 3em;
  }
  .stMultiSelect [data-baseweb="select"] > div,
  .stMultiSelect [data-baseweb="tag"] {
    height: auto !important;
  }
  }
  </style>
  """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.2, 5, 0.2])
with col2:
  header_text = '''
    <p class="header_text" style="margin-top: -1.25em; margin-bottom: 1.25em; text-align: center;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1.8em; ">Supervised Machine Learning Classification</span></p>
  '''

  header_media_query = '''
      <style>
      @media (max-width: 1024px) {
          p.header_text {
            font-size: 0.6em;
          }
      }
      </style>
  '''
  st.markdown(header_media_query + header_text, unsafe_allow_html=True)

with st.sidebar:
  subheader_text = '''
  <p class="subheader_text" style="margin-top: -4em; margin-bottom: -4em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">Create a Dataset</span></p>
  '''

  subheader_media_query = '''
    <style>
    @media (max-width: 1024px) {
        p.subheader_text {
          font-size: 0.6em;
        }
    }
    </style>
  '''
  st.markdown(subheader_media_query + subheader_text, unsafe_allow_html=True)


  text = '<p class="text" style="margin-top: -3em; margin-bottom: -6em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Data Type</span></p>'
  text_media_query1 = '''
  <style>
  @media (max-width: 1024px) {
      p.text {
          font-size: 3.5em;
      }
  }
  </style>
  '''
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  data_type_options = ["", "Blobs", "Circles", "Spirals"]
  st.selectbox(label="", label_visibility="collapsed", options=data_type_options,
               format_func=lambda x: "Select Data Type" if x == "" else x, key="user_data_type")











  


