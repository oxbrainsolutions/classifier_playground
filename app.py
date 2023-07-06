import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import base64
from utils.functions import generate_data, plot_scatter


st.set_page_config(page_title="Classifier Playground", page_icon="", layout="wide")

hide_st_style = """
                <style>
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__1S137 {display: none !important;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

line1 = '<hr class="line1" style="height:0.1em; border:0em; background-color: #FCBC24; margin-top: -1.2em; margin-bottom: -2em;">'
line2 = '<hr class="line1" style="height:0.1em; border:0em; background-color: #FCBC24; margin-top: 0em; margin-bottom: -2em;">'
line_media_query = '''
    <style>
    @media (max-width: 1024px) {
        .line1 {
            padding: 0.3em;
        }
    }
    </style>
'''

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
            padding: 2em;  /* Convert 10px to em units */
        }}
        .middle-column img {{
            max-width: 200%;
            display: inline-block;
            vertical-align: middle;
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
            padding: 3em;
       }}

        .middle-column {{
            width: 100%;  /* Set width to 100% for full width on smaller screens */
            justify-content: center;
            text-align: center;
            display: flex;
            align-items: center;
            float: center;
            margin-bottom: 0em;  /* Adjust margin for smaller screens */
            padding: 0em;
        }}
        .middle-column img {{
            width: 30%;
            display: flex;
            align-items: center;
            justify-content: center;
            float: center;
          }}
    }}
    </style>
    <div class="header">
        <div class="middle-column">
            <img src="data:image/png;base64,{}" class="img-fluid" alt="comrate_logo" width="8%">
        </div>
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
        border-width: 0.15em;
        width: 100%;
        height: 0.2em !important;
        margin-top: 0em;
        font-family: sans-serif;
    }
    div.stButton > button:hover {
        background-color: #76787A;
        color: #FAFAFA;
        border-color: #002147;
    }
    @media (max-width: 1024px) {
    div.stButton > button:first-child {
        width: 100% !important;
        height: 0.8em !important;
        margin-top: 0em;
        border-width: 0.6em; !important;
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
  color: #FAFAFA;
  background-color: #4F5254;
  border: 0.25em solid #002147;
  font-size: 0.8em;
  font-family: sans-serif;
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

st.markdown("""
    <style>
        div.css-1inwz65.ektn3o0 {
            font-size: 0.7em;
            font-family: sans-serif;
        }
        div.StyledThumbValue.css-12gsf70.ektn3o2{
            font-size: 0.7em;
            font-family: sans-serif;
            color: #FAFAFA;
        }
        @media (max-width: 1024px) {
          div.css-1inwz65.ektn3o0 {
            font-size: 3.5em;
          }
          div.StyledThumbValue.css-12gsf70.ektn3o2{
            font-size: 3.5em;
        }
      }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
  subheader_text1 = '''
  <p class="subheader_text" style="margin-top: -2.2em; margin-bottom: 0em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">Create a Dataset</span></p>
  '''

  subheader_media_query = '''
    <style>
    @media (max-width: 1024px) {
        p.subheader_text {
          font-size: 4em;
        }
    }
    </style>
  '''
  st.markdown(subheader_media_query + subheader_text1, unsafe_allow_html=True)
  st.markdown(line_media_query + line1, unsafe_allow_html=True)



  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Data Type</span></p>'
  text_media_query1 = '''
  <style>
  @media (max-width: 1024px) {
      p.text {
          font-size: 4em;
      }
  }
  </style>
  '''
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  data_type_options = ["", "Blobs", "Circles", "Spirals"]
  user_data_type = st.selectbox(label="", label_visibility="collapsed", options=data_type_options,
               format_func=lambda x: "Select Data Type" if x == "" else x, key="key1")

  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Sample Size</span></p>'
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  user_n_samples = st.number_input(label="", label_visibility="collapsed", min_value=50, max_value=1000, step=10, value=300, key="key2")
  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Training Data Noise</span></p>'
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  user_train_noise = st.slider(label="", label_visibility="collapsed", min_value=0.01, max_value=1.0, step=0.005, value=0.2, key="key3")
  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Testing Data Noise</span></p>'
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  user_test_noise = st.slider(label="", label_visibility="collapsed", min_value=0.01, max_value=1.0, step=0.005, value=user_train_noise, key="key4")
  submit_button = st.button("Generate Dataset", key="key5")
  subheader_text_field1 = st.empty()
  line_field = st.empty()
  subheader_text2 = '''
  <p class="subheader_text" style="margin-top: 1em; margin-bottom: 0em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">Select a ML Model</span></p>
  '''



col1, col2, col3 = st.columns([1, 4, 1])
with col2:
  header_text = '''
    <p class="header_text" style="margin-top: 3em; margin-bottom: 0em; text-align: center;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1.8em; ">Supervised Machine Learning Classification</span></p>
  '''

  header_media_query = '''
      <style>
      @media (max-width: 1024px) {
          p.header_text {
            font-size: 3.2em;
          }
      }
      </style>
  '''
  st.markdown(header_media_query + header_text, unsafe_allow_html=True)
  information_text1 = '''
    <p class="information_text" style="margin-top: 2em; margin-bottom: 0em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">In this playground tool, you can explore the capabilities of multiple AI and ML models for classifying data. To begin, create your own dataset using the options provided in the side menu. Once your dataset is ready, you can select a ML model to train and test on the data. The model will learn from the patterns and relationships within the dataset to make predictions and classify new instances. Explore the various models, tweak the parameters and see how different algorithms perform on your dataset.</span></p>
  '''
  information_media_query = '''
      <style>
      @media (max-width: 1024px) {
          p.information_text {
            font-size: 3.6em;
          }
      }
      </style>
  '''
  subheader_text_field2 = st.empty()
  subheader_text_field2.markdown(information_media_query + information_text1, unsafe_allow_html=True)
 
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
    if submit_button:
      if user_data_type == "":
        subheader_text_field1.error("**Error**: incomplete selection.")
      else:
        x_train_out, y_train_out, x_test_out, y_test_out = generate_data(user_data_type, user_n_samples, user_train_noise, user_test_noise, n_classes=2)
        information_text2 = '''<p class="information_text" style="margin-top: 2em; margin-bottom: 1em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">The figure below shows a generated dataset based on your selected specifications composed of {} data points categorized into two distinct classes. Select between the training and testing datasets to compare the underlying structural patterns.</span></p>'''.format(user_n_samples)
         
        subheader_text_field2.markdown(information_media_query + information_text2, unsafe_allow_html=True)
        scatter_fig = plot_scatter(x_train_out, y_train_out, x_test_out, y_test_out)
        scatter_fig_field = st.empty()
        scatter_fig_field.plotly_chart(scatter_fig, config={'displayModeBar': False}, use_container_width=True)
        subheader_text_field1.markdown(subheader_media_query + subheader_text2, unsafe_allow_html=True)
        line_field.markdown(line_media_query + line2, unsafe_allow_html=True)









  


