import streamlit as st
from utils.functions import generate_data


st.set_page_config(page_title="Classifier Playground", page_icon="", layout="wide")

st.markdown(
    """
    <style>
     div.stButton > button:first-child {
        background-color: #25476A;
        color: #FAFAFA;
        border-color: #FAFAFA;
        border-width: 3px;
        width: 5.4em;
        height: 1.8em;
        margin-top: 1.5em;
    }
    div.stButton > button:hover {
        background-color: rgba(111, 114, 222, 0.6);
        color: #25476A;
        border-color: #25476A;
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
  margin-top: 0;
  color: #FAFAFA;
  background-color: #4F5254;
  border: 0.2em solid #002147;
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
  <p class="subheader_text" style="margin-top: -3em; margin-bottom: -4em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">Create a Dataset</span></p>
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


  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Data Type</span></p>'
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
  data_type_options = ["Blobs", "Circles", "Spirals"]
  st.selectbox(label="", label_visibility="collapsed", options=data_type_options,
               format_func=lambda x: "Select Data Type" if x == "" else x, key="user_data_type")











  
  st.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

hide_st_style = """
                <style>
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__1S137 {display: none !important;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)
