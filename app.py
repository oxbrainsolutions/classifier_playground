import streamlit as st


st.set_page_config(page_title="Paydar", page_icon="", layout="wide", initial_sidebar_state="expanded")

col1, col2, col3 = st.columns([0.2, 5, 0.2])
with col2:
  header_text = '''
    <p class="header_text" style="margin-top: -1.25em; margin-bottom: 1.25em; text-align: center;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 2em; ">Supervised Machine Learning Classification</span></p>
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

add_selectbox = st.sidebar.selectbox(
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
