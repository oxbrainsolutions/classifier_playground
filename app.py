import streamlit as st



col1, col2, col3 = st.columns([0.2, 5, 0.2])
with col2:
  introduction_text = '''
    <p class="header_text" style="margin-top: -1.25em; margin-bottom: 1.25em; text-align: center;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 2em; ">Supervised Machine Learning Classification</span></p>
  '''

  text_media_query1 = '''
      <style>
      @media (max-width: 1024px) {
          p.introduction_text {
              font-size: 0.2em;
          }
      }
      </style>
  '''
  st.markdown(text_media_query1 + introduction_text, unsafe_allow_html=True)

hide_st_style = """
                <style>
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__1S137 {display: none !important;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)
