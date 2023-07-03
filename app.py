import streamlit as st



col1, col2, col3:
with col2:
  introduction_text = '''
    <p class="introduction_text" style="margin-top: -1.25em; margin-bottom: 1.25em; text-align: justify;"><span style="color: #25476A; background-color: rgba(3, 169, 244, 0.2); border-radius: 0.375em; padding-left: 0.75em; padding-right: 0.75em; padding-top: 0.5em; padding-bottom: 0.5em; font-family: sans-serif; font-size: 1em; font-weight: bold; display: block; width: 100%; border: 0.1875em solid #25476A;">Welcome to the Paydar wargame scenario analysis application, empowering you to make informed decisions regarding the current and future financial performance of target companies based on manual and scenario-based financial statement and credit rating analyses.</span></p>
  '''

  text_media_query1 = '''
      <style>
      @media (max-width: 1024px) {
          p.introduction_text {
              font-size: 3.2em;
              border-width: 0.5em;
              position: relative;
              top: 0.5em;
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
