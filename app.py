import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pathlib
import base64
import time
from utils.functions import generate_data, plot_scatter, plot_scatter_decision_boundary, create_gauge, convert_rating, add_polynomial_features, train_model, lr_param_selector, nb_param_selector, lda_param_selector, qda_param_selector, knn_param_selector, nn_param_selector, svm_param_selector, ct_param_selector, rf_param_selector, ad_param_selector, gb_param_selector


st.set_page_config(page_title="Classifier Playground", page_icon="", layout="wide")

if "user_data_type" not in st.session_state or "user_n_samples" not in st.session_state or "user_train_noise" not in st.session_state or "user_test_noise" not in st.session_state or "user_model" not in st.session_state:
    st.session_state["user_data_type"] = ""
    st.session_state["user_n_samples"] = ""
    st.session_state["user_train_noise"] = ""
    st.session_state["user_test_noise"] = ""
    st.session_state["user_model"] = ""

if "x_train_out" not in st.session_state or "y_train_out" not in st.session_state or "x_test_out" not in st.session_state or "y_test_out" not in st.session_state or "x_train_out_update" not in st.session_state or "x_test_out_update" not in st.session_state:
    st.session_state["x_train_out"] = []
    st.session_state["y_train_out"] = []
    st.session_state["x_test_out"] = []
    st.session_state["y_test_out"] = []
    st.session_state["x_train_out_update"] = []
    st.session_state["x_test_out_update"] = []

if "submit_confirm1" not in st.session_state or "submit_confirm2" not in st.session_state:
    st.session_state["submit_confirm1"] = False
    st.session_state["submit_confirm2"] = False

if "modal1" not in st.session_state or "modal2" not in st.session_state:
    st.session_state["modal1"] = False
    st.session_state["modal2"] = False

hide_st_style = """
                <style>
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__1S137 {display: none !important;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
  <style>
    div.block-container.css-ysnqb2.e1g8pov64 {
        margin-top: -3em;
    }
    div[data-modal-container='true'][key='Modal1'] > div:first-child > div:first-child {
        background-color: rgb(203, 175, 175) !important;
        z-index:99999 !important;
    }
    div[data-modal-container='true'][key='Modal1'] > div > div:nth-child(2) > div {
        max-width: 3em !important;
        z-index:999991 !important;
    }
    div[data-modal-container='true'][key='Modal2'] > div:first-child > div:first-child {
        background-color: rgb(203, 175, 175) !important;
        z-index:99999 !important;
    }
    div[data-modal-container='true'][key='Modal2'] > div > div:nth-child(2) > div {
        max-width: 3em !important;
        z-index:999991 !important;
    }
    @media (max-width: 1024px) {
        div.block-container.css-ysnqb2.e1g8pov64 {
            margin-top: -15em;
        }
    }
  </style>
""", unsafe_allow_html=True)

st.markdown("""
  <style>
    .css-1l7fe1m {
        background-color: #FCBC24;
    }
    .css-5qhjmn {
        z-index: 1000 !important;
    }
    .css-15d9ls5{
        z-index: 1000 !important;
    }
  </style>
""", unsafe_allow_html=True)

line1 = '<hr class="line1" style="height:0.1em; border:0em; background-color: #FCBC24; margin-top: 0em; margin-bottom: -2em;">'
line_media_query1 = '''
    <style>
    @media (max-width: 1024px) {
        .line1 {
            padding: 0.3em;
        }
    }
    </style>
'''

line2 = '<hr class="line2" style="height:0.1em; border:0em; background-color: #FAFAFA; margin-top: 0em; margin-bottom: -2em;">'
line_media_query2 = '''
    <style>
    @media (max-width: 1024px) {
        .line2 {
            padding: 0.05em;
        }
    }
    </style>
'''

def change_callback1():
    st.session_state.submit_confirm1 = False
    st.session_state.submit_confirm2 = False
    if st.session_state.modal1.is_open():
        st.session_state.modal1.close() 
    if st.session_state.modal2.is_open():
        st.session_state.modal2.close() 

def change_callback2():
    st.session_state.submit_confirm2 = False
    if st.session_state.modal2.is_open():
        st.session_state.modal2.close() 

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
    div.stButton {
        display: flex !important;
        justify-content: center !important;
    }
    
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
        border-width: 0.15em; !important;
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
            font-size: 0.8em;
            font-family: sans-serif;
        }
        div.StyledThumbValue.css-12gsf70.ektn3o2{
            font-size: 0.8em;
            font-family: sans-serif;
            color: #FAFAFA;
        }
        @media (max-width: 1024px) {
          div.css-1inwz65.ektn3o0 {
            font-size: 0.8em;
          }
          div.StyledThumbValue.css-12gsf70.ektn3o2{
            font-size: 0.8em;
        }
      }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    subheader_text1 = '''<p class="subheader_text" style="margin-top: 0em; margin-bottom: 0em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">Create a Dataset</span></p>'''
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
    st.markdown(line_media_query1 + line1, unsafe_allow_html=True)

dataset_container = st.sidebar.expander("", expanded = True)
with dataset_container:
  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Data Type</span></p>'
  text_media_query1 = '''
  <style>
  @media (max-width: 1024px) {
      p.text {
          font-size: 1em;
      }
  }
  </style>
  '''
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  data_type_options = ["", "Blobs", "Circles", "Spirals"]
  st.session_state.user_data_type = st.selectbox(label="", label_visibility="collapsed", options=data_type_options,
               format_func=lambda x: "Select Data Type" if x == "" else x, key="key1", on_change=change_callback1)

  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Sample Size</span></p>'
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  st.session_state.user_n_samples = st.number_input(label="", label_visibility="collapsed", min_value=50, max_value=1000, step=10, value=300, key="key2", on_change=change_callback1)
  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Training Data Noise</span></p>'
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  st.session_state.user_train_noise = st.slider(label="", label_visibility="collapsed", min_value=0.01, max_value=1.0, step=0.005, value=0.2, key="key3", on_change=change_callback1)
  text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Testing Data Noise</span></p>'
  st.markdown(text_media_query1 + text, unsafe_allow_html=True)
  st.session_state.user_test_noise = st.slider(label="", label_visibility="collapsed", min_value=0.01, max_value=1.0, step=0.005, value=st.session_state.user_train_noise, key="key4", on_change=change_callback1)
  submit_button1 = st.button("Generate Dataset", key="key5")

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

st.session_state.modal1 = Modal("", key="Modal1", padding=20, max_width=240)
st.session_state.modal2 = Modal("", key="Modal2", padding=20, max_width=250)

if submit_button1:
    if st.session_state.user_data_type == "":
        #with st.sidebar:
            #st.error("**Error**: select data type.")
        st.session_state.submit_confirm1 = False
        st.session_state.modal1.open()
    else:
      st.session_state.submit_confirm1 = True    

if st.session_state.modal1.is_open():
    with st.session_state.modal1.container():
        error_text1 = '''<p class="error_text1" style="margin-top: 0em; margin-bottom: 1em; text-align: right;"><span style="color: #850101; font-family: sans-serif; font-size: 1em; font-weight: bold;">Error: select data type</span></p>'''
        error_media_query1 = '''
        <style>
        @media (max-width: 1024px) {
            p.error_text1 {
              font-size: 4em;
            }
        }
        </style>
        '''
        st.markdown(error_media_query1 + error_text1 , unsafe_allow_html=True)

if st.session_state.submit_confirm1 == True:
    if st.session_state.modal1.is_open():
        st.session_state.modal1.close()
    
    with st.sidebar:
      subheader_text_field1 = st.empty()
      line_field = st.empty()
      subheader_text2 = '''<p class="subheader_text" style="margin-top: 1em; margin-bottom: 0em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">Select a ML Model</span></p>'''
      subheader_text_field1.markdown(subheader_media_query + subheader_text2, unsafe_allow_html=True)
      line_field.markdown(line_media_query1 + line1, unsafe_allow_html=True)
    
    model_container = st.sidebar.expander("", expanded = True)        
    with model_container:
      model_text_field = st.empty()
      user_model_field = st.empty()
      text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">ML Model</span></p>'
      model_options = ["", "Logistic Regression", "Naive Bayes", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "K Nearest Neighbor", "Neural Network", "Support Vector Machine", "Classification Tree", "Random Forest", "Adaptive Boosting Machine", "Gradient Boosting Machine"]
      model_text_field.markdown(text_media_query1 + text, unsafe_allow_html=True)
      st.session_state.user_model = user_model_field.selectbox(label="", label_visibility="collapsed", options=model_options, format_func=lambda x: "Select Model" if x == "" else x, key="key6", on_change=change_callback2)
      if st.session_state.user_model == "Logistic Regression":
        model = lr_param_selector()
      elif st.session_state.user_model == "Naive Bayes":
        model = nb_param_selector()
      elif st.session_state.user_model == "Linear Discriminant Analysis":
        model = lda_param_selector()
      elif st.session_state.user_model == "Quadratic Discriminant Analysis":
        model = qda_param_selector()
      elif st.session_state.user_model == "K Nearest Neighbor":
          model = knn_param_selector()
      elif st.session_state.user_model == "Neural Network":
          model = nn_param_selector()
      elif st.session_state.user_model == "Support Vector Machine":
          model = svm_param_selector()
      elif st.session_state.user_model == "Classification Tree":
          model = ct_param_selector()
      elif st.session_state.user_model == "Random Forest":
          model = rf_param_selector()
      elif st.session_state.user_model == "Adaptive Boosting Machine":
          model = ad_param_selector()
      elif st.session_state.user_model == "Gradient Boosting Machine":
          model = gb_param_selector()
        
      if st.session_state.user_model != "":
          text = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; ">Feature Polynomial Degree</span></p>'
          st.markdown(text_media_query1 + text, unsafe_allow_html=True)
          user_poly_degree = st.number_input(label="", label_visibility="collapsed", min_value=1, max_value=10, step=1, value=1, key="key7", on_change=change_callback2)
          submit_button2 = st.button("Train Model", key="key8")
          st.markdown(line_media_query2 + line2, unsafe_allow_html=True)
          if st.session_state.user_model == "Logistic Regression":
              info_text1 = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><ul><li style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; text-align: justify;">Logistic regression is a statistical model used for binary classification, predicting the probability of an instance belonging to a particular class based on input features.</li></ul></p>'
              st.markdown(text_media_query1 + info_text1, unsafe_allow_html=True)
              info_text2 = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><ul><li style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; text-align: justify;">Regularization and complexity constraints: Logistic regression can use regularization techniques to prevent overfitting and control model complexity.</li></ul></p>'
              st.markdown(text_media_query1 + info_text2, unsafe_allow_html=True)
              info_text3 = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><ul><li style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; text-align: justify;">Pros: Interpretable results as it provides coefficients indicating the impact of each feature on the classification. Efficient and computationally inexpensive.</li></ul></p>'
              st.markdown(text_media_query1 + info_text3, unsafe_allow_html=True)
              info_text4 = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><ul><li style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; text-align: justify;">Cons: Assumes a linear relationship. May not perform well with complex data patterns.</li></ul></p>'
              st.markdown(text_media_query1 + info_text4, unsafe_allow_html=True)


          st.markdown(line_media_query2 + line2, unsafe_allow_html=True)
          info_text_poly1 = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><ul><li style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; text-align: justify;">Polynomial feature terms: ML models can incorporate polynomial feature terms to capture nonlinear relationships between features and improve model performance.</li></ul></p>'
          st.markdown(text_media_query1 + info_text_poly1, unsafe_allow_html=True)
          info_text_poly2 = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><ul><li style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; text-align: justify;">Pros: Allows for modeling complex relationships and interactions between features.</li></ul></p>'
          st.markdown(text_media_query1 + info_text_poly2, unsafe_allow_html=True)
          info_text_poly3 = '<p class="text" style="margin-top: 0em; margin-bottom: 0em;"><ul><li style="font-family:sans-serif; color:#FAFAFA; font-size: 0.8em; text-align: justify;">Cons: Increases the dimensionality of the feature space, which can lead to overfitting if not properly controlled.</li></ul></p>'
          st.markdown(text_media_query1 + info_text_poly3, unsafe_allow_html=True)

          if submit_button2:
              st.session_state.x_train_out_update, st.session_state.x_test_out_update = add_polynomial_features(st.session_state.x_train_out, st.session_state.x_test_out, user_poly_degree)
              try:
                  model, train_accuracy, train_f1, test_accuracy, test_f1, duration = train_model(model, st.session_state.x_train_out_update, st.session_state.y_train_out, st.session_state.x_test_out_update, st.session_state.y_test_out)
                  st.session_state.submit_confirm2 = True
                  st.write(train_accuracy)
              except:
                  st.session_state.submit_confirm2 = False
                  st.session_state.modal2.open()
                  #with st.sidebar:
                      #st.error("**Error**: complete selection.")
  
if st.session_state.modal2.is_open():
    with st.session_state.modal2.container():
        error_text2 = '''<p class="error_text1" style="margin-top: 0em; margin-bottom: 1em; text-align: right;"><span style="color: #850101; font-family: sans-serif; font-size: 1em; font-weight: bold;">Error: complete selection</span></p>'''
        error_media_query2 = '''
        <style>
        @media (max-width: 1024px) {
            p.error_text1 {
              font-size: 4em;
            }
        }
        </style>
        '''
        st.markdown(error_media_query2 + error_text2 , unsafe_allow_html=True)

st.text("")
subheader_text_field3 = st.empty()
st.text("")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.session_state.submit_confirm2 == True:
        if st.session_state.modal2.is_open():
            st.session_state.modal2.close()      
        create_gauge(num_value='{:.2f}'.format(np.round(train_accuracy, 2)), label="Train\nAccuracy", key="key_gauge1")
        text_rating1 = '<p class="text2" style="margin-top: -4em; margin-bottom: 0em; text-align: center;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 2em; ">{}</span></p>'.format(convert_rating(train_accuracy))
        text_media_query2 = '''
          <style>
          @media (max-width: 1024px) {
              p.text2 {
                  font-size: 3em;
              }
          }
          </style>
          '''
        rating1 = st.empty()
        create_gauge(num_value='{:.2f}'.format(np.round(train_f1, 2)), label="Train\nF1 Score", key="key_gauge2")
        text_rating2 = '<p class="text2" style="margin-top: -4em; margin-bottom: 0em; text-align: center;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 2em; ">{}</span></p>'.format(convert_rating(train_f1))
        rating2 = st.empty()
with col3:
    if st.session_state.submit_confirm2 == True:
        create_gauge(num_value='{:.2f}'.format(np.round(test_accuracy, 2)), label="Test\nAccuracy", key="key_gauge3")
        text_rating3 = '<p class="text2" style="margin-top: -4em; margin-bottom: 0em; text-align: center;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 2em; ">{}</span></p>'.format(convert_rating(test_accuracy))
        rating3 = st.empty()
        create_gauge(num_value='{:.2f}'.format(np.round(test_f1, 2)), label="Test\nF1 Score", key="key_gauge4")
        text_rating4 = '<p class="text2" style="margin-top: -4em; margin-bottom: 0em; text-align: center;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 2em; ">{}</span></p>'.format(convert_rating(test_f1))
        rating4 = st.empty()
        time.sleep(1)
        rating1.markdown(text_media_query2 + text_rating1, unsafe_allow_html=True)
        rating2.markdown(text_media_query2 + text_rating2, unsafe_allow_html=True)
        rating3.markdown(text_media_query2 + text_rating3, unsafe_allow_html=True)
        rating4.markdown(text_media_query2 + text_rating4, unsafe_allow_html=True)
with col2:
  if st.session_state.submit_confirm1 == True:
    st.session_state.x_train_out, st.session_state.y_train_out, st.session_state.x_test_out, st.session_state.y_test_out = generate_data(st.session_state.user_data_type, st.session_state.user_n_samples, st.session_state.user_train_noise, st.session_state.user_test_noise, n_classes=2)
    information_text2 = '''<p class="information_text" style="margin-top: 2em; margin-bottom: 1em; text-align: justify;"><span style="color: #FAFAFA; font-family: sans-serif; font-size: 1em; ">The figure below shows a generated dataset based on your selected specifications composed of {} data points categorized into two distinct classes. Select between the training and testing datasets to compare the underlying structural patterns.</span></p>'''.format(st.session_state.user_n_samples)
     
    subheader_text_field2.markdown(information_media_query + information_text2, unsafe_allow_html=True)
    subheader_text3 = '<p class="subheader_text2" style="margin-top: 0em; margin-bottom: 0em; text-align: center;"><span style="font-family:sans-serif; color:rgba(0, 0, 0, 0); font-size: 2em; ">Gap</span></p>'
    subheader_media_query2 = '''
          <style>
          @media (max-width: 1024px) {
              p.subheader_text2 {
                font-size: 3.2em;
              }
          }
          </style>
      '''
    subheader_text_field3.markdown(subheader_media_query2 + subheader_text3, unsafe_allow_html=True)
    scatter_fig = plot_scatter(st.session_state.x_train_out, st.session_state.y_train_out, st.session_state.x_test_out, st.session_state.y_test_out)
    scatter_fig_field = st.empty()
    scatter_fig_field.plotly_chart(scatter_fig, config={'displayModeBar': False}, use_container_width=True)
    if st.session_state.submit_confirm2 == True:
        subheader_text3 = '<p class="subheader_text2" style="margin-top: 0em; margin-bottom: 0em; text-align: center;"><span style="font-family:sans-serif; color:#FCBC24; font-size: 2em; ">{}</span></p>'.format(st.session_state.user_model)
        subheader_text_field3.markdown(subheader_media_query2 + subheader_text3, unsafe_allow_html=True)
        duration_text = '<p class="information_text" style="margin-top: -0.5em; margin-bottom: 2em; text-align: center;"><span style="font-family:sans-serif; color:#FAFAFA; font-size: 1em; ">Model training completed in {} seconds</span></p>'.format('{:,.3f}'.format(duration))
        st.markdown(information_media_query + duration_text, unsafe_allow_html=True)
        scatter_boundary_fig = plot_scatter_decision_boundary(model, st.session_state.x_train_out_update, st.session_state.y_train_out, st.session_state.x_test_out_update, st.session_state.y_test_out)
        scatter_fig_field.plotly_chart(scatter_boundary_fig, config={'displayModeBar': False}, use_container_width=True)






  


