import winsound
import streamlit as st
import cv2
import numpy as np
import face_recognition

st.title('Face Detection Using face_recognition')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Select an Option')

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Detect Face']
)

if app_mode =='About App':
    st.markdown('In this application we are using Face recognition for detecting the face of a person. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown('''
          # About Me \n 
            Hey this is ** Deep Karki **. \n

            Social Media
            - [LinkedIn](https://www.linkedin.com/in/deep-karki-73717b217)
            - [Instagram](https://www.instagram.com/dero_703/)
            ''')
elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    stframe = st.empty()
    imag = st.sidebar.file_uploader("Upload imgae here", type=['jpeg','jpg','png'])
    
    if imag:
        img = face_recognition.load_image_file(imag) 
        img_face_encoding = face_recognition.face_encodings(img)[0]
        use_webcam = st.button('Start detecting')

        if use_webcam:
            vid = cv2.VideoCapture(0)

            known_face_encodings = [img_face_encoding]

            known_face_names = ["Found"]

            face_locations = []

            face_encodings = []

            face_names = []

            process_this_frame = True

            while True:

                success, frame = vid.read() 

                if not success:
                    break

                else:
                    small_frame = cv2.resize(frame, (0, 0),None,0.25,0.25)
                    small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
                    rgb_small_frame = small_frame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []

                    for face_encoding,face_loc in zip(face_encodings,face_locations):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        name = "Unknown"
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_names.append(name)

                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        
                        if name != 'Unknown':
                            winsound.Beep(500,200)

                    stframe.image(frame,channels = 'BGR',use_column_width=True)

            vid.release()

    else:
        st.markdown('Upload an IMAGE To Detect the Person')