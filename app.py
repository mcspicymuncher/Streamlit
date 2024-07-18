import streamlit as st
import PIL
import cv2
import numpy
import utils
import io

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
            visualised_image = utils.predict_image(frame, confs_threshold = conf_threshold)
            st_frame.image(visualised_image,channels= "BGR")

        else:
            camera.release()
            break

st.set_page_config(
    page_title="Age/Gender/Emotion",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Age/Gender/Emotion :sun_with_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE","VIDEO","WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the confidence Threshold", 10, 100,20))/100

input = None
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an Image.", type=("jpg", "png"))
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image),cv2.COLOR_RGB2BGR)
        visualised_image = utils.predict_image(uploaded_image_cv, confs_threshold = conf_threshold)
        st.image(visualised_image, channels = "BGR")

temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an Image.", type=("mp4"))
    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"

        with open(temporary_location, "wb") as out:
            out.write(g.read())
            out.close()

    if temporary_location is not None:
        play_video(temporary_location)
        if st.button("Replay", type="primary"):
            pass

if source_radio == "WEBCAM":
    play_video(0)
