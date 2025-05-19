import streamlit as st
import face_recognition
import numpy as np
import requests
import datetime
import json

GAS_ENDPOINT = (
    "https://script.google.com/macros/s/AKfycbz85q3-5fifClgDUqGQ6hrN3cDa3AgywAwzUSf7Q7VMWz-GI56RWV0IchCpyE7Q-jJjuQ/exec"
)  # Replace with your GAS endpoint

def safe_get_json(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Cannot reach GAS: {e}")
        st.stop()

    if "application/json" not in resp.headers.get("content-type", ""):
        st.error(f"GAS did not return JSON.\nFirst 300 chars:\n{resp.text[:300]}")
        st.stop()

    try:
        return resp.json()
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}\n{resp.text[:300]}")
        st.stop()

@st.cache_data(show_spinner=False)
def fetch_registered():
    data = safe_get_json(GAS_ENDPOINT)
    names = [d["name"] for d in data]
    encs = [np.fromstring(d["encoding"], sep=",") for d in data]
    return names, encs, data

def post_student(row):
    try:
        requests.post(GAS_ENDPOINT, json=row, timeout=10).raise_for_status()
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        st.stop()

def draw_face_boxes(image, face_locations):
    import cv2
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    # Convert back to RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

st.set_page_config(page_title="Student Face System", page_icon="🎓", layout="wide")
st.title("🎓 Student Face System — Camera Input with Face Detection")

names_known, encs_known, full_data = fetch_registered()

tab_reg, tab_log = st.tabs(["📝 Register", "✅ Login"])

with tab_reg:
    st.subheader("Register New Student")
    name = st.text_input("Full Name")
    sid = st.text_input("Student ID")

    img_file = st.camera_input("Take a photo for registration")
    reg_img = None

    if img_file is not None:
        reg_img = face_recognition.load_image_file(img_file)
        faces = face_recognition.face_locations(reg_img)
        if faces:
            img_with_boxes = draw_face_boxes(reg_img, faces)
            st.image(img_with_boxes, caption=f"Detected {len(faces)} face(s)", use_container_width=True)
        else:
            st.warning("No faces detected in the photo.")

    can_register = reg_img is not None and name.strip() != "" and sid.strip() != ""

    if st.button("Register", disabled=not can_register):
        faces = face_recognition.face_locations(reg_img)
        if len(faces) != 1:
            st.error("Exactly ONE face must be visible for registration.")
            st.stop()

        enc = face_recognition.face_encodings(reg_img, faces)[0]

        if encs_known:
            matches = face_recognition.compare_faces(encs_known, enc, tolerance=0.45)
            if True in matches:
                st.error(f"Duplicate! Already registered as {names_known[matches.index(True)]}.")
                st.stop()

        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "student_id": sid.strip(),
            "name": name.strip(),
            "encoding": ",".join(map(str, enc.tolist())),
        }
        post_student(row)
        st.success("✅ Registered & stored!")

        fetch_registered.clear()
        names_known, encs_known, full_data = fetch_registered()

    with st.expander("📄 View registered students"):
        st.dataframe(full_data, use_container_width=True)

with tab_log:
    st.subheader("Student Login / Check-in")

    img_file = st.camera_input("Take a photo for login")
    login_img = None

    if img_file is not None:
        login_img = face_recognition.load_image_file(img_file)
        faces = face_recognition.face_locations(login_img)
        if faces:
            img_with_boxes = draw_face_boxes(login_img, faces)
            st.image(img_with_boxes, caption=f"Detected {len(faces)} face(s)", use_container_width=True)
        else:
            st.warning("No faces detected in the photo.")

    if st.button("Login", disabled=(login_img is None)):
        if login_img is None:
            st.warning("Please take a photo first.")
            st.stop()

        faces = face_recognition.face_locations(login_img)
        if len(faces) != 1:
            st.error("Exactly ONE face must be visible for login.")
            st.stop()

        enc = face_recognition.face_encodings(login_img, faces)[0]

        if not encs_known:
            st.error("No students registered yet.")
            st.stop()

        matches = face_recognition.compare_faces(encs_known, enc, tolerance=0.45)
        if True in matches:
            st.success(f"✅ Welcome back, {names_known[matches.index(True)]}! You are checked in.")
            st.session_state["logged_in"] = names_known[matches.index(True)]
        else:
            st.error("Face not recognised. Please register first.")

    if "logged_in" in st.session_state:
        st.markdown(f"### Logged in as: {st.session_state['logged_in']}")
        if st.button("Log out"):
            del st.session_state["logged_in"]
            st.experimental_rerun()
