from sklearn.preprocessing import normalize
import streamlit as st
from src.face_ops import extract_embedding, recognize, add_to_database, load_model
from PIL import Image

# Load model and index
model, index, database = load_model()

# ---------- Streamlit UI ---------- #
st.set_page_config(page_title="Incremental Face Recognition", layout="centered")
st.title("Incremental Face Recognition System (PyTorch)")

mode = st.radio("Choose mode", ["Enroll New Face", "Recognize Face"])

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    embedding = extract_embedding(model, image, model_name="vggface2")

    if mode == "Enroll New Face":
        name = st.text_input("Enter name")
        if st.button("Enroll") and name:
            success = add_to_database(name, embedding, index, database)
            if success:
                st.success(f"{name} has been enrolled successfully!")
            else:
                st.warning("Face already enrolled")

    elif mode == "Recognize Face":
        if st.button("Identify"):
            st.write("Button clicked. Recognizing...")
            embedding = normalize(embedding, axis=1).astype("float32")
            identity, dist = recognize(embedding, index, database)
            if dist is not None or identity!= "Unknown":
                st.write(
                f"Immediate check → Identified as: {identity} at similarity: {dist:.4f}"
            )
            else:
                st.write("No match found.")
           
    

if st.checkbox("Show enrolled database"):
    if not database:
        st.write("No faces enrolled.")
    else:
        st.write([name for name, _ in database])
