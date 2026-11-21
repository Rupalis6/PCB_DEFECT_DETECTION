
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from groq import Groq

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="PCB Defect Detection & AI Assistant",
    page_icon="🔧",
    layout="wide"
)

st.title("PCB DEFECT DETECTION SYSTEM")
st.caption("YOLO11n Detection | Groq AI Remedy Assistant")
st.divider()

# ---------------- LOAD YOLO MODEL ----------------

MODEL_PATH = "PCB_YOLO11n_Best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Keep PCB_YOLO11n_Best.pt "
        "in the project folder."
    )
    st.stop()

model = load_model()

# ---------------- GROQ CLIENT ----------------

try:
    api_key = st.secrets["GROQ_API_KEY"]

    groq_client = Groq(api_key=api_key)

except Exception:
    st.error(
        "Groq API key not found. Please add GROQ_API_KEY "
        "to .streamlit/secrets.toml"
    )
    st.stop()

# ---------------- SESSION STATE ----------------

if "detected_defects" not in st.session_state:
    st.session_state.detected_defects = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ---------------- IMAGE UPLOAD ----------------

st.subheader("Upload PCB Image")

uploaded_file = st.file_uploader(
    "Choose a PCB image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Detect Defects", type="primary"):

        with st.spinner("Analysing PCB image..."):

            try:
                results = model.predict(
                    source=image,
                    imgsz=512,
                    verbose=False
                )

                result = results[0]

                # Generate annotated image
                annotated = result.plot()
                annotated = annotated[:, :, ::-1]

                # Count defect categories
                defect_counts = {}

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    defect_counts[class_name] = (
                        defect_counts.get(class_name, 0) + 1
                    )

                # Save detection results
                st.session_state.original_image = image
                st.session_state.annotated_image = annotated
                st.session_state.detected_defects = defect_counts

                # Reset previous chatbot conversation
                st.session_state.chat_history = []

                st.session_state.last_uploaded_file = uploaded_file.name

            except Exception as e:
                st.error(f"Detection error: {e}")

# ---------------- DISPLAY DETECTION RESULTS ----------------

if st.session_state.annotated_image is not None:

    st.divider()
    st.subheader("Detection Results")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original PCB Image")

        st.image(
            st.session_state.original_image,
            use_container_width=True
        )

    with col2:

        st.markdown("### Detected PCB Defects")

        st.image(
            st.session_state.annotated_image,
            use_container_width=True
        )

    defects = st.session_state.detected_defects

    total_defects = sum(defects.values())

    st.metric(
        "Total Defects Detected",
        total_defects
    )

    if total_defects > 0:

        st.markdown("### Defect Classification")

        for defect, count in defects.items():

            st.write(
                f"**{defect.replace('_', ' ').title()}**: {count}"
            )

    else:

        st.success(
            "No defects detected at the model's default settings."
        )

    # ---------------- PCB AI CHATBOT ----------------

    st.divider()

    st.header("PCB AI Remedy Assistant")

    st.write(
        "Ask questions about detected PCB defects, "
        "their causes, repair methods and prevention."
    )

    # Prepare defect information
    if defects:

        defect_context = "\n".join(
            [
                f"- {name.replace('_', ' ').title()}: {count}"
                for name, count in defects.items()
            ]
        )

    else:

        defect_context = "No defects were detected in the PCB image."

    # System instructions
    system_prompt = f"""
You are an expert PCB manufacturing and troubleshooting assistant.

YOLO11n analysed the PCB image and detected the following results:

{defect_context}

Your responsibilities:

1. Explain each detected PCB defect.
2. Describe its possible manufacturing causes.
3. Explain how to inspect and verify the defect.
4. Suggest practical repair methods.
5. List the tools and equipment required.
6. Explain preventive measures.
7. Provide step-by-step troubleshooting guidance.
8. Answer follow-up questions in simple technical language.

Important instructions:

- Use the detected defect information as context.
- Do not claim to see or analyse the PCB image yourself.
- Do not invent additional detected defects.
- Distinguish between manufacturing repair and field repair.
- Explain when a defect may make the PCB unsuitable for use.
- Mention that repair feasibility depends on PCB design and damage severity.
- Recommend professional inspection for complex or multilayer PCB damage.
- Never guarantee that a damaged PCB is safe after repair.
- Keep answers structured, accurate and easy to understand.
"""

    # Display previous conversation
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask about PCB defects or their remedies..."
    )

    if user_question:

        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Prepare messages for Groq
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        messages.extend(
            st.session_state.chat_history[-10:]
        )

        with st.chat_message("assistant"):

            with st.spinner("Preparing PCB remedy suggestions..."):

                try:

                    response = groq_client.chat.completions.create(

                        # Updated Groq model
                        model="openai/gpt-oss-20b",

                        messages=messages,

                        temperature=0.3,

                        max_tokens=1500
                    )

                    answer = response.choices[0].message.content

                    if not answer:
                        answer = "The AI did not return a response. Please try again."

                    st.markdown(answer)

                    # Save assistant response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        f"Groq API error: {e}"
                    )

    # Clear conversation
    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

else:

    st.info(
        "Upload a PCB image and click Detect Defects to start."
    )
