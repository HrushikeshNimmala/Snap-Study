
import streamlit as st
import json
import base64
import time
import fitz

from google import genai
from prompts import STUDY_PROMPT
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.lib.enums import TA_CENTER


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Snap & Study AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Snap & Study")
st.subheader("Your personal AI-powered study assistant")

st.write(
    "Upload your notes, textbook pages, diagrams, or question papers. "
    "Generate summaries, key points, important questions, and quizzes."
)

st.caption("Learn smarter, not harder.")

st.divider()


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    st.error(
        "Gemini API key not found. Check your "
        ".streamlit/secrets.toml file."
    )
    st.code('GEMINI_API_KEY = "YOUR_API_KEY_HERE"')
    st.stop()


MODEL_NAME = "gemini-3.8-flash"

# Number of PDF pages sent in each batch.
PDF_PAGES_PER_CHUNK = 4


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "study_data": None,
    "study_input": None,
    "study_source_text": "",
    "previous_quiz_questions": [],
    "quiz_current": 0,
    "quiz_score": 0,
    "quiz_answered": False,
    "quiz_finished": False,
    "quiz_feedback": "",
    "quiz_selected": {},
    "analysis_done": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# PROMPTS
# =========================================================

CHUNK_PROMPT = """
You are an expert study assistant.

Analyze the provided section of a student's study material carefully.

The material may contain textbook text, scientific concepts,
formulas, definitions, examples, diagrams, and scanned pages.

Return ONLY valid JSON. Do not use Markdown or code fences.

Use exactly this structure:

{
  "section_summary": "A clear explanation of this section.",
  "key_points": [
    "Important point",
    "Important point",
    "Important point",
    "Important point",
    "Important point"
  ],
  "important_facts": [
    "Important fact, formula, definition, or example"
  ]
}

Rules:
- Analyze only the supplied section.
- Include important definitions, formulas, examples, and concepts.
- Describe relevant diagrams when visible.
- Keep explanations clear and student-friendly.
- Do not invent information.
- Preserve the meaning of formulas and scientific notation.
- Include page-specific details when useful.
- Return valid JSON only.
"""


FINAL_STUDY_PROMPT = STUDY_PROMPT + """

The supplied material contains notes generated from multiple sections
of the SAME uploaded PDF.

Combine the notes into one complete study guide.

IMPORTANT:
- Cover the concepts from all supplied sections.
- Do not focus only on the first section.
- Remove duplicate points while preserving important information.
- Create the exact number of questions and MCQs required by the
  original instructions.
- Base the final guide only on the supplied section notes.
- Return valid JSON only, following the original required structure.
"""


# =========================================================
# GEMINI RESPONSE HELPERS
# =========================================================

def get_response_text(response):
    """Extract text from the Gemini response."""

    output_text = getattr(response, "output_text", None)

    if output_text:
        return output_text

    response_text = getattr(response, "text", None)

    if response_text:
        return response_text

    output = getattr(response, "output", None)

    if output:
        collected_text = []

        for item in output:
            item_text = getattr(item, "text", None)

            if item_text:
                collected_text.append(item_text)

            elif isinstance(item, dict) and item.get("text"):
                collected_text.append(item["text"])

        if collected_text:
            return "\n".join(collected_text)

    raise ValueError("No text was returned by Gemini.")


def clean_json_response(response_text):
    """Remove Markdown fences if the model returns them."""

    response_text = response_text.strip()

    if response_text.startswith("```"):
        lines = response_text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response_text = "\n".join(lines).strip()

    return response_text


def analyze_with_gemini(input_content):
    """Call Gemini with retry handling for temporary failures."""

    last_error = None

    for attempt in range(3):
        try:
            response = client.interactions.create(
                model=MODEL_NAME,
                input=input_content,
            )

            return get_response_text(response)

        except Exception as error:
            last_error = error

            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    st.error(
        "Gemini could not process this request after three attempts. "
        "The service may be temporarily busy, or the request may be "
        "too large."
    )

    with st.expander("Technical details"):
        st.code(str(last_error))

    return None


def create_text_input(text):
    return [{"type": "text", "text": text}]


def create_image_input(prompt, image_bytes, mime_type="image/png"):
    """Create Gemini input containing text and an image."""

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    return [
        {"type": "text", "text": prompt},
        {
            "type": "image",
            "data": encoded_image,
            "mime_type": mime_type,
        },
    ]


# =========================================================
# VALIDATION
# =========================================================

def validate_study_data(data):
    """Check that the generated study guide has the required structure."""

    required_keys = [
        "summary",
        "key_points",
        "important_questions",
        "mcqs",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required field: {key}")

    if not isinstance(data["summary"], str):
        raise ValueError("The summary must be text.")

    if not isinstance(data["key_points"], list):
        raise ValueError("Key points must be a list.")

    if len(data["key_points"]) != 5:
        raise ValueError("The study guide must contain 5 key points.")

    if not isinstance(data["important_questions"], list):
        raise ValueError("Important questions must be a list.")

    if len(data["important_questions"]) != 5:
        raise ValueError("The study guide must contain 5 important questions.")

    if not isinstance(data["mcqs"], list) or len(data["mcqs"]) != 5:
        raise ValueError("The study guide must contain 5 MCQs.")

    for question in data["important_questions"]:
        if not isinstance(question, dict):
            raise ValueError("An important question has an invalid format.")

        if "question" not in question or "answer" not in question:
            raise ValueError("An important question is missing its answer.")

    for mcq in data["mcqs"]:
        if not isinstance(mcq, dict):
            raise ValueError("An MCQ has an invalid format.")

        if not all(
            key in mcq
            for key in ["question", "options", "correct_answer"]
        ):
            raise ValueError("An MCQ is missing required fields.")

        if not isinstance(mcq["options"], list) or len(mcq["options"]) != 4:
            raise ValueError("Each MCQ must contain exactly 4 options.")

        if mcq["correct_answer"] not in mcq["options"]:
            raise ValueError(
                "An MCQ's correct answer must match one of its options."
            )

    return True


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_content(pdf_bytes):
    """
    Extract every PDF page.

    Pages containing selectable text are stored as text.
    Pages without selectable text are rendered as images so
    scanned pages can also be analyzed.
    """

    page_items = []
    total_pages = 0

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(document)

        if total_pages == 0:
            document.close()
            raise ValueError("The uploaded PDF contains no pages.")

        progress = st.progress(0, text="Reading PDF pages...")

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_text = page.get_text("text").strip()

            page_item = {
                "page_number": page_number,
                "text": page_text,
                "image": None,
            }

            # If a page has no selectable text, render it for visual analysis.
            if not page_text:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(1.5, 1.5),
                    alpha=False,
                )

                page_item["image"] = pixmap.tobytes("png")

            page_items.append(page_item)

            progress.progress(
                (page_index + 1) / total_pages,
                text=f"Reading PDF page {page_number} of {total_pages}...",
            )

        progress.empty()
        document.close()

        return page_items, total_pages

    except Exception:
        raise


# =========================================================
# PDF CHUNKING AND ANALYSIS
# =========================================================

def build_pdf_chunks(page_items, pages_per_chunk=PDF_PAGES_PER_CHUNK):
    """Split PDF pages into manageable batches."""

    chunks = []

    for start in range(0, len(page_items), pages_per_chunk):
        chunks.append(page_items[start:start + pages_per_chunk])

    return chunks


def analyze_pdf_chunk(chunk, chunk_number, total_chunks):
    """Analyze one batch of PDF pages, including scanned pages."""

    page_text_parts = []

    for page in chunk:
        if page["text"]:
            page_text_parts.append(
                f"--- PDF Page {page['page_number']} ---\n"
                f"{page['text']}"
            )
        elif page["image"] is not None:
            page_text_parts.append(
                f"--- PDF Page {page['page_number']} ---\n"
                "This page is image-based. Analyze the attached page image."
            )
        else:
            page_text_parts.append(
                f"--- PDF Page {page['page_number']} ---\n"
                "No selectable text was found."
            )

    prompt = (
        f"{CHUNK_PROMPT}\n\n"
        f"Analyze PDF section {chunk_number} of {total_chunks}.\n\n"
        + "\n\n".join(page_text_parts)
    )

    chunk_input = create_text_input(prompt)

    # Add scanned pages belonging to this chunk.
    for page in chunk:
        if page["image"] is not None:
            encoded_image = base64.b64encode(
                page["image"]
            ).decode("utf-8")

            chunk_input.append(
                {
                    "type": "text",
                    "text": f"Scanned image of PDF page {page['page_number']}:",
                }
            )

            chunk_input.append(
                {
                    "type": "image",
                    "data": encoded_image,
                    "mime_type": "image/png",
                }
            )

    response_text = analyze_with_gemini(chunk_input)

    if response_text is None:
        raise RuntimeError(
            f"Gemini failed to analyze PDF section {chunk_number}."
        )

    try:
        return json.loads(clean_json_response(response_text))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Gemini returned invalid JSON for PDF section {chunk_number}."
        ) from error


def analyze_entire_pdf(page_items):
    """
    Analyze all PDF pages in batches and combine the results
    into one final study guide.
    """

    chunks = build_pdf_chunks(page_items)

    if not chunks:
        raise ValueError("No PDF pages were found to analyze.")

    all_chunk_notes = []

    progress = st.progress(0, text="Preparing PDF analysis...")

    for index, chunk in enumerate(chunks):
        chunk_number = index + 1

        progress.progress(
            index / len(chunks),
            text=(
                f"Analyzing PDF section {chunk_number} of "
                f"{len(chunks)}..."
            ),
        )

        chunk_result = analyze_pdf_chunk(
            chunk,
            chunk_number,
            len(chunks),
        )

        all_chunk_notes.append(
            {
                "section": chunk_number,
                "pages": [
                    page["page_number"] for page in chunk
                ],
                "notes": chunk_result,
            }
        )

    progress.progress(
        1.0,
        text="Combining all PDF sections into a study guide...",
    )

    # Compact notes from every section become the input to the final request.
    combined_notes = json.dumps(
        all_chunk_notes,
        ensure_ascii=False,
        indent=2,
    )

    final_input = create_text_input(
        FINAL_STUDY_PROMPT
        + "\n\nNOTES FROM ALL PDF SECTIONS:\n"
        + combined_notes
    )

    final_response = analyze_with_gemini(final_input)
    progress.empty()

    if final_response is None:
        raise RuntimeError("Could not create the final study guide.")

    try:
        final_data = json.loads(clean_json_response(final_response))
    except json.JSONDecodeError as error:
        raise ValueError(
            "Gemini returned invalid JSON for the final study guide."
        ) from error

    validate_study_data(final_data)

    return final_data, combined_notes


# =========================================================
# PDF NOTES EXPORT
# =========================================================

def create_notes_pdf(study_data):
    """Generate a downloadable PDF containing the study notes."""

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "StudyTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=18,
    )

    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = [
        Paragraph("Snap &amp; Study - Study Notes", title_style),
        Spacer(1, 12),
        Paragraph("Summary", heading_style),
        Paragraph(escape(study_data["summary"]), body_style),
        Spacer(1, 12),
        Paragraph("Key Points", heading_style),
    ]

    for point in study_data["key_points"]:
        story.append(Paragraph("• " + escape(str(point)), body_style))
        story.append(Spacer(1, 5))

    story.extend(
        [
            Spacer(1, 10),
            Paragraph("Important Questions and Answers", heading_style),
        ]
    )

    for index, item in enumerate(study_data["important_questions"], start=1):
        story.append(
            Paragraph(
                f"<b>Q{index}. {escape(str(item['question']))}</b>",
                body_style,
            )
        )
        story.append(
            Paragraph(
                f"<b>Answer:</b> {escape(str(item['answer']))}",
                body_style,
            )
        )
        story.append(Spacer(1, 10))

    document.build(story)

    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# QUIZ GENERATION
# =========================================================

def generate_new_quiz():
    """Generate five new MCQs based on the uploaded study material."""

    study_input = st.session_state.study_input
    source_text = st.session_state.study_source_text

    previous_questions = st.session_state.previous_quiz_questions

    quiz_prompt = f"""
You are an expert quiz creator.

Create exactly 5 NEW multiple-choice questions using the study material
provided below.

Return ONLY valid JSON. Do not use Markdown or code fences.

Use this structure:

{{
  "mcqs": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correct_answer": "Option A"
    }}
  ]
}}

Rules:
- Create exactly 5 MCQs.
- Every MCQ must have exactly 4 options.
- The correct answer must exactly match one option.
- Questions must be related to the uploaded study material.
- Avoid repeating any previous question.
- Use different questions on every reattempt.
- Do not invent facts beyond the supplied material.
- Keep the questions student-friendly.
- Return valid JSON only.

PREVIOUSLY USED QUESTIONS:
{json.dumps(previous_questions, ensure_ascii=False)}

"""

    # For a PDF, use the notes from every analyzed section.
    if isinstance(study_input, str):
        quiz_input = create_text_input(
            quiz_prompt
            + "\n\nSTUDY MATERIAL:\n"
            + study_input
        )

    # For an image upload, preserve the image while replacing the old prompt.
    elif isinstance(study_input, list):
        quiz_input = [
            {"type": "text", "text": quiz_prompt}
        ]

        for item in study_input:
            if item.get("type") == "image":
                quiz_input.append(item)

        if source_text:
            quiz_input.append(
                {
                    "type": "text",
                    "text": "Additional study notes:\n" + source_text,
                }
            )

    else:
        st.error("Study material is unavailable. Please upload it again.")
        return False

    with st.spinner("Generating a fresh set of MCQs..."):
        response_text = analyze_with_gemini(quiz_input)

    if response_text is None:
        return False

    try:
        quiz_data = json.loads(clean_json_response(response_text))
        new_mcqs = quiz_data["mcqs"]

        if len(new_mcqs) != 5:
            raise ValueError("Gemini did not generate exactly 5 MCQs.")

        for mcq in new_mcqs:
            if len(mcq.get("options", [])) != 4:
                raise ValueError("An MCQ does not contain four options.")

            if mcq.get("correct_answer") not in mcq["options"]:
                raise ValueError("An MCQ has an invalid correct answer.")

        st.session_state.study_data["mcqs"] = new_mcqs

        for mcq in new_mcqs:
            question = mcq["question"]

            if question not in st.session_state.previous_quiz_questions:
                st.session_state.previous_quiz_questions.append(question)

        st.session_state.quiz_current = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_answered = False
        st.session_state.quiz_finished = False
        st.session_state.quiz_feedback = ""
        st.session_state.quiz_selected = {}

        return True

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        st.error(f"Could not validate the new quiz: {error}")
        return False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("📚 Snap & Study")

    st.write("Study smarter with your own material.")

    if st.button("➕ New Study Session", use_container_width=True):
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value

        st.rerun()

    st.divider()
    st.caption("Supported files: PDF, JPG, JPEG, PNG")


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your study material",
    type=["pdf", "jpg", "jpeg", "png"],
    help="PDF files are analyzed in batches. Images are analyzed directly.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    is_pdf = uploaded_file.name.lower().endswith(".pdf")

    if is_pdf:
        st.info(
            "Your entire PDF will be analyzed in batches. "
            "Large PDFs may take longer because they require multiple "
            "Gemini requests."
        )
    else:
        st.image(
            file_bytes,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    if st.button("🚀 Analyze Study Material", type="primary"):
        st.session_state.study_data = None
        st.session_state.study_input = None
        st.session_state.study_source_text = ""
        st.session_state.previous_quiz_questions = []
        st.session_state.quiz_current = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_answered = False
        st.session_state.quiz_finished = False
        st.session_state.quiz_feedback = ""
        st.session_state.quiz_selected = {}
        st.session_state.analysis_done = False

        try:
            if is_pdf:
                with st.spinner("Extracting all PDF pages..."):
                    page_items, total_pages = extract_pdf_content(file_bytes)

                scanned_pages = sum(
                    1 for page in page_items if page["image"] is not None
                )

                chunks_count = (
                    len(page_items) + PDF_PAGES_PER_CHUNK - 1
                ) // PDF_PAGES_PER_CHUNK

                st.write(f"📄 Total PDF pages: **{total_pages}**")
                st.write(f"🧩 Analysis batches: **{chunks_count}**")
                st.write(f"🖼️ Scanned pages: **{scanned_pages}**")

                study_data, combined_notes = analyze_entire_pdf(page_items)

                # Store compact notes from every section for future quiz attempts.
                st.session_state.study_input = combined_notes
                st.session_state.study_source_text = combined_notes

            else:
                mime_type = (
                    "image/jpeg"
                    if uploaded_file.name.lower().endswith((".jpg", ".jpeg"))
                    else "image/png"
                )

                input_content = create_image_input(
                    STUDY_PROMPT,
                    file_bytes,
                    mime_type,
                )

                with st.spinner("Analyzing your image with Gemini..."):
                    response_text = analyze_with_gemini(input_content)

                if response_text is None:
                    st.stop()

                study_data = json.loads(
                    clean_json_response(response_text)
                )

                validate_study_data(study_data)

                st.session_state.study_input = input_content
                st.session_state.study_source_text = json.dumps(
                    study_data,
                    ensure_ascii=False,
                )

            st.session_state.study_data = study_data
            st.session_state.analysis_done = True

            # Record the original questions to avoid repeating them.
            st.session_state.previous_quiz_questions = [
                item["question"] for item in study_data["mcqs"]
            ]

            st.success("Your study guide is ready!")
            st.rerun()

        except Exception as error:
            st.error("The study material could not be analyzed.")
            st.write(
                "Please try again. If the PDF is very large, try a "
                "smaller document or check whether Gemini is available."
            )

            with st.expander("Technical details"):
                st.code(str(error))


# =========================================================
# STUDY GUIDE
# =========================================================

if st.session_state.study_data is not None:
    study_data = st.session_state.study_data

    st.divider()
    st.header("📖 Your Study Guide")

    summary_tab, points_tab, questions_tab, quiz_tab = st.tabs(
        [
            "📝 Summary",
            "🔑 Key Points",
            "❓ Important Questions",
            "🎯 MCQ Quiz",
        ]
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    with summary_tab:
        st.subheader("Smart Summary")
        st.write(study_data["summary"])

    # -----------------------------------------------------
    # KEY POINTS
    # -----------------------------------------------------

    with points_tab:
        st.subheader("Key Points")

        for index, point in enumerate(study_data["key_points"], start=1):
            st.markdown(f"**{index}.** {point}")

    # -----------------------------------------------------
    # IMPORTANT QUESTIONS
    # -----------------------------------------------------

    with questions_tab:
        st.subheader("Important Questions and Answers")

        for index, item in enumerate(
            study_data["important_questions"],
            start=1,
        ):
            with st.expander(f"Q{index}. {item['question']}"):
                st.write("**Answer:**")
                st.write(item["answer"])

    # -----------------------------------------------------
    # MCQ QUIZ
    # -----------------------------------------------------

    with quiz_tab:
        st.subheader("🎯 Test Your Knowledge")

        mcqs = study_data["mcqs"]
        total_questions = len(mcqs)

        if not st.session_state.quiz_finished:
            question_index = st.session_state.quiz_current
            current_mcq = mcqs[question_index]

            st.progress(
                (question_index + 1) / total_questions,
                text=(
                    f"Question {question_index + 1} "
                    f"of {total_questions}"
                ),
            )

            st.markdown(f"### {current_mcq['question']}")

            selected_answer = st.radio(
                "Choose your answer:",
                current_mcq["options"],
                key=f"quiz_answer_{question_index}",
                index=None,
                disabled=st.session_state.quiz_answered,
            )

            if not st.session_state.quiz_answered:
                if st.button("Check Answer", type="primary"):
                    if selected_answer is None:
                        st.warning("Please select an answer first.")
                    else:
                        st.session_state.quiz_selected[
                            question_index
                        ] = selected_answer

                        if selected_answer == current_mcq["correct_answer"]:
                            st.session_state.quiz_score += 1
                            st.session_state.quiz_feedback = (
                                "✅ Correct! Well done."
                            )
                        else:
                            st.session_state.quiz_feedback = (
                                "❌ Incorrect. The correct answer is: "
                                + current_mcq["correct_answer"]
                            )

                        st.session_state.quiz_answered = True
                        st.rerun()

            if st.session_state.quiz_answered:
                feedback = st.session_state.quiz_feedback

                if feedback.startswith("✅"):
                    st.success(feedback)
                else:
                    st.error(feedback)

                if question_index + 1 < total_questions:
                    if st.button("Next Question ➡️", type="primary"):
                        st.session_state.quiz_current += 1
                        st.session_state.quiz_answered = False
                        st.session_state.quiz_feedback = ""
                        st.rerun()
                else:
                    if st.button("Finish Quiz 🏁", type="primary"):
                        st.session_state.quiz_finished = True
                        st.rerun()

        else:
            score = st.session_state.quiz_score
            percentage = (score / total_questions) * 100

            st.header("🏆 Quiz Results")
            st.metric("Your Score", f"{score}/{total_questions}")
            st.metric("Accuracy", f"{percentage:.0f}%")

            if percentage < 60:
                st.warning(
                    "📚 Low Score! Keep Practicing. "
                    "Review your notes and try again."
                )
            elif percentage < 80:
                st.info(
                    "👍 Good effort! Practice more to improve your score."
                )
            else:
                st.success(
                    "🎉 Excellent work! You understand this topic well."
                )

            if st.button("🔄 Reattempt with New Questions", type="primary"):
                success = generate_new_quiz()

                if success:
                    st.success("A fresh set of questions is ready!")
                    st.rerun()

            st.caption(
                "Your next attempt generates different questions based "
                "on the analyzed study material."
            )

    # =====================================================
    # DOWNLOAD STUDY NOTES
    # =====================================================

    st.divider()
    st.subheader("📥 Download Your Study Notes")

    pdf_bytes = create_notes_pdf(study_data)

    st.download_button(
        label="Download Study Notes as PDF",
        data=pdf_bytes,
        file_name="Snap_and_Study_Notes.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📚 Snap & Study — Learn smarter, not harder."
)