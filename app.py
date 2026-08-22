import streamlit as st
from transformers import ( AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline )
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
from keybert import KeyBERT
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import numpy as np
import torch
import fitz
import re

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Multilingual News Intelligence",
    page_icon="📰",
    layout="wide"
)

# CUSTOM CSS
st.markdown(
    """
    <style>
      .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
      }
      .subtitle {
        text-align: center;
        color: gray;
        font-size: 1.1rem;
        margin-bottom: 2rem;
      }
      .feature-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 10px;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# TITLE
st.markdown(
    '<div class="main-title">📰 AI Multilingual News Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload any news images or PDFs, extract text, summarize, translate, '
    'analyze, ask questions, and generate speech.'
    '</div>',
    unsafe_allow_html=True
)

# SESSION STATE
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""

# LOAD OCR MODEL
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# LOAD SUMMARIZATION MODEL
@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# LOAD CATEGORY CLASSIFIER
@st.cache_resource
def load_category_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# LOAD SENTIMENT MODEL
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

# LOAD QUESTION ANSWERING MODEL
@st.cache_resource
def load_qa_model():
    model_name = "deepset/roberta-base-squad2"
    qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return qa_tokenizer, qa_model


# LOAD KEYWORD MODEL
@st.cache_resource
def load_keyword_model():
    return KeyBERT()


# INITIALIZE MODELS
with st.spinner("🚀 Loading AI models..."):
    reader = load_ocr()
    tokenizer, summarization_model = load_summarization_model()
    category_classifier = load_category_classifier()
    sentiment_model = load_sentiment_model()
    qa_tokenizer, qa_model = load_qa_model()
    keyword_model = load_keyword_model()

# IMAGE PREPROCESSING
def preprocess_image(image):
    image = image.convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    image = image.filter(ImageFilter.SHARPEN)
    return image

# OCR FUNCTION
def extract_text_from_image(image):
    processed_image = preprocess_image(image)
    image_array = np.array(processed_image)
    results = reader.readtext(image_array)
    text = " ".join(result[1] for result in results)
    return text

# PDF TEXT EXTRACTION
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_bytes = pdf_file.read()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in document:
        page_text = page.get_text()
        text += page_text + "\n"
    document.close()
    return text

# TEXT CLEANING
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# CHUNK TEXT
def chunk_text(text, chunk_size=900):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# SUMMARIZATION FUNCTION
def summarize_text(text, summary_type):
    text = clean_text(text)
    chunks = chunk_text(text)
    summaries = []

    if summary_type == "Short Summary":
        max_length, min_length = 80, 25
    elif summary_type == "Detailed Summary":
        max_length, min_length = 180, 60
    elif summary_type == "Bullet Points":
        max_length, min_length = 150, 50
    else:
        max_length, min_length = 120, 40

    for chunk in chunks[:5]:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        with torch.no_grad():
            summary_ids = summarization_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    final_summary = " ".join(summaries)

    # Bullet points mode
    if summary_type == "Bullet Points":
        sentences = re.split(r"(?<=[.!?])\s+", final_summary)
        final_summary = "\n".join(
            f"• {sentence}" for sentence in sentences if sentence.strip()
        )

    # Key facts mode
    elif summary_type == "5 Key Facts":
        sentences = re.split(r"(?<=[.!?])\s+", final_summary)[:5]
        final_summary = "\n".join(
            f"{i + 1}. {sentence}" for i, sentence in enumerate(sentences)
        )

    return final_summary


# TRANSLATION
def translate_text(text, target_language):
    language_codes = {
        "English": "en",
        "Nepali": "ne",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh-CN",
        "Japanese": "ja",
        "Arabic": "ar"
    }
    target_code = language_codes[target_language]

    # GoogleTranslator has limits for long text
    if len(text) > 4000:
        text = text[:4000]

    translated = GoogleTranslator(source="auto", target=target_code).translate(text)
    return translated, target_code

# CATEGORY CLASSIFICATION
def classify_news(text):
    categories = [
        "Politics", "Sports", "Technology", "Business", "Health",
        "Agriculture", "Entertainment", "International", "Education",
        "Environment", "Crime", "Science"
    ]
    result = category_classifier(text[:2000], candidate_labels=categories)
    return result["labels"][0], result["scores"][0]

# SENTIMENT ANALYSIS
def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    return result["label"], result["score"]

# KEYWORD EXTRACTION
def extract_keywords(text):
    keywords = keyword_model.extract_keywords(
        text[:5000],
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=8
    )
    return [keyword[0] for keyword in keywords]

# QUESTION ANSWERING
def answer_question(question, context):
    words = context.split()
    chunk_size = 350
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    best_answer, best_confidence = "", 0.0
    for chunk in chunks:
        inputs = qa_tokenizer(question, chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)

        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        start_index = torch.argmax(start_logits, dim=-1)[0].item()
        end_index = torch.argmax(end_logits, dim=-1)[0].item()

        # Fix invalid range
        if end_index < start_index:
            end_index = start_index
        if end_index - start_index > 30:
            end_index = start_index + 30

        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[start_index:end_index + 1]
        answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

        start_prob = torch.softmax(start_logits, dim=-1)[0][start_index].item()
        end_prob = torch.softmax(end_logits, dim=-1)[0][end_index].item()
        confidence = start_prob * end_prob

        if confidence > best_confidence and answer.strip() and answer.lower() != question.lower():
            best_confidence, best_answer = confidence, answer

    if not best_answer:
        best_answer = "I could not find a clear answer in the uploaded article."
        best_confidence = 0.0

    return best_answer, best_confidence

# TEXT TO SPEECH
def generate_speech(text, language_code):
    supported_languages = {"en", "ne", "hi", "es", "fr", "de", "zh-CN", "ja", "ar"}
    if language_code not in supported_languages:
        language_code = "en"

    tts = gTTS(text=text, lang=language_code)
    mp3_bytes = BytesIO()
    tts.write_to_fp(mp3_bytes)
    mp3_bytes.seek(0)
    return mp3_bytes.getvalue()

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Processing Options")

    summary_type = st.selectbox(
        "Summary Type",
        ["Short Summary", "Detailed Summary", "Bullet Points", "5 Key Facts"]
    )

    target_language = st.selectbox(
        "Translate To",
        [
            "Nepali 🇳🇵", "English 🇬🇧", "Hindi 🇮🇳", "Spanish 🇪🇸",
            "French 🇫🇷", "German 🇩🇪", "Chinese 🇨🇳",
            "Japanese 🇯🇵", "Arabic 🇸🇦"
        ]
    )
    # Remove emoji from language selection
    target_language = target_language.split()[0]

    st.divider()
    st.subheader("🤖 AI Features")

    enable_category = st.checkbox("News Category", value=True)
    enable_sentiment = st.checkbox("Sentiment Analysis", value=True)
    enable_keywords = st.checkbox("Keyword Extraction", value=True)
    enable_translation = st.checkbox("Translation", value=True)
    enable_speech = st.checkbox("Text to Speech", value=True)

# INPUT SECTION
st.subheader("📤 Upload News Content")
input_type = st.radio(
    "Choose Input Type",
    ["🖼️ Image", "📄 PDF", "✍️ Paste Text"],
    horizontal=True
)

extracted_text = ""

# IMAGE INPUT
if input_type == "🖼️ Image":
    uploaded_file = st.file_uploader("Upload a news image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("🔍 Extracting text using OCR..."):
            extracted_text = extract_text_from_image(image)

        with col2:
            processed = preprocess_image(image)
            st.image(processed, caption="Processed Image", use_container_width=True)

# PDF INPUT
elif input_type == "📄 PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("📄 Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

# TEXT INPUT
elif input_type == "✍️ Paste Text":
    extracted_text = st.text_area(
        "Paste your news article",
        height=300,
        placeholder="Paste the complete news article here..."
    )

# PROCESS CONTENT
if extracted_text and extracted_text.strip():
    st.divider()
    st.subheader("📄 Extracted / Input Text")

    edited_text = st.text_area(
        "You can edit and correct the text before AI processing",
        value=extracted_text,
        height=300
    )
    st.session_state.processed_text = edited_text

    word_count = len(edited_text.split())
    character_count = len(edited_text)

    col1, col2, col3 = st.columns(3)
    col1.metric("Words", word_count)
    col2.metric("Characters", character_count)
    col3.metric("Input Type", input_type)

    st.divider()

    # ANALYZE BUTTON
    if st.button("🚀 Analyze News Article", type="primary", use_container_width=True):
        if len(edited_text.split()) < 20:
            st.warning("⚠️ Please provide at least 20 words.")
        else:
            # SUMMARIZATION
            with st.spinner("🧠 Generating AI summary..."):
                summary = summarize_text(edited_text, summary_type)
            st.session_state.summary = summary

            # TRANSLATION
            translated = ""
            if enable_translation:
                with st.spinner("🌐 Translating summary..."):
                    translated, language_code = translate_text(summary, target_language)
                st.session_state.translated_text = translated

            # CATEGORY
            if enable_category:
                with st.spinner("🏷️ Detecting news category..."):
                    category, category_score = classify_news(edited_text)
                st.session_state.category = category
                st.session_state.category_score = category_score

            # SENTIMENT
            if enable_sentiment:
                with st.spinner("😊 Analyzing sentiment..."):
                    sentiment, sentiment_score = analyze_sentiment(edited_text)
                st.session_state.sentiment = sentiment
                st.session_state.sentiment_score = sentiment_score

            # KEYWORDS
            if enable_keywords:
                with st.spinner("🔑 Extracting keywords..."):
                    keywords = extract_keywords(edited_text)
                st.session_state.keywords = keywords

            # TEXT TO SPEECH
            if enable_speech and translated:
                with st.spinner("🔊 Generating speech..."):
                    audio_bytes = generate_speech(translated, language_code)
                st.session_state.audio_bytes = audio_bytes

            st.success("✅ News analysis completed successfully!")

# RESULTS SECTION
if st.session_state.summary:
    st.divider()
    st.header("📊 AI Analysis Results")

    # METRICS
    metrics = st.columns(3)
    if hasattr(st.session_state, "category"):
        metrics[0].metric(
            "🏷️ Category",
            st.session_state.category,
            f"{st.session_state.category_score:.0%} confidence"
        )
    if hasattr(st.session_state, "sentiment"):
        metrics[1].metric(
            "😊 Sentiment",
            st.session_state.sentiment,
            f"{st.session_state.sentiment_score:.0%} confidence"
        )
    metrics[2].metric("📝 Summary Type", summary_type)

    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🧠 Summary", "🌐 Translation", "🔑 Keywords", "🤖 Ask AI", "🔊 Speech & Download"]
    )

    # SUMMARY TAB
    with tab1:
        st.subheader("🧠 AI Generated Summary")
        st.write(st.session_state.summary)
        st.download_button(
            "⬇️ Download Summary",
            st.session_state.summary,
            file_name="news_summary.txt",
            mime="text/plain"
        )

    # TRANSLATION TAB
    with tab2:
        if st.session_state.translated_text:
            st.subheader(f"🌐 {target_language} Translation")
            st.write(st.session_state.translated_text)
            st.download_button(
                "⬇️ Download Translation",
                st.session_state.translated_text,
                file_name="translated_news.txt",
                mime="text/plain"
            )
        else:
            st.info("Translation is disabled.")

    # KEYWORDS TAB
    with tab3:
        st.subheader("🔑 Important Keywords")
        if hasattr(st.session_state, "keywords"):
            columns = st.columns(4)
            for index, keyword in enumerate(st.session_state.keywords):
                with columns[index % 4]:
                    st.button(f"🔑 {keyword}", key=f"keyword_{index}")
        else:
            st.info("Keyword extraction is disabled.")

    # QUESTION ANSWERING TAB
    with tab4:
        st.subheader("🤖 Ask Questions About the Article")
        st.write("Ask anything based on the uploaded news content.")
        question = st.text_input("Your Question", placeholder="Example: What happened in this news?")
        if st.button("🔍 Get Answer"):
            if question.strip():
                with st.spinner("🤖 Finding the answer..."):
                    answer, score = answer_question(question, st.session_state.processed_text)
                st.success(answer)
                st.caption(f"Confidence: {score:.2%}")
            else:
                st.warning("Please enter a question.")

    # SPEECH TAB
    with tab5:
        st.subheader("🔊 AI Generated Speech")
        if st.session_state.audio_bytes:
            st.audio(st.session_state.audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download Audio",
                st.session_state.audio_bytes,
                file_name="news_audio.mp3",
                mime="audio/mp3"
            )
        else:
            st.info("Speech generation is disabled.")

# FOOTER
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:gray;">
        OCR • NLP • Summarization • Translation • Classification •
        Sentiment • Keywords • Question Answering • Text-to-Speech
    </div>
    """,
    unsafe_allow_html=True
)


# To run this
# streamlit run app.py --server.address=0.0.0.0 --server.port=8501
