import streamlit as st
import json
import time
import re
from datetime import datetime
from transformers import pipeline
from llama_cpp import Llama
import plotly.graph_objects as go
from textblob import TextBlob
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🤖 TalentScout - AI Hiring Assistant", 
    page_icon="🧑‍💻", 
    layout="centered"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stChat {
        background-color: #f0f2f5;
        border-radius: 10px;
        padding: 10px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .score-badge {
        background-color: #007bff;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# CONSTANTS
# -------------------------------
EXIT_KEYWORDS = ['exit', 'quit', 'bye', 'stop', 'cancel', 'end']

# Predefined question bank
QUESTION_BANK = {
    "python": [
        "What is the difference between a list and a tuple in Python?",
        "How do you handle exceptions in Python? Can you give an example?",
        "Explain list comprehension in Python with an example.",
        "What are Python decorators and when would you use them?",
        "How does memory management work in Python?",
        "What is the Global Interpreter Lock (GIL) in Python?"
    ],
    "sql": [
        "What is the difference between INNER JOIN and LEFT JOIN in SQL?",
        "How would you optimize a slow-running SQL query?",
        "Explain database normalization and why it's important.",
        "What are indexes and how do they improve query performance?",
        "Describe the difference between DELETE and TRUNCATE.",
        "What are stored procedures and when would you use them?"
    ],
    "machine learning": [
        "What is the difference between supervised and unsupervised learning?",
        "How do you handle overfitting in machine learning models?",
        "Explain the bias-variance tradeoff in machine learning.",
        "What is cross-validation and why is it important?",
        "How do you evaluate the performance of a classification model?",
        "What is feature engineering and why is it important?"
    ],
    "tensorflow": [
        "What is the difference between TensorFlow and Keras?",
        "Explain what tensors are in the context of TensorFlow.",
        "How do you prevent overfitting in neural networks?",
        "What are the different types of activation functions and their uses?",
        "How does backpropagation work in neural networks?",
        "What is the purpose of batch normalization?"
    ],
    "javascript": [
        "What is the difference between let, const, and var in JavaScript?",
        "Explain closures in JavaScript with an example.",
        "What is the difference between == and === in JavaScript?",
        "How does asynchronous programming work in JavaScript?",
        "What are promises and how do they differ from callbacks?",
        "Explain the concept of hoisting in JavaScript."
    ],
    "react": [
        "What is the difference between state and props in React?",
        "Explain the React component lifecycle methods.",
        "What are React hooks and how do they work?",
        "How does the virtual DOM work in React?",
        "What is Redux and when would you use it?",
        "How do you optimize performance in React applications?"
    ],
    "general": [
        "Describe a challenging technical problem you solved recently.",
        "How do you approach debugging complex issues?",
        "What is your experience with version control systems?",
        "How do you ensure code quality in your projects?",
        "Describe your experience working in a team environment.",
        "How do you stay updated with new technologies?"
    ]
}

# -------------------------------
# INITIALIZE SESSION STATE
# -------------------------------
if 'step' not in st.session_state:
    st.session_state.step = "greeting"
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
if 'candidate' not in st.session_state:
    st.session_state.candidate = {}
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = {}
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_llm():
    """Load the LLM model"""
    try:
        llm = Llama.from_pretrained(
            repo_id="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
            filename="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )
        return llm
    except:
        return None

@st.cache_resource
def load_ai_detector():
    """Load AI text detector model"""
    try:
        # Use a different model that's more reliable
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
        model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.warning(f"AI detector not available: {str(e)}")
        return None

# Load models
llm = load_llm()
ai_detector = load_ai_detector()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def add_message(role, content):
    """Add message to conversation history"""
    st.session_state.messages.append({
        "role": role, 
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def display_chat():
    """Display conversation history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def check_exit(user_input):
    """Check if user wants to exit"""
    return any(keyword in user_input.lower() for keyword in EXIT_KEYWORDS)

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number"""
    digits = re.sub(r'\D', '', phone)
    return len(digits) >= 10

def validate_experience(exp):
    """Validate years of experience"""
    try:
        years = float(exp)
        return 0 <= years <= 50
    except:
        return False

def get_fallback_response(step):
    """Get fallback response for invalid input"""
    fallbacks = {
        "name": "I need your full name to proceed. Please enter your first and last name.",
        "email": "Please provide a valid email address (e.g., john@example.com)",
        "phone": "Please provide a valid phone number with at least 10 digits.",
        "exp": "Please enter your years of experience as a number (e.g., 3 or 5.5)",
        "role": "What position are you interested in? (e.g., Full Stack Developer, Data Scientist)",
        "loc": "Where are you currently located? (City, Country)",
        "stack": "Please list your technical skills separated by commas (e.g., Python, React, AWS)"
    }
    return fallbacks.get(step, "I didn't understand that. Could you please try again?")

def get_questions_for_stack(stack_string, role, experience):
    """Get relevant questions based on tech stack"""
    tech_items = [tech.strip().lower() for tech in stack_string.split(',')]
    questions = []
    used_questions = set()
    
    # Get questions for each technology
    for tech in tech_items:
        # Check if we have questions for this tech
        for key in QUESTION_BANK:
            if key in tech or tech in key:
                available_questions = [q for q in QUESTION_BANK[key] if q not in used_questions]
                if available_questions:
                    questions.append(available_questions[0])
                    used_questions.add(available_questions[0])
                    break
        
        if len(questions) >= 4:
            break
    
    # If we need more questions, add from general category
    if len(questions) < 4:
        general_questions = [q for q in QUESTION_BANK["general"] if q not in used_questions]
        for q in general_questions:
            questions.append(q)
            if len(questions) >= 4:
                break
    
    # Ensure we have exactly 4 questions
    return questions[:4]

def llm_completion(prompt, max_tokens=400, temp=0.7):
    """Get completion from LLM with proper timeout"""
    if not llm:
        return None
    
    try:
        # Use threading to add timeout
        import threading
        result = [None]
        exception = [None]
        
        def run_completion():
            try:
                res = llm.create_completion(
                    prompt=prompt, 
                    max_tokens=max_tokens, 
                    temperature=temp,
                    stop=["\n\n"]
                )
                result[0] = res["choices"][0]["text"].strip()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=run_completion)
        thread.start()
        thread.join(timeout=15.0)  # 15 second timeout
        
        if thread.is_alive():
            return None  # Timeout
        if exception[0]:
            return None  # Error
        return result[0]
    except:
        return None

def evaluate_answer(question: str, answer: str, tech_stack: str):
    """Lightweight answer evaluation without LLM"""
    score = 5  # Base score
    answer_length = len(answer.strip())
    
    # Length-based scoring
    if answer_length < 30:
        score = 3
        comment = "Answer is too brief. More detail would strengthen your response."
    elif answer_length < 100:
        score = 5
        comment = "Acceptable answer, but could benefit from more examples."
    elif answer_length < 300:
        score = 7
        comment = "Good detailed response showing understanding."
    else:
        score = 8
        comment = "Comprehensive answer with excellent detail."
    
    # Keyword bonus
    tech_keywords = [tech.strip().lower() for tech in tech_stack.split(',')[:3]]
    answer_lower = answer.lower()
    
    # Check for technical keywords
    keyword_count = sum(1 for keyword in tech_keywords if keyword in answer_lower)
    if keyword_count > 0:
        score = min(10, score + keyword_count)
        comment += " Good use of relevant technologies."
    
    # Quality indicators
    quality_phrases = ["for example", "such as", "in my experience", "specifically", "because"]
    if any(phrase in answer_lower for phrase in quality_phrases):
        score = min(10, score + 1)
        comment = "Excellent answer with specific examples and clear explanation."
    
    return {"score": score, "comment": comment}

def detect_ai_text(answer):
    """Detect if text is AI-generated"""
    if not ai_detector or len(answer) < 50:
        return {"label": "Not Checked", "score": 0}
    
    try:
        result = ai_detector(answer[:512])[0]
        return {"label": result['label'], "score": round(result['score'], 2)}
    except:
        return {"label": "Check Failed", "score": 0}

def generate_summary(candidate_info, evaluations, answers):
    """Generate comprehensive HR summary with LLM and fallback"""
    avg_score = round(
        sum([e["score"] for e in evaluations.values()]) / len(evaluations), 2
    ) if evaluations else 0
    
    # Check for AI-generated answers
    ai_flags = sum(1 for a in answers.values() if a.get("ai_check", {}).get("label") == "FAKE")
    
    # First try LLM generation
    if llm:
        # Build detailed context
        eval_details = []
        for i, (q_num, eval_info) in enumerate(evaluations.items()):
            answer_info = answers.get(q_num, {})
            question = answer_info.get('question', f'Question {i+1}')
            eval_details.append(f"{q_num} ({question[:50]}...): Score {eval_info['score']}/10 - {eval_info['comment']}")
        
        summary_prompt = f"""Generate a detailed HR assessment for this technical screening:

Candidate: {candidate_info.get('Name', 'Unknown')}
Role: {candidate_info.get('Role', 'Unknown')}
Experience: {candidate_info.get('Experience', '0')} years
Tech Stack: {candidate_info.get('Stack', 'Not specified')}
Average Score: {avg_score}/10
Location: {candidate_info.get('Location', 'Unknown')}

Detailed Performance:
{chr(10).join(eval_details)}

Provide a comprehensive assessment with:
1. Overall Assessment (2-3 sentences about performance)
2. Technical Strengths (3 specific points based on their answers)
3. Areas for Improvement (2-3 specific points)
4. Cultural Fit Indicators
5. Risk Factors (including {ai_flags} AI-generated answers if any)
6. Recommendation: [Strongly Recommend / Recommend / Maybe / Do Not Recommend] with clear reasoning

Be specific and reference their actual performance. Format with bullet points."""

        summary = llm_completion(summary_prompt, max_tokens=600, temp=0.5)
        
        if summary and len(summary) > 100:  # Ensure we got a meaningful response
            return summary
    
    # Enhanced fallback template
    high_scores = [e for e in evaluations.items() if e[1]['score'] >= 7]
    low_scores = [e for e in evaluations.items() if e[1]['score'] <= 4]
    
    # Determine recommendation with nuance
    if avg_score >= 8:
        recommendation = "Strongly Recommend"
        assessment = "Exceptional performance demonstrating deep technical knowledge and clear communication."
        next_steps = "Fast-track to final interview with senior technical team"
    elif avg_score >= 6.5:
        recommendation = "Recommend"
        assessment = "Solid technical foundation with good problem-solving abilities."
        next_steps = "Schedule technical deep-dive with team lead"
    elif avg_score >= 5:
        recommendation = "Maybe - Further Evaluation Needed"
        assessment = "Shows potential but has knowledge gaps in key areas."
        next_steps = "Consider for junior role or additional technical screening"
    else:
        recommendation = "Do Not Recommend"
        assessment = "Significant gaps in required technical knowledge."
        next_steps = "Provide constructive feedback and suggest areas for improvement"
    
    # Build specific feedback
    strengths = []
    weaknesses = []
    
    for i, (q_num, eval_info) in enumerate(evaluations.items()):
        question_topic = answers.get(q_num, {}).get('question', f'Question {i+1}')[:50] + "..."
        if eval_info['score'] >= 7:
            strengths.append(f"Strong answer on '{question_topic}' - {eval_info['comment']}")
        elif eval_info['score'] <= 4:
            weaknesses.append(f"Struggled with '{question_topic}' - {eval_info['comment']}")
    
    return f"""**🎯 Detailed HR Assessment Report**

**📊 Overall Performance**
• **Assessment:** {assessment}
• **Average Score:** {avg_score}/10
• **Recommendation:** {recommendation}

**💪 Technical Strengths:**
{chr(10).join([f"• {s}" for s in strengths[:3]]) if strengths else "• Demonstrated effort in completing all questions\n• Professional communication throughout"}

**📈 Areas for Development:**
{chr(10).join([f"• {w}" for w in weaknesses[:3]]) if weaknesses else "• Could provide more concrete examples\n• Consider elaborating on technical decisions"}

**🎓 Candidate Profile Analysis:**
• **Role Fit:** {"Excellent" if avg_score >= 7 else "Good" if avg_score >= 5 else "Needs development"} match for {candidate_info.get('Role', 'the role')}
• **Experience Level:** {candidate_info.get('Experience', '0')} years - {"Exceeds" if avg_score >= 8 else "Meets" if avg_score >= 6 else "Below"} expectations
• **Technical Breadth:** {len(candidate_info.get('Stack', '').split(','))} technologies listed
• **Communication:** {"Excellent" if avg_score >= 7 else "Good" if avg_score >= 5 else "Needs improvement"} technical communication skills

**🔍 Quality Indicators:**
• **Consistency:** {len(high_scores)} high-scoring answers, {len(low_scores)} low-scoring answers
• **AI Detection:** {ai_flags} potentially AI-generated {"answer" if ai_flags == 1 else "answers"} {"⚠️" if ai_flags > 0 else "✅"}
• **Response Quality:** {"Detailed and thoughtful" if avg_score >= 7 else "Adequate" if avg_score >= 5 else "Minimal effort"}

**🚀 Recommended Next Steps:** {next_steps}

**📝 Additional Notes:**
• Interview completed in {len(answers)} questions
• {candidate_info.get('Name', 'Candidate')} from {candidate_info.get('Location', 'Unknown location')}
• {"Strong cultural fit potential" if avg_score >= 7 else "Average cultural fit" if avg_score >= 5 else "Cultural fit concerns"}
"""

def show_progress():
    """Show progress indicator"""
    steps = ["Name", "Email", "Phone", "Experience", "Role", "Location", "Tech Stack", "Questions"]
    current_index = 0
    
    step_mapping = {
        "name": 0, "email": 1, "phone": 2, "exp": 3,
        "role": 4, "loc": 5, "stack": 6, "questions": 7
    }
    
    if st.session_state.step in step_mapping:
        current_index = step_mapping[st.session_state.step]
    
    progress = (current_index + 1) / len(steps)
    st.progress(progress)
    st.caption(f"Step {current_index + 1} of {len(steps)}")

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.markdown("### 🎯 TalentScout AI")
    st.markdown("---")
    
    if st.session_state.step != "greeting":
        st.markdown("#### 📊 Progress")
        show_progress()
        st.markdown("---")
    
    if st.session_state.candidate:
        st.markdown("#### 👤 Candidate Info")
        for key, value in st.session_state.candidate.items():
            st.write(f"**{key}:** {value}")
    
    st.markdown("---")
    st.caption("Type 'exit' anytime to end")

# -------------------------------
# MAIN CHAT INTERFACE
# -------------------------------
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🤖 TalentScout AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Intelligent screening for tech talent</p>", unsafe_allow_html=True)
st.markdown("---")

# Display chat history
display_chat()

# Initialize greeting
# Initialize greeting
if st.session_state.step == "greeting" and not st.session_state.messages:
    greeting = """👋 Hello! I'm TalentScout AI, your intelligent hiring assistant. 
    
I'll help you complete your technical screening in a few simple steps:
1. Basic information collection
2. Technical assessment based on your skills
3. Evaluation and feedback

Let's begin! What's your **full name**?"""
    add_message("assistant", greeting)
    st.session_state.step = "name"
    st.rerun()

# Chat input
if st.session_state.step != "summary":
    user_input = st.chat_input("Type your response here...")
    
    if user_input:
        # Check for exit commands
        if check_exit(user_input):
            add_message("user", user_input)
            add_message("assistant", "Thank you for your time! Feel free to return anytime. Goodbye! 👋")
            st.session_state.step = "ended"
            st.rerun()
        
        # Add user message
        add_message("user", user_input)
        
        # Process based on current step
        if st.session_state.step == "name":
            if len(user_input.strip()) >= 2:
                st.session_state.candidate["Name"] = user_input
                add_message("assistant", f"Nice to meet you, {user_input}! 📧 What's your **email address**?")
                st.session_state.step = "email"
            else:
                add_message("assistant", get_fallback_response("name"))
            st.rerun()
            
        elif st.session_state.step == "email":
            if validate_email(user_input):
                st.session_state.candidate["Email"] = user_input
                add_message("assistant", "Perfect! 📱 What's your **phone number**?")
                st.session_state.step = "phone"
            else:
                add_message("assistant", get_fallback_response("email"))
            st.rerun()
            
        elif st.session_state.step == "phone":
            if validate_phone(user_input):
                st.session_state.candidate["Phone"] = user_input
                add_message("assistant", "Great! 📊 How many **years of experience** do you have in tech?")
                st.session_state.step = "exp"
            else:
                add_message("assistant", get_fallback_response("phone"))
            st.rerun()
            
        elif st.session_state.step == "exp":
            if validate_experience(user_input):
                st.session_state.candidate["Experience"] = user_input
                add_message("assistant", "Excellent! 💼 What **position** are you applying for?")
                st.session_state.step = "role"
            else:
                add_message("assistant", get_fallback_response("exp"))
            st.rerun()
            
        elif st.session_state.step == "role":
            if len(user_input.strip()) >= 2:
                st.session_state.candidate["Role"] = user_input
                add_message("assistant", "Perfect! 📍 Where are you currently **located**? (City, Country)")
                st.session_state.step = "loc"
            else:
                add_message("assistant", get_fallback_response("role"))
            st.rerun()
            
        elif st.session_state.step == "loc":
            if len(user_input.strip()) >= 2:
                st.session_state.candidate["Location"] = user_input
                add_message("assistant", """Almost done! 💻 
                
Please list your **technical skills** separated by commas. 
For example: Python, Django, PostgreSQL, AWS, Docker

This will help me generate relevant technical questions for you.""")
                st.session_state.step = "stack"
            else:
                add_message("assistant", get_fallback_response("loc"))
            st.rerun()
            
        elif st.session_state.step == "stack":
            if len(user_input.strip()) > 2 and "," in user_input:
                st.session_state.candidate["Stack"] = user_input
                
                # Generate questions
                questions = get_questions_for_stack(
                    user_input,
                    st.session_state.candidate.get("Role", ""),
                    st.session_state.candidate.get("Experience", "0")
                )
                st.session_state.questions = questions
                
                # Display all questions
                add_message("assistant", f"""Excellent! I've prepared {len(questions)} technical questions based on your skills.

📝 **Your Technical Assessment Questions:**""")
                
                questions_text = "\n\n".join([f"**Question {i+1}:** {q}" for i, q in enumerate(questions)])
                add_message("assistant", questions_text)
                
                add_message("assistant", "Please answer **Question 1** now:")
                add_message("assistant", f"**{questions[0]}**")
                
                st.session_state.step = "questions"
                st.session_state.current_q = 0
            else:
                add_message("assistant", get_fallback_response("stack"))
            st.rerun()
            
        elif st.session_state.step == "questions":
            q_index = st.session_state.current_q
            
            if q_index < len(st.session_state.questions):
                current_question = st.session_state.questions[q_index]
                
                # Process answer
                with st.spinner("🔍 Evaluating your response..."):
                    # AI detection
                    ai_check = detect_ai_text(user_input)
                    
                    # Evaluate answer
                    evaluation = evaluate_answer(
                        current_question, 
                        user_input,
                        st.session_state.candidate.get("Stack", "")
                    )
                    
                    # Store results
                    st.session_state.answers[f"Q{q_index+1}"] = {
                        "question": current_question,
                        "answer": user_input,
                        "ai_check": ai_check
                    }
                    st.session_state.evaluations[f"Q{q_index+1}"] = evaluation
                
                # Show feedback
                score_color = "🟢" if evaluation["score"] >= 7 else "🟡" if evaluation["score"] >= 5 else "🔴"
                add_message("assistant", f"""✅ Answer received for Question {q_index + 1}!

{score_color} **Score: {evaluation['score']}/10**

**Feedback:** {evaluation['comment']}""")
                
                # Next question or finish
                st.session_state.current_q += 1
                
                if st.session_state.current_q < len(st.session_state.questions):
                    next_q = st.session_state.questions[st.session_state.current_q]
                    add_message("assistant", f"Please answer **Question {st.session_state.current_q + 1}:**")
                    add_message("assistant", f"**{next_q}**")
                else:
                    add_message("assistant", "🎉 Excellent! You've completed all questions. Preparing your assessment...")
                    st.session_state.step = "summary"
                
            st.rerun()

# Summary section (continued)
elif st.session_state.step == "summary":
    if not st.session_state.summary:
        with st.spinner("📊 Generating assessment report..."):
            st.session_state.summary = generate_summary(
                st.session_state.candidate,
                st.session_state.evaluations,
                st.session_state.answers
            )
            
            # Calculate final score
            avg_score = round(
                sum([e["score"] for e in st.session_state.evaluations.values()]) / len(st.session_state.evaluations),
                2
            ) if st.session_state.evaluations else 0
            st.session_state.candidate["Average Score"] = avg_score
    
    # Display summary
    st.success("✅ **Screening Complete!**")
    st.markdown("### 📋 Assessment Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{st.session_state.candidate.get('Average Score', 0)}/10")
    with col2:
        duration = (datetime.now() - st.session_state.start_time).total_seconds() / 60
        st.metric("Duration", f"{round(duration, 1)} min")
    with col3:
        total_questions = len(st.session_state.questions)
        st.metric("Questions", total_questions)
    
    # Display candidate info
    st.markdown("### 👤 Candidate Information")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.write(f"**Name:** {st.session_state.candidate.get('Name', 'N/A')}")
        st.write(f"**Email:** {st.session_state.candidate.get('Email', 'N/A')}")
        st.write(f"**Phone:** {st.session_state.candidate.get('Phone', 'N/A')}")
    with info_col2:
        st.write(f"**Role:** {st.session_state.candidate.get('Role', 'N/A')}")
        st.write(f"**Experience:** {st.session_state.candidate.get('Experience', 'N/A')} years")
        st.write(f"**Location:** {st.session_state.candidate.get('Location', 'N/A')}")
    
    st.write(f"**Tech Stack:** {st.session_state.candidate.get('Stack', 'N/A')}")
    
    # Display detailed evaluation
    st.markdown("### 📊 Detailed Evaluation")
    
    # Create evaluation dataframe
    eval_data = []
    for q_num, eval_info in st.session_state.evaluations.items():
        answer_info = st.session_state.answers.get(q_num, {})
        eval_data.append({
            "Question": q_num,
            "Score": f"{eval_info['score']}/10",
            "AI Detection": answer_info.get("ai_check", {}).get("label", "N/A"),
            "Feedback": eval_info["comment"]
        })
    
    df = pd.DataFrame(eval_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Display HR summary
    st.markdown("### 📝 HR Summary Report")
    st.info(st.session_state.summary)
    
    # Q&A Details Expander
    with st.expander("📋 View Detailed Q&A"):
        for q_num, answer_info in st.session_state.answers.items():
            st.markdown(f"**{q_num}: {answer_info['question']}**")
            st.write(f"*Answer:* {answer_info['answer']}")
            st.write(f"*Score:* {st.session_state.evaluations[q_num]['score']}/10")
            st.write(f"*AI Check:* {answer_info['ai_check']['label']}")
            st.markdown("---")
    
    # Download options
    st.markdown("### 📥 Download Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare detailed report
        report_data = {
            "meta": {
                "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_minutes": round(duration, 2),
                "version": "1.0"
            },
            "candidate": st.session_state.candidate,
            "technical_assessment": {
                "questions": st.session_state.questions,
                "answers": st.session_state.answers,
                "evaluations": st.session_state.evaluations,
                "average_score": st.session_state.candidate.get("Average Score", 0)
            },
            "summary": st.session_state.summary,
            "conversation_log": st.session_state.messages
        }
        
        st.download_button(
            label="📄 Download Full Report (JSON)",
            data=json.dumps(report_data, indent=2),
            file_name=f"{st.session_state.candidate.get('Name', 'candidate').replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Complete screening data in JSON format"
        )
    
    with col2:
        # Create summary text
        summary_text = f"""TALENTSCOUT AI SCREENING REPORT
================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CANDIDATE INFORMATION
--------------------
Name: {st.session_state.candidate.get('Name', 'N/A')}
Email: {st.session_state.candidate.get('Email', 'N/A')}
Phone: {st.session_state.candidate.get('Phone', 'N/A')}
Location: {st.session_state.candidate.get('Location', 'N/A')}
Role: {st.session_state.candidate.get('Role', 'N/A')}
Experience: {st.session_state.candidate.get('Experience', 'N/A')} years
Tech Stack: {st.session_state.candidate.get('Stack', 'N/A')}

ASSESSMENT RESULTS
-----------------
Average Score: {st.session_state.candidate.get('Average Score', 0)}/10
Duration: {round(duration, 1)} minutes
Questions Answered: {len(st.session_state.answers)}

DETAILED SCORES
--------------
"""
        for q_num, eval_info in st.session_state.evaluations.items():
            summary_text += f"{q_num}: {eval_info['score']}/10 - {eval_info['comment']}\n"
        
        summary_text += f"""
HR SUMMARY
----------
{st.session_state.summary}

================================
Generated by TalentScout AI
"""
        
        st.download_button(
            label="📝 Download Summary (TXT)",
            data=summary_text,
            file_name=f"{st.session_state.candidate.get('Name', 'candidate').replace(' ', '_')}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Human-readable summary report"
        )
    
    # Thank you message
    st.markdown("---")
    st.markdown("""
    ### 🙏 Thank You!
    
    Thank you for completing the technical screening with TalentScout AI. 
    Your responses have been recorded and will be reviewed by our hiring team.
    
    **Next Steps:**
    - Our team will review your assessment within 2-3 business days
    - You will receive an email with the results
    - Successful candidates will be invited for the next round
    
    Good luck! 🍀
    """)
    
    # Option to restart
    if st.button("🔄 Start New Screening", type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Handle ended state
elif st.session_state.step == "ended":
    st.info("👋 The screening session has ended. Thank you for your time!")
    if st.button("🔄 Start New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Powered by TalentScout AI | Built with Streamlit</p>", unsafe_allow_html=True)