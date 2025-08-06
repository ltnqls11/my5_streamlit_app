# main_simple.py - 간단한 버전으로 메뉴 기능 테스트
import streamlit as st
from utils import (
    pdf_to_text, create_vectorstore, create_qa_chain, get_pdf_list,
    authenticate_user, create_user, load_user_documents, get_document_summary,
    save_user_documents, check_plan_limits, update_user_activity,
    create_multi_vectorstore, create_cross_document_qa_chain, vector_manager,
    update_user_usage, load_chat_history, generate_direct_answer,
    save_chat_message, save_user_study_history, save_study_history,
    generate_simple_answer, summarize_text, generate_quiz,
    generate_short_answer_quiz, generate_flashcards, load_user_study_history,
    load_study_history, calculate_progress, calculate_user_progress,
    generate_learning_recommendations, create_personalized_learning_path,
    generate_adaptive_quiz, create_premium_study_package,
    create_instructor_chatbot, generate_shareable_quiz_link,
    create_academy_dashboard, analyze_chapters, generate_study_notes,
    generate_cornell_notes_advanced, generate_cornell_notes_html_advanced,
    text_to_speech, generate_premium_quiz, generate_share_link
)
import os
from dotenv import load_dotenv
import openai
import datetime
import json

load_dotenv()

# 세션 상태 초기화 (가장 먼저 실행)
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if 'multi_mode' not in st.session_state:
    st.session_state.multi_mode = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# 페이지 설정
st.set_page_config(
    page_title="PDF 학습 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 헤더
st.markdown("# 📚 나만의 AI 학습 튜터")
st.markdown("PDF 교재를 업로드하고 스마트한 학습을 시작하세요!")

# 로그인 시스템 (간단 버전)
if not st.session_state.logged_in:
    st.markdown("## 🔐 로그인")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 로그인")
        username = st.text_input("사용자명")
        password = st.text_input("비밀번호", type="password")
        
        if st.button("로그인"):
            if username and password:
                user_data = authenticate_user(username, password)
                if user_data:
                    st.session_state.user_profile = user_data
                    st.session_state.logged_in = True
                    st.success(f"환영합니다, {username}님!")
                    st.rerun()
                else:
                    st.error("로그인 실패")
            else:
                st.error("사용자명과 비밀번호를 입력해주세요")
    
    with col2:
        st.markdown("### 회원가입")
        new_username = st.text_input("새 사용자명")
        new_password = st.text_input("새 비밀번호", type="password")
        plan_type = st.selectbox("플랜 선택", ["free", "premium"])
        
        if st.button("회원가입"):
            if new_username and new_password:
                if create_user(new_username, new_password, plan_type):
                    st.success("회원가입 성공! 로그인해주세요.")
                else:
                    st.error("회원가입 실패")
            else:
                st.error("사용자명과 비밀번호를 입력해주세요")
    
    st.stop()

# 로그인 후 메인 화면
if st.session_state.user_profile:
    st.sidebar.markdown(f"### 👤 {st.session_state.user_profile['username']}님")
    st.sidebar.markdown(f"**플랜**: {st.session_state.user_profile['plan']}")

    if st.sidebar.button("🚪 로그아웃"):
        st.session_state.user_profile = None
        st.session_state.logged_in = False
        st.rerun()

# 사이드바 메뉴
with st.sidebar:
    st.markdown("### 🎯 학습 도구")
    
    menu_options = {
        "💬 질의응답": "AI와 대화하며 궁금한 점을 해결하세요",
        "📝 요약": "핵심 내용을 간단하게 정리해드려요", 
        "🧩 퀴즈": "객관식/단답형 문제로 실력을 점검하세요",
        "🎴 플래시카드": "핵심 개념을 카드로 만들어 암기 학습하세요",
        "📊 학습 이력": "학습 진행률과 기록을 확인하세요",
        "👤 사용자 대시보드": "개인 학습 통계와 사용량을 확인하세요"
    }
    
    menu = st.radio(
        "원하는 기능을 선택하세요:",
        list(menu_options.keys()),
        format_func=lambda x: x
    )
    
    st.info(menu_options[menu])

# PDF 선택
st.markdown("### 📁 PDF 파일 선택")
pdf_list = get_pdf_list("pdfs")

if not pdf_list:
    st.warning("📂 'pdfs' 폴더에 PDF 파일이 없습니다.")
    if st.button("📁 pdfs 폴더 생성"):
        os.makedirs("pdfs", exist_ok=True)
        st.success("✅ 'pdfs' 폴더가 생성되었습니다!")
        st.rerun()
else:
    selected_pdf = st.selectbox(
        "📚 학습할 PDF를 선택하세요:",
        ["선택하세요..."] + pdf_list
    )
    
    if selected_pdf and selected_pdf != "선택하세요...":
        st.session_state.selected_documents = [selected_pdf]
        st.success(f"✅ '{selected_pdf}' 파일이 선택되었습니다!")

# 🎯 메뉴별 기능 실행
st.markdown("---")

# 질의응답 기능
if menu == "💬 질의응답":
    st.markdown("## 💬 AI 질의응답")
    
    if not st.session_state.selected_documents:
        st.warning("📄 PDF 파일을 먼저 선택해주세요.")
    else:
        user_question = st.text_area("💭 질문을 입력하세요:")
        
        if st.button("🚀 질문하기") and user_question:
            with st.spinner("🤖 AI가 답변을 생성하고 있습니다..."):
                try:
                    pdf_path = os.path.join("pdfs", st.session_state.selected_documents[0])
                    text = pdf_to_text(pdf_path)
                    answer = generate_direct_answer(text, user_question)
                    st.markdown(f"**답변:** {answer}")
                except Exception as e:
                    st.error(f"오류: {str(e)}")

# 요약 기능
elif menu == "📝 요약":
    st.markdown("## 📝 스마트 요약")
    
    if not st.session_state.selected_documents:
        st.warning("📄 PDF 파일을 먼저 선택해주세요.")
    else:
        if st.button("📝 요약 생성하기"):
            with st.spinner("📝 AI가 요약을 생성하고 있습니다..."):
                try:
                    pdf_path = os.path.join("pdfs", st.session_state.selected_documents[0])
                    text = pdf_to_text(pdf_path)
                    summary = summarize_text(text)
                    st.markdown(f"**요약:**\n\n{summary}")
                except Exception as e:
                    st.error(f"오류: {str(e)}")

# 퀴즈 기능
elif menu == "🧩 퀴즈":
    st.markdown("## 🧩 스마트 퀴즈")
    
    if not st.session_state.selected_documents:
        st.warning("📄 PDF 파일을 먼저 선택해주세요.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            quiz_type = st.selectbox("퀴즈 유형", ["객관식", "단답형"])
        with col2:
            num_questions = st.slider("문제 수", 3, 10, 5)
        
        if st.button("🎯 퀴즈 생성하기"):
            with st.spinner("🧩 AI가 퀴즈를 생성하고 있습니다..."):
                try:
                    pdf_path = os.path.join("pdfs", st.session_state.selected_documents[0])
                    text = pdf_to_text(pdf_path)
                    
                    if quiz_type == "객관식":
                        quiz_content = generate_quiz(text, num_questions)
                    else:
                        quiz_content = generate_short_answer_quiz(text, num_questions)
                    
                    st.markdown(f"**퀴즈:**\n\n{quiz_content}")
                except Exception as e:
                    st.error(f"오류: {str(e)}")

# 플래시카드 기능
elif menu == "🎴 플래시카드":
    st.markdown("## 🎴 스마트 플래시카드")
    
    if not st.session_state.selected_documents:
        st.warning("📄 PDF 파일을 먼저 선택해주세요.")
    else:
        num_cards = st.slider("카드 수", 5, 20, 10)
        
        if st.button("🎴 플래시카드 생성하기"):
            with st.spinner("🎴 AI가 플래시카드를 생성하고 있습니다..."):
                try:
                    pdf_path = os.path.join("pdfs", st.session_state.selected_documents[0])
                    text = pdf_to_text(pdf_path)
                    flashcards = generate_flashcards(text, num_cards)
                    st.markdown(f"**플래시카드:**\n\n{flashcards}")
                except Exception as e:
                    st.error(f"오류: {str(e)}")

# 학습 이력 기능
elif menu == "📊 학습 이력":
    st.markdown("## 📊 학습 이력")
    
    username = st.session_state.user_profile['username']
    history = load_user_study_history(username)
    
    if history:
        st.markdown(f"### 총 {len(history)}개의 학습 기록")
        
        for i, record in enumerate(reversed(history[-10:])):  # 최근 10개
            with st.expander(f"기록 {len(history)-i}: {record['timestamp'][:16]}"):
                st.write(f"**질문:** {record['question']}")
                st.write(f"**답변:** {record['answer'][:200]}...")
                st.write(f"**주제:** {record.get('topic', '일반')}")
    else:
        st.info("아직 학습 기록이 없습니다. 질의응답을 시작해보세요!")

# 사용자 대시보드
elif menu == "👤 사용자 대시보드":
    st.markdown("## 👤 사용자 대시보드")
    
    username = st.session_state.user_profile['username']
    user_plan = st.session_state.user_profile['plan']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("사용자", username)
    
    with col2:
        st.metric("플랜", "🆓 무료" if user_plan == 'free' else "💎 프리미엄")
    
    with col3:
        join_date = st.session_state.user_profile.get('created_at', '')[:10]
        st.metric("가입일", join_date)
    
    # 사용량 통계
    st.markdown("### 📊 사용량 통계")
    
    try:
        history = load_user_study_history(username)
        chat_history = load_chat_history(username)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 질문 수", len(history))
        
        with col2:
            st.metric("챗 기록", len(chat_history))
        
        with col3:
            if user_plan == "free":
                st.metric("플랜 상태", "제한적")
            else:
                st.metric("플랜 상태", "무제한")
    
    except Exception as e:
        st.error(f"통계 로드 오류: {str(e)}")

st.markdown("---")
st.markdown("### 🚀 수익화 기능")
st.info("💎 프리미엄으로 업그레이드하면 더 많은 기능을 이용할 수 있습니다!")