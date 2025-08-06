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

# 플래시카드 관련 함수들
def parse_flashcards(content):
    """플래시카드 내용을 파싱하는 함수"""
    cards = []
    
    try:
        # 간단한 파싱 로직
        lines = content.split('\n')
        current_card = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('카드') and ':' in line:
                if current_card:
                    cards.append(current_card)
                current_card = {}
            elif line.startswith('앞면:') or line.startswith('질문:'):
                current_card['front'] = line.split(':', 1)[1].strip()
            elif line.startswith('뒷면:') or line.startswith('답변:'):
                current_card['back'] = line.split(':', 1)[1].strip()
        
        if current_card:
            cards.append(current_card)
        
        # 파싱이 실패한 경우 기본 카드 생성
        if not cards:
            # 내용을 문장 단위로 나누어 카드 생성
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
            for i, sentence in enumerate(sentences[:10]):  # 최대 10개
                cards.append({
                    'front': f"다음 내용에 대해 설명하세요: {sentence[:50]}...",
                    'back': sentence
                })
    
    except Exception as e:
        st.error(f"플래시카드 파싱 오류: {str(e)}")
    
    return cards

def next_card(result):
    """다음 카드로 이동하는 함수"""
    # 통계 업데이트
    if result == "correct":
        st.session_state.flashcard_stats['correct'] += 1
    else:
        st.session_state.flashcard_stats['incorrect'] += 1
    
    # 다음 카드로 이동
    if st.session_state.flashcard_current_card < st.session_state.flashcard_stats['total'] - 1:
        st.session_state.flashcard_current_card += 1
        st.session_state.flashcard_show_answer = False
        st.rerun()
    else:
        # 플래시카드 학습 완료 시 이력 저장
        save_flashcard_completion_history()
        show_final_stats()

def save_flashcard_completion_history():
    """플래시카드 학습 완료 이력 저장"""
    try:
        if st.session_state.user_profile:
            username = st.session_state.user_profile['username']
            stats = st.session_state.flashcard_stats
            total_answered = stats['correct'] + stats['incorrect']
            accuracy = stats['correct'] / total_answered * 100 if total_answered > 0 else 0
            
            # 학습 완료 이력 저장
            save_user_study_history(username, f"플래시카드 학습 완료 ({st.session_state.get('card_type', '정의형')})", f"총 {stats['total']}개 카드 중 {total_answered}개 학습 완료. 정답률: {accuracy:.1f}%", '플래시카드 완료')
            
            # 사용자 활동 업데이트
            update_user_activity(username, "flashcard_completed", {
                'total_cards': stats['total'],
                'accuracy': accuracy,
                'document': st.session_state.selected_documents[0] if st.session_state.selected_documents else 'Unknown'
            })
    except Exception as e:
        print(f"플래시카드 완료 이력 저장 오류: {e}")

def generate_cornell_notes(text, note_style="standard"):
    """코넬 노트 형식으로 내용을 정리하는 함수"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 내용을 코넬 노트 필기법에 따라 정리해주세요. 반드시 아래 형식을 정확히 따라주세요:

        내용: {text[:4000]}

        다음 형식으로 정리해주세요:

        === CUE COLUMN ===
        (핵심 키워드와 질문들을 한 줄씩 작성)
        - 키워드1
        - 키워드2  
        - 질문: 핵심 질문?
        - 키워드3

        === NOTE TAKING AREA ===
        (상세한 내용을 체계적으로 작성)
        • 주요 개념 설명
        • 구체적인 내용과 예시
        • 중요한 포인트들
        • 세부 사항들

        === SUMMARY ===
        (전체 내용을 2-3문장으로 요약)
        핵심 내용을 간단명료하게 요약한 문장들...

        스타일: {note_style}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"코넬 노트 생성 실패: {str(e)}"

def display_cornell_notes(notes_content):
    """코넬 노트를 시각적으로 표시하는 함수"""
    
    # 노트 내용 파싱
    sections = parse_cornell_notes(notes_content)
    
    # 실제 코넬 노트 양식으로 표시
    st.markdown("### 📋 Cornell Notes")
    st.markdown(f"**문서:** {st.session_state.selected_documents[0] if st.session_state.selected_documents else 'Study Notes'}")
    st.markdown(f"**날짜:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")
    
    # 메인 노트 영역 (2열 레이아웃)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🔑 Cue Column")
        st.markdown("*키워드 & 질문*")
        
        # 키워드 영역을 박스로 표시
        cue_box = f"""
        <div style="
            border: 1px solid #ddd; 
            padding: 15px; 
            background-color: #f8f9fa; 
            min-height: 400px;
            border-radius: 5px;
            font-size: 14px;
            line-height: 1.6;
        ">
        {sections['cues']}
        </div>
        """
        st.markdown(cue_box, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📝 Note-Taking Area")
        st.markdown("*상세 내용*")
        
        # 노트 영역을 박스로 표시
        notes_box = f"""
        <div style="
            border: 1px solid #ddd; 
            padding: 15px; 
            background-color: white; 
            min-height: 400px;
            border-radius: 5px;
            font-size: 14px;
            line-height: 1.8;
        ">
        {sections['notes']}
        </div>
        """
        st.markdown(notes_box, unsafe_allow_html=True)
    
    # 요약 영역 (하단 전체 폭)
    st.markdown("---")
    st.markdown("#### 📊 Summary")
    st.markdown("*핵심 요약*")
    
    summary_box = f"""
    <div style="
        border: 1px solid #ddd; 
        padding: 15px; 
        background-color: #e9ecef; 
        border-radius: 5px;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 20px;
    ">
    {sections['summary']}
    </div>
    """
    st.markdown(summary_box, unsafe_allow_html=True)

def parse_cornell_notes(content):
    """코넬 노트 내용을 파싱하여 섹션별로 분리"""
    sections = {
        'cues': '',
        'notes': '',
        'summary': ''
    }
    
    try:
        # 섹션 구분자로 내용 분리
        if "=== CUE COLUMN ===" in content:
            parts = content.split("=== CUE COLUMN ===")
            if len(parts) > 1:
                remaining = parts[1]
                
                if "=== NOTE TAKING AREA ===" in remaining:
                    cue_part, remaining = remaining.split("=== NOTE TAKING AREA ===", 1)
                    sections['cues'] = format_cue_section(cue_part.strip())
                    
                    if "=== SUMMARY ===" in remaining:
                        notes_part, summary_part = remaining.split("=== SUMMARY ===", 1)
                        sections['notes'] = format_notes_section(notes_part.strip())
                        sections['summary'] = format_summary_section(summary_part.strip())
                    else:
                        sections['notes'] = format_notes_section(remaining.strip())
        
        # 파싱 실패 시 전체 내용을 노트 영역에 표시
        if not sections['cues'] and not sections['notes'] and not sections['summary']:
            sections['notes'] = content.replace('\n', '<br>')
            sections['cues'] = '키워드 추출 필요'
            sections['summary'] = '요약 작성 필요'
            
    except Exception as e:
        sections = {
            'cues': '파싱 오류 발생',
            'notes': content.replace('\n', '<br>'),
            'summary': f'오류: {str(e)}'
        }
    
    return sections

def format_cue_section(text):
    """Cue 섹션 포맷팅"""
    if not text:
        return '키워드를 추출하지 못했습니다.'
    
    lines = text.split('\n')
    formatted = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('==='):
            if line.startswith('-') or line.startswith('•'):
                formatted.append(f"• {line[1:].strip()}")
            elif line:
                formatted.append(f"• {line}")
    
    return '<br>'.join(formatted) if formatted else '키워드를 추출하지 못했습니다.'

def format_notes_section(text):
    """Notes 섹션 포맷팅"""
    if not text:
        return '상세 내용을 생성하지 못했습니다.'
    
    lines = text.split('\n')
    formatted = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('==='):
            if line.startswith('•') or line.startswith('-'):
                formatted.append(f"• {line[1:].strip()}")
            elif line:
                formatted.append(line)
    
    return '<br><br>'.join(formatted) if formatted else '상세 내용을 생성하지 못했습니다.'

def format_summary_section(text):
    """Summary 섹션 포맷팅"""
    if not text:
        return '요약을 생성하지 못했습니다.'
    
    return text.strip().replace('\n', ' ')

def show_final_stats():
    """최종 통계를 표시하는 함수"""
    stats = st.session_state.flashcard_stats
    total_answered = stats['correct'] + stats['incorrect']
    
    if total_answered > 0:
        accuracy = stats['correct'] / total_answered * 100
        
        st.balloons()
        
        st.markdown(f"""
        ## 🎉 학습 완료!
        
        ### 📊 최종 결과
        - 총 카드 수: {stats['total']}개
        - 학습한 카드: {total_answered}개
        - 정답: {stats['correct']}개
        - 오답: {stats['incorrect']}개
        - **정답률: {accuracy:.1f}%**
        
        ### 🎯 학습 평가
        """)
        
        if accuracy >= 90:
            st.success("🏆 완벽합니다! 이 주제를 매우 잘 이해하고 있습니다.")
        elif accuracy >= 70:
            st.info("👍 잘했습니다! 조금 더 복습하면 완벽할 것 같습니다.")
        elif accuracy >= 50:
            st.warning("📚 더 공부가 필요합니다. 틀린 부분을 다시 확인해보세요.")
        else:
            st.error("💪 처음부터 다시 학습해보세요. 포기하지 마세요!")
        
        # 재시작 버튼
        if st.button("🔄 다시 시작", key="restart_flashcards"):
            st.session_state.flashcard_current_card = 0
            st.session_state.flashcard_show_answer = False
            st.session_state.flashcard_stats = {'correct': 0, 'incorrect': 0, 'total': stats['total']}
            st.rerun()

def display_interactive_flashcards(flashcards_content, card_type):
    """인터랙티브 플래시카드를 표시하는 함수"""
    
    # 플래시카드 파싱
    cards = parse_flashcards(flashcards_content)
    
    if not cards:
        st.error("플래시카드를 생성할 수 없습니다.")
        return
    
    # 세션 상태 초기화 - 키 이름을 더 구체적으로 변경
    if 'flashcard_current_card' not in st.session_state:
        st.session_state.flashcard_current_card = 0
    if 'flashcard_show_answer' not in st.session_state:
        st.session_state.flashcard_show_answer = False
    if 'flashcard_stats' not in st.session_state:
        st.session_state.flashcard_stats = {'correct': 0, 'incorrect': 0, 'total': len(cards)}
    
    # 진행률 표시
    progress = (st.session_state.flashcard_current_card + 1) / len(cards)
    st.progress(progress, text=f"카드 {st.session_state.flashcard_current_card + 1}/{len(cards)}")
    
    # 현재 카드 표시
    current_card = cards[st.session_state.flashcard_current_card]
    
    # 카드 스타일 CSS
    st.markdown("""
    <style>
    .flashcard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }
    
    .flashcard-front {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .flashcard-back {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .card-content {
        font-size: 1.2rem;
        line-height: 1.6;
    }
    
    .card-stats {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 카드 표시
    if not st.session_state.flashcard_show_answer:
        # 앞면 (질문)
        st.markdown(f"""
        <div class="flashcard flashcard-front">
            <div class="card-content">
                <h3>🤔 질문</h3>
                <p>{current_card['front']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 답 보기", key=f"show_answer_{st.session_state.flashcard_current_card}", use_container_width=True):
                st.session_state.flashcard_show_answer = True
                st.rerun()
    
    else:
        # 뒷면 (답변)
        st.markdown(f"""
        <div class="flashcard flashcard-back">
            <div class="card-content">
                <h3>💡 답변</h3>
                <p>{current_card['back']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 자가 평가 버튼
        st.markdown("### 📊 이 카드를 얼마나 잘 알고 있나요?")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("😰 모름", key=f"dont_know_{st.session_state.flashcard_current_card}", use_container_width=True):
                next_card("incorrect")
        
        with col2:
            if st.button("🤔 어려움", key=f"difficult_{st.session_state.flashcard_current_card}", use_container_width=True):
                next_card("incorrect")
        
        with col3:
            if st.button("😊 알겠음", key=f"know_{st.session_state.flashcard_current_card}", use_container_width=True):
                next_card("correct")
        
        with col4:
            if st.button("🎯 완벽!", key=f"perfect_{st.session_state.flashcard_current_card}", use_container_width=True):
                next_card("correct")
    
    # 네비게이션 버튼
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⏮️ 이전", key="prev_card", disabled=(st.session_state.flashcard_current_card == 0)):
            if st.session_state.flashcard_current_card > 0:
                st.session_state.flashcard_current_card -= 1
                st.session_state.flashcard_show_answer = False
                st.rerun()
    
    with col2:
        if st.button("🔄 다시", key="reset_card"):
            st.session_state.flashcard_show_answer = False
            st.rerun()
    
    with col3:
        if st.button("⏭️ 다음", key="next_card", disabled=(st.session_state.flashcard_current_card == len(cards) - 1)):
            if st.session_state.flashcard_current_card < len(cards) - 1:
                st.session_state.flashcard_current_card += 1
                st.session_state.flashcard_show_answer = False
                st.rerun()
    
    with col4:
        if st.button("🏁 완료", key="finish_cards"):
            show_final_stats()
    
    # 통계 표시
    stats = st.session_state.flashcard_stats
    if stats['correct'] + stats['incorrect'] > 0:
        st.markdown(f"""
        <div class="card-stats">
            <h4>📈 학습 통계</h4>
            <p>✅ 맞춤: {stats['correct']}개 | ❌ 틀림: {stats['incorrect']}개</p>
            <p>정답률: {stats['correct']/(stats['correct']+stats['incorrect'])*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

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
        "📋 코넬 노트": "체계적인 코넬 노트 필기법으로 학습 내용을 정리하세요",
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
                    
                    # 학습 이력에 저장
                    username = st.session_state.user_profile['username']
                    save_user_study_history(username, user_question, answer, '질의응답')
                    
                    # 사용자 활동 업데이트
                    update_user_activity(username, "question_asked", {
                        'question': user_question[:100],
                        'document': st.session_state.selected_documents[0]
                    })
                    
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
                    
                    # 학습 이력에 저장
                    username = st.session_state.user_profile['username']
                    save_user_study_history(username, f"'{st.session_state.selected_documents[0]}' 문서 요약 요청", summary, '요약')
                    
                    # 사용자 활동 업데이트
                    update_user_activity(username, "summary_generated", {
                        'document': st.session_state.selected_documents[0],
                        'summary_length': len(summary)
                    })
                    
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
                    
                    # 학습 이력에 저장
                    username = st.session_state.user_profile['username']
                    save_user_study_history(username, f"{quiz_type} 퀴즈 {num_questions}문제 생성 요청", quiz_content, '퀴즈')
                    
                    # 사용자 활동 업데이트
                    update_user_activity(username, "quiz_generated", {
                        'quiz_type': quiz_type,
                        'num_questions': num_questions,
                        'document': st.session_state.selected_documents[0]
                    })
                    
                except Exception as e:
                    st.error(f"오류: {str(e)}")

# 플래시카드 기능 (업그레이드 버전)
elif menu == "🎴 플래시카드":
    st.markdown("## 🎴 스마트 플래시카드")
    
    if not st.session_state.selected_documents:
        st.warning("📄 PDF 파일을 먼저 선택해주세요.")
    else:
        # 플래시카드가 이미 생성되었는지 확인
        if 'flashcards_generated' not in st.session_state:
            st.session_state.flashcards_generated = False
        
        if 'flashcards_content' not in st.session_state:
            st.session_state.flashcards_content = None
        
        # 플래시카드가 생성되지 않았거나 새로 생성하려는 경우
        if not st.session_state.flashcards_generated:
            # 플래시카드 설정
            col1, col2 = st.columns(2)
            
            with col1:
                num_cards = st.slider("카드 수", 5, 20, 10)
            
            with col2:
                card_type = st.selectbox("카드 유형", ["정의형", "문제형", "키워드형", "혼합형"])
            
            if st.button("🎴 플래시카드 생성하기"):
                with st.spinner("🎴 AI가 플래시카드를 생성하고 있습니다..."):
                    try:
                        pdf_path = os.path.join("pdfs", st.session_state.selected_documents[0])
                        text = pdf_to_text(pdf_path)
                        flashcards_raw = generate_flashcards(text, num_cards)
                        
                        # 플래시카드 내용을 세션에 저장
                        st.session_state.flashcards_content = flashcards_raw
                        st.session_state.flashcards_generated = True
                        st.session_state.card_type = card_type
                        
                        # 플래시카드 관련 세션 상태 초기화
                        st.session_state.flashcard_current_card = 0
                        st.session_state.flashcard_show_answer = False
                        
                        # 카드 파싱해서 총 개수 설정
                        cards = parse_flashcards(flashcards_raw)
                        st.session_state.flashcard_stats = {'correct': 0, 'incorrect': 0, 'total': len(cards)}
                        
                        # 학습 이력에 저장
                        username = st.session_state.user_profile['username']
                        save_user_study_history(username, f"{card_type} 플래시카드 {num_cards}개 생성 요청", f"플래시카드 {len(cards)}개가 생성되었습니다.", '플래시카드')
                        
                        # 사용자 활동 업데이트
                        update_user_activity(username, "flashcards_generated", {
                            'card_type': card_type,
                            'num_cards': len(cards),
                            'document': st.session_state.selected_documents[0]
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"오류: {str(e)}")
        
        # 플래시카드가 생성된 경우 표시
        if st.session_state.flashcards_generated and st.session_state.flashcards_content:
            # 새로 생성하기 버튼
            if st.button("🔄 새 플래시카드 생성"):
                st.session_state.flashcards_generated = False
                st.session_state.flashcards_content = None
                # 플래시카드 관련 세션 상태 초기화
                if 'flashcard_current_card' in st.session_state:
                    del st.session_state.flashcard_current_card
                if 'flashcard_show_answer' in st.session_state:
                    del st.session_state.flashcard_show_answer
                if 'flashcard_stats' in st.session_state:
                    del st.session_state.flashcard_stats
                st.rerun()
            
            st.markdown("---")
            
            # 플래시카드 표시
            display_interactive_flashcards(st.session_state.flashcards_content, st.session_state.get('card_type', '정의형'))

# 코넬 노트 기능
elif menu == "📋 코넬 노트":
    st.markdown("## 📋 코넬 노트 필기법")
    
    if not st.session_state.selected_documents:
        st.warning("📄 PDF 파일을 먼저 선택해주세요.")
    else:
        # 코넬 노트 설정
        col1, col2 = st.columns(2)
        
        with col1:
            note_style = st.selectbox("노트 스타일", ["standard", "detailed", "concise"], 
                                    format_func=lambda x: {
                                        "standard": "📝 표준 (균형잡힌 구성)",
                                        "detailed": "📚 상세 (자세한 설명)",
                                        "concise": "⚡ 간결 (핵심만 정리)"
                                    }[x])
        
        with col2:
            st.markdown("### 📖 코넬 노트란?")
            st.info("""
            **코넬 노트 필기법**은 효과적인 학습을 위한 체계적인 노트 정리 방법입니다.
            
            - **키워드 영역**: 핵심 개념과 질문
            - **노트 영역**: 상세한 내용과 설명  
            - **요약 영역**: 전체 내용의 핵심 정리
            """)
        
        if st.button("📋 코넬 노트 생성하기"):
            with st.spinner("📋 AI가 코넬 노트를 생성하고 있습니다..."):
                try:
                    pdf_path = os.path.join("pdfs", st.session_state.selected_documents[0])
                    text = pdf_to_text(pdf_path)
                    cornell_notes = generate_cornell_notes(text, note_style)
                    
                    # 코넬 노트 표시
                    display_cornell_notes(cornell_notes)
                    
                    # 다운로드 버튼 추가
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("💾 텍스트로 저장"):
                            st.download_button(
                                label="📄 텍스트 파일 다운로드",
                                data=cornell_notes,
                                file_name=f"cornell_notes_{st.session_state.selected_documents[0][:-4]}.txt",
                                mime="text/plain"
                            )
                    
                    with col2:
                        if st.button("🖨️ 인쇄용 버전"):
                            st.markdown("### 📄 인쇄용 코넬 노트")
                            st.text_area("인쇄용 텍스트", cornell_notes, height=400)
                    
                    with col3:
                        if st.button("📧 이메일로 전송"):
                            st.info("이메일 전송 기능은 프리미엄 플랜에서 이용 가능합니다.")
                    
                    # 학습 이력에 저장
                    username = st.session_state.user_profile['username']
                    save_user_study_history(username, f"코넬 노트 생성 ({note_style} 스타일)", f"'{st.session_state.selected_documents[0]}' 문서의 코넬 노트가 생성되었습니다.", '코넬 노트')
                    
                    # 사용자 활동 업데이트
                    update_user_activity(username, "cornell_notes_generated", {
                        'note_style': note_style,
                        'document': st.session_state.selected_documents[0]
                    })
                    
                except Exception as e:
                    st.error(f"오류: {str(e)}")

# 학습 이력 기능
elif menu == "📊 학습 이력":
    st.markdown("## 📊 학습 이력")
    
    username = st.session_state.user_profile['username']
    history = load_user_study_history(username)
    
    if history:
        st.markdown(f"### 총 {len(history)}개의 학습 기록")
        
        # 주제별 통계
        topic_stats = {}
        for record in history:
            topic = record.get('topic', '일반')
            topic_stats[topic] = topic_stats.get(topic, 0) + 1
        
        # 통계 표시
        st.markdown("#### 📊 주제별 학습 통계")
        cols = st.columns(len(topic_stats))
        for i, (topic, count) in enumerate(topic_stats.items()):
            with cols[i % len(cols)]:
                st.metric(topic, f"{count}회")
        
        st.markdown("---")
        
        # 최근 학습 기록 표시
        st.markdown("#### 📝 최근 학습 기록")
        for i, record in enumerate(reversed(history[-15:])):  # 최근 15개
            timestamp = record.get('timestamp', '')[:16]
            topic = record.get('topic', '일반')
            
            # 주제별 아이콘
            topic_icons = {
                '질의응답': '💬',
                '요약': '📝',
                '퀴즈': '🧩',
                '플래시카드': '🎴',
                '플래시카드 완료': '🏆',
                '코넬 노트': '📋'
            }
            icon = topic_icons.get(topic, '📚')
            
            with st.expander(f"{icon} {topic} - {timestamp}"):
                st.write(f"**질문/요청:** {record['question']}")
                
                # 답변 길이에 따라 표시 방식 조정
                answer = record.get('answer', '')
                if len(answer) > 300:
                    st.write(f"**답변:** {answer[:300]}...")
                    if st.button(f"전체 보기", key=f"show_full_{i}"):
                        st.write(f"**전체 답변:** {answer}")
                else:
                    st.write(f"**답변:** {answer}")
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