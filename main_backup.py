# main.py
import streamlit as st
from utils import pdf_to_text, create_vectorstore, create_qa_chain
import os
from dotenv import load_dotenv
import openai

load_dotenv()

# API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# API 키 테스트
def test_openai_api():
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return True, "API 키가 정상적으로 작동합니다."
    except Exception as e:
        return False, f"API 키 오류: {str(e)}"

# API 키 상태 표시
api_status, api_message = test_openai_api()
if api_status:
    st.success(api_message)
else:
    st.warning(f"{api_message}")
    st.info("HuggingFace 무료 임베딩을 사용하여 계속 진행합니다.")

# 페이지 설정
st.set_page_config(
    page_title="PDF 학습 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .cornell-section {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .cornell-cue {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .cornell-notes {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .cornell-summary {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>📚 AI 학습 도우미</h1>
    <p>PDF 교재를 업로드하고 스마트한 학습을 시작하세요!</p>
</div>
""", unsafe_allow_html=True)

# 사이드바 메뉴 (더 깔끔하게)
with st.sidebar:
    st.markdown("### 🎯 학습 도구")
    
    menu_options = {
        "💬 질의응답": "AI와 대화하며 궁금한 점을 해결하세요",
        "📝 요약": "핵심 내용을 간단하게 정리해드려요", 
        "🧩 퀴즈": "객관식/단답형 문제로 실력을 점검하세요",
        "🎴 플래시카드": "핵심 개념을 카드로 만들어 암기 학습하세요",
        "� 학습 이석력": "학습 진행률과 기록을 확인하세요",
        "� 챕터 분석": ""내용을 주제별로 체계적으로 분석해요",
        "� 학습 노트":: "다양한 스타일의 노트를 자동 생성해요",
        "📋 코넬 노트": "효과적인 코넬 노트 필기법을 적용해요",
        "🎵 음성 요약": "요약 내용을 음성으로 들어보세요"
    }
    
    menu = st.radio(
        "원하는 기능을 선택하세요:",
        list(menu_options.keys()),
        format_func=lambda x: x
    )
    
    # 선택된 메뉴 설명
    st.info(menu_options[menu])

# PDF 선택 방식
from utils import get_pdf_list

st.markdown("### 📁 PDF 파일 선택")

# PDF 목록 가져오기
pdf_list = get_pdf_list("pdfs")

if not pdf_list:
    st.warning("📂 'pdfs' 폴더에 PDF 파일이 없습니다.")
    st.info("💡 사용법: 프로젝트 폴더에 'pdfs' 폴더를 만들고 PDF 파일을 넣어주세요.")
    
    # 폴더 생성 버튼
    if st.button("📁 pdfs 폴더 생성"):
        import os
        os.makedirs("pdfs", exist_ok=True)
        st.success("✅ 'pdfs' 폴더가 생성되었습니다! PDF 파일을 넣고 새로고침하세요.")
        st.experimental_rerun()
else:
    # PDF 파일 선택
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_pdf = st.selectbox(
            "📚 학습할 PDF를 선택하세요:",
            ["선택하세요..."] + pdf_list,
            help="pdfs 폴더에 있는 PDF 파일 목록입니다"
        )
    
    with col2:
        if st.button("🔄 목록 새로고침"):
            st.experimental_rerun()

if selected_pdf and selected_pdf != "선택하세요...":
    try:
        pdf_path = os.path.join("pdfs", selected_pdf)
        
        # PDF 정보 표시
        file_size = os.path.getsize(pdf_path) / 1024  # KB
        st.markdown(f"""
        <div class="feature-card">
            <h4>📄 선택된 파일: {selected_pdf}</h4>
            <p>📊 파일 크기: {file_size:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("📖 PDF를 분석하고 있습니다..."):
            text = pdf_to_text(pdf_path)
            vectorstore = create_vectorstore(text)
            chain = create_qa_chain(vectorstore)

        st.success("✅ PDF 분석 완료! 원하는 학습 기능을 선택하세요.")
        
        # 질의응답 기능
        if menu == "💬 질의응답":
            st.markdown("""
            <div class="feature-card">
                <h2>💬 AI 질의응답</h2>
                <p>PDF 내용에 대해 궁금한 점을 자유롭게 물어보세요!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 질문 입력
            user_question = st.text_area(
                "💭 질문을 입력하세요:",
                placeholder="예: 이 챕터의 핵심 개념은 무엇인가요?",
                height=100
            )
            
            if st.button("🚀 질문하기", key="qa_submit") and user_question:
                with st.spinner("🤖 AI가 답변을 생성하고 있습니다..."):
                    answer = chain.run(user_question)
                
                # 답변 표시
                st.markdown("""
                <div class="feature-card">
                    <h4>🤖 AI 답변</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Q:** {user_question}")
                st.markdown(f"**A:** {answer}")
                
                # 학습 이력 저장
                from utils import save_study_history
                save_study_history(user_question, answer)
                
                st.success("✅ 답변이 학습 이력에 저장되었습니다!")
        
        # 요약 기능
        elif menu == "📝 요약":
            st.markdown("""
            <div class="feature-card">
                <h2>📝 스마트 요약</h2>
                <p>긴 PDF 내용을 핵심만 간추려 정리해드려요</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 요약 설정
            col1, col2 = st.columns([2, 1])
            with col1:
                summary_length = st.slider(
                    "📏 요약 길이 설정", 
                    200, 1000, 500,
                    help="더 긴 요약일수록 상세한 내용이 포함됩니다"
                )
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>설정된 길이</h4>
                    <h2>{summary_length}자</h2>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("✨ 요약 생성하기", key="summary_generate"):
                with st.spinner("🔄 AI가 핵심 내용을 분석하고 있습니다..."):
                    from utils import summarize_text
                    summary = summarize_text(text, summary_length)
                
                st.markdown("""
                <div class="feature-card">
                    <h3>📋 생성된 요약</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(summary)
                
                # 다운로드 버튼
                st.download_button(
                    label="💾 요약 파일 저장",
                    data=summary,
                    file_name=f"{selected_pdf}_summary.txt",
                    mime="text/plain",
                    help="요약 내용을 텍스트 파일로 저장합니다"
                )
        
        # 퀴즈 기능
        elif menu == "🧩 퀴즈":
            st.subheader("🧩 퀴즈 생성")
            
            # 퀴즈 타입 선택
            quiz_type = st.radio("퀴즈 유형", ["객관식", "단답형"])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                num_questions = st.slider("문제 개수", 3, 10, 5)
            with col2:
                if st.button("🧩 퀴즈 생성"):
                    with st.spinner("퀴즈를 생성하고 있습니다..."):
                        if quiz_type == "객관식":
                            from utils import generate_quiz
                            quiz = generate_quiz(text, num_questions)
                        else:
                            from utils import generate_short_answer_quiz
                            quiz = generate_short_answer_quiz(text, num_questions)
                    
                    st.subheader(f"📝 생성된 {quiz_type} 퀴즈")
                    st.write(quiz)
                    
                    # 퀴즈를 다운로드할 수 있게 제공
                    st.download_button(
                        label="📥 퀴즈 다운로드",
                        data=quiz,
                        file_name=f"{selected_pdf}_{quiz_type}_quiz.txt",
                        mime="text/plain"
                    )
        
        # 학습 이력 기능
        elif menu == "📊 학습 이력":
            st.markdown("""
            <div class="feature-card">
                <h2>📊 학습 대시보드</h2>
                <p>나의 학습 진행 상황과 기록을 한눈에 확인하세요</p>
            </div>
            """, unsafe_allow_html=True)
            
            from utils import load_study_history, calculate_progress
            history = load_study_history()
            
            if history:
                # 학습 통계 카드들
                col1, col2, col3 = st.columns(3)
                
                progress, topics = calculate_progress(history)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📈 학습 진행률</h4>
                        <h2>{progress:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>💬 총 질문 수</h4>
                        <h2>{len(history)}개</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📚 학습 주제</h4>
                        <h2>{len(topics)}개</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 진행률 바
                st.progress(progress / 100)
                
                # 주제별 학습 현황
                if topics:
                    st.markdown("### 📈 주제별 학습 현황")
                    for topic, count in topics.items():
                        col_topic, col_count = st.columns([3, 1])
                        with col_topic:
                            st.write(f"**{topic}**")
                        with col_count:
                            st.metric("", f"{count}개")
                
                # 최근 학습 기록
                st.markdown("### 📝 최근 학습 기록")
                
                for i, record in enumerate(reversed(history[-5:])):  # 최근 5개만 표시
                    with st.expander(f"📅 {record['timestamp'][:16]} - 질문 #{len(history)-i}"):
                        st.markdown(f"**❓ 질문:** {record['question']}")
                        st.markdown(f"**💡 답변:** {record['answer']}")
                
                # 다운로드 옵션
                st.markdown("### 💾 데이터 내보내기")
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    import json
                    history_json = json.dumps(history, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📊 JSON 형식",
                        data=history_json,
                        file_name="study_history.json",
                        mime="application/json"
                    )
                
                with col_download2:
                    # CSV 형식으로도 제공
                    import pandas as pd
                    df = pd.DataFrame(history)
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📈 CSV 형식",
                        data=csv,
                        file_name="study_history.csv",
                        mime="text/csv"
                    )
            else:
                st.markdown("""
                <div class="feature-card">
                    <h3>📝 학습을 시작해보세요!</h3>
                    <p>아직 학습 기록이 없습니다. 질의응답 기능을 사용하여 첫 번째 질문을 해보세요.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 🆕 챕터 분석 기능
        elif menu == "📚 챕터 분석":
            st.subheader("📚 챕터별 분석")
            
            if st.button("📚 챕터 분석 시작"):
                with st.spinner("챕터를 분석하고 있습니다..."):
                    from utils import analyze_chapters
                    analysis = analyze_chapters(text)
                
                st.subheader("📋 챕터별 분석 결과")
                st.markdown(analysis)
                
                st.download_button(
                    label="📥 챕터 분석 다운로드",
                    data=analysis,
                    file_name=f"{selected_pdf}_chapter_analysis.md",
                    mime="text/markdown"
                )
        
        # 🆕 학습 노트 기능
        elif menu == "📝 학습 노트":
            st.subheader("📝 자동 학습 노트 생성")
            
            note_style = st.selectbox(
                "노트 스타일",
                ["bullet", "outline", "mindmap"],
                format_func=lambda x: {"bullet": "불릿 포인트", "outline": "아웃라인", "mindmap": "마인드맵"}[x]
            )
            
            if st.button("📝 학습 노트 생성"):
                with st.spinner("학습 노트를 생성하고 있습니다..."):
                    from utils import generate_study_notes
                    notes = generate_study_notes(text, note_style)
                
                st.subheader("📋 생성된 학습 노트")
                st.markdown(notes)
                
                st.download_button(
                    label="📥 학습 노트 다운로드",
                    data=notes,
                    file_name=f"{selected_pdf}_study_notes_{note_style}.md",
                    mime="text/markdown"
                )
        
        # 🆕 플래시카드 기능
        elif menu == "🎴 플래시카드":
            st.markdown("""
            <div class="feature-card">
                <h2>🎴 학습 플래시카드</h2>
                <p>핵심 개념을 카드 형태로 만들어 효과적인 암기 학습을 도와드려요</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 플래시카드 설정
            col1, col2 = st.columns([2, 1])
            
            with col1:
                num_cards = st.slider(
                    "🎴 생성할 카드 수", 
                    5, 20, 10,
                    help="더 많은 카드일수록 다양한 개념을 학습할 수 있습니다"
                )
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>카드 수</h4>
                    <h2>{num_cards}장</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # 플래시카드 유형 안내
            with st.expander("🎯 플래시카드 유형 안내"):
                col_type1, col_type2 = st.columns(2)
                
                with col_type1:
                    st.markdown("""
                    #### 📚 생성되는 카드 유형:
                    - **정의 암기**: 개념 → 정의
                    - **공식 암기**: 공식명 → 공식
                    - **예시 문제**: 문제 → 해답
                    - **핵심 키워드**: 키워드 → 설명
                    """)
                
                with col_type2:
                    st.markdown("""
                    #### 💡 효과적인 사용법:
                    - 카드를 클릭해서 답 확인
                    - 키보드 화살표로 이동
                    - 스페이스바로 카드 뒤집기
                    - 섞기 기능으로 반복 학습
                    """)
            
            if st.button("🚀 플래시카드 생성하기", key="flashcard_generate"):
                with st.spinner("🎴 AI가 학습 카드를 만들고 있습니다..."):
                    from utils import generate_flashcards, generate_flashcards_html
                    flashcards_content = generate_flashcards(text, num_cards)
                
                st.markdown("### 🎴 생성된 플래시카드")
                
                # 플래시카드 미리보기
                cards_preview = flashcards_content.split('카드')[1:6]  # 처음 5개만 미리보기
                
                for i, card in enumerate(cards_preview, 1):
                    lines = card.strip().split('\n')
                    front = ""
                    back = ""
                    
                    for line in lines:
                        if line.startswith('앞면:'):
                            front = line.replace('앞면:', '').strip()
                        elif line.startswith('뒷면:'):
                            back = line.replace('뒷면:', '').strip()
                    
                    if front and back:
                        with st.expander(f"🎴 카드 {i} 미리보기"):
                            col_front, col_back = st.columns(2)
                            
                            with col_front:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                                           padding: 15px; border-radius: 10px; text-align: center;">
                                    <h4>🔍 앞면</h4>
                                    <p>{front}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_back:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                                           padding: 15px; border-radius: 10px; text-align: center;">
                                    <h4>💡 뒷면</h4>
                                    <p>{back}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                # 다운로드 옵션
                st.markdown("### 💾 플래시카드 다운로드")
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    st.download_button(
                        label="📄 텍스트 형식",
                        data=flashcards_content,
                        file_name=f"{selected_pdf}_flashcards.txt",
                        mime="text/plain",
                        help="플래시카드 내용을 텍스트로 저장"
                    )
                
                with col_download2:
                    # 인터랙티브 HTML 버전
                    html_content = generate_flashcards_html(flashcards_content, selected_pdf)
                    st.download_button(
                        label="🎮 인터랙티브 HTML",
                        data=html_content,
                        file_name=f"{selected_pdf}_flashcards.html",
                        mime="text/html",
                        help="브라우저에서 실행 가능한 인터랙티브 플래시카드"
                    )
                
                st.success("✅ 플래시카드가 생성되었습니다! HTML 파일을 다운로드하여 브라우저에서 학습하세요.")
        
        # 🆕 코넬 노트 기능
        elif menu == "📋 코넬 노트":
            st.markdown("""
            <div class="feature-card">
                <h2>📋 코넬 노트 필기법</h2>
                <p>효과적인 학습을 위한 체계적인 노트 정리 시스템</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 코넬 노트 구조 미리보기
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("""
                <div class="cornell-cue">
                    <h4>🔑 단서 영역</h4>
                    <p>• 핵심 키워드<br>• 중요 질문<br>• 기억할 점</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="cornell-notes">
                    <h4>📝 노트 영역</h4>
                    <p>• 주요 내용<br>• 상세 설명<br>• 예시와 개념</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="cornell-summary">
                    <h4>📌 요약 영역</h4>
                    <p>• 핵심 요약<br>• 전체 정리<br>• 복습 포인트</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 생성 버튼
            if st.button("🚀 코넬 노트 생성하기", key="cornell_generate"):
                with st.spinner("🔄 AI가 코넬 노트를 생성하고 있습니다..."):
                    from utils import generate_cornell_notes, generate_cornell_notes_html
                    cornell_content = generate_cornell_notes(text)
                
                # 생성된 노트를 파싱하여 각 영역별로 표시
                st.markdown("### 📋 생성된 코넬 노트")
                
                # 간단한 파싱으로 각 영역 분리
                sections = cornell_content.split('##')
                
                for section in sections:
                    if '노트 영역' in section or 'Note-taking Area' in section:
                        st.markdown(f"""
                        <div class="cornell-notes">
                            <h4>📝 노트 영역</h4>
                            {section.replace('노트 영역', '').replace('Note-taking Area', '')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif '단서 영역' in section or 'Cue Column' in section:
                        st.markdown(f"""
                        <div class="cornell-cue">
                            <h4>🔑 단서 영역</h4>
                            {section.replace('단서 영역', '').replace('Cue Column', '')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif '요약 영역' in section or 'Summary' in section:
                        st.markdown(f"""
                        <div class="cornell-summary">
                            <h4>📌 요약 영역</h4>
                            {section.replace('요약 영역', '').replace('Summary', '')}
                        </div>
                        """, unsafe_allow_html=True)
                
                # 다운로드 옵션들
                st.markdown("### 💾 다운로드 옵션")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.download_button(
                        label="📄 마크다운 파일",
                        data=cornell_content,
                        file_name=f"{selected_pdf}_cornell_notes.md",
                        mime="text/markdown",
                        help="마크다운 형식으로 다운로드"
                    )
                
                with col_b:
                    # HTML 버전 생성 (인쇄용)
                    html_content = generate_cornell_notes_html(
                        cornell_content.replace('\n', '\\n').replace('"', '\\"'), 
                        selected_pdf
                    )
                    st.download_button(
                        label="🖨️ 인쇄용 HTML",
                        data=html_content,
                        file_name=f"{selected_pdf}_cornell_notes.html",
                        mime="text/html",
                        help="인쇄하기 좋은 HTML 형식"
                    )
            
            # 사용법 안내 (접을 수 있는 형태)
            with st.expander("📚 코넬 노트 사용법 가이드"):
                col_guide1, col_guide2 = st.columns(2)
                
                with col_guide1:
                    st.markdown("""
                    #### 🎯 코넬 노트란?
                    코넬 대학교에서 개발된 **과학적으로 검증된** 노트 필기 시스템입니다.
                    
                    #### 📊 효과:
                    - 📈 기억력 **40% 향상**
                    - 🎯 집중력 증대
                    - 📝 체계적 정리
                    - 🔄 효율적 복습
                    """)
                
                with col_guide2:
                    st.markdown("""
                    #### 💡 활용 방법:
                    1. **📖 학습 단계**: 노트 영역 활용
                    2. **🔍 복습 단계**: 단서 영역으로 회상
                    3. **📝 정리 단계**: 요약 영역 완성
                    4. **🎯 시험 단계**: 요약으로 최종 점검
                    """)
        
        # 🆕 음성 요약 기능
        elif menu == "🎵 음성 요약":
            st.subheader("🎵 AI 음성 요약")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                summary_length = st.slider("요약 길이", 200, 800, 400)
            with col2:
                if st.button("🎵 음성 요약 생성"):
                    with st.spinner("요약을 생성하고 음성으로 변환하고 있습니다..."):
                        from utils import summarize_text, text_to_speech
                        
                        # 먼저 텍스트 요약
                        summary = summarize_text(text, summary_length)
                        st.subheader("📋 요약 내용")
                        st.write(summary)
                        
                        # 음성 변환
                        audio_data = text_to_speech(summary)
                        
                        if audio_data:
                            st.subheader("🎵 음성 재생")
                            st.audio(audio_data, format='audio/mp3')
                            
                            st.download_button(
                                label="📥 음성 파일 다운로드",
                                data=audio_data,
                                file_name=f"{selected_pdf}_summary.mp3",
                                mime="audio/mp3"
                            )
                        else:
                            st.error("음성 변환에 실패했습니다. gTTS 패키지를 설치해주세요.")
    
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.write("API 키를 확인하거나 잠시 후 다시 시도해주세요.")
