# utils.py
# PDF 업로드 → 텍스트 추출
from PyPDF2 import PdfReader
import os
import glob

def get_pdf_list(folder_path="pdfs"):
    """지정된 폴더에서 PDF 파일 목록을 가져옵니다."""
    try:
        # pdfs 폴더가 없으면 생성
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return []
        
        # 폴더 내 모든 파일 확인
        all_files = os.listdir(folder_path)
        
        # PDF 파일만 필터링 (대소문자 구분 없이)
        pdf_files = []
        for file in all_files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(file)
        
        # 중복 제거 및 정렬
        pdf_files = list(set(pdf_files))  # 중복 제거
        return sorted(pdf_files)
    except Exception as e:
        print(f"PDF 목록 가져오기 오류: {e}")
        return []

def pdf_to_text(file_path_or_uploaded):
    """파일 경로 또는 업로드된 파일에서 텍스트를 추출합니다."""
    try:
        if isinstance(file_path_or_uploaded, str):
            # 파일 경로인 경우
            if not os.path.exists(file_path_or_uploaded):
                return f"파일을 찾을 수 없습니다: {file_path_or_uploaded}"
            
            with open(file_path_or_uploaded, 'rb') as file:
                pdf = PdfReader(file)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
        else:
            # 업로드된 파일인 경우
            pdf = PdfReader(file_path_or_uploaded)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        
        if not text.strip():
            return "PDF에서 텍스트를 추출할 수 없습니다. 이미지 기반 PDF이거나 보호된 파일일 수 있습니다."
        
        return text
    except Exception as e:
        return f"PDF 읽기 오류: {str(e)}"

# 텍스트 → 문단 나누고 임베딩
try:
    from langchain.text_splitter import CharacterTextSplitter
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain import 오류: {e}")
    LANGCHAIN_AVAILABLE = False
    
import os

def create_vectorstore(text):
    if not LANGCHAIN_AVAILABLE:
        print("LangChain이 설치되지 않았습니다.")
        return None
    
    try:
        if not text or len(text.strip()) < 50:
            print("텍스트가 너무 짧습니다.")
            return None
            
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        if not chunks:
            print("텍스트 분할에 실패했습니다.")
            return None

        # HuggingFace 무료 임베딩 모델 사용 (OpenAI API 키 문제 해결)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_texts(chunks, embeddings)

        return vectorstore
    except Exception as e:
        print(f"벡터스토어 생성 중 오류: {str(e)}")
        return None

# 🚀 수익화 기능들

# 1. 사용자 맞춤 학습 이력 관리
import datetime
import json
import hashlib
import uuid
import openai

def create_personalized_learning_path(username, learning_history, preferences=None):
    """사용자 맞춤 학습 경로 생성"""
    try:
        client = openai.OpenAI()
        
        # 학습 이력 분석
        recent_topics = []
        weak_areas = []
        strong_areas = []
        
        for record in learning_history[-10:]:  # 최근 10개 기록
            question = record.get('question', '')
            answer = record.get('answer', '')
            
            # 간단한 키워드 분석
            if '어려워' in question or '모르겠' in question:
                weak_areas.append(question[:50])
            elif '잘 알겠' in answer or '이해했' in answer:
                strong_areas.append(question[:50])
        
        prompt = f"""
        사용자 {username}의 학습 이력을 분석하여 맞춤 학습 경로를 제안해주세요.
        
        최근 학습 주제: {recent_topics}
        약한 영역: {weak_areas}
        강한 영역: {strong_areas}
        
        다음 형식으로 제안해주세요:
        1. 현재 학습 수준 평가
        2. 추천 학습 순서 (3단계)
        3. 각 단계별 예상 소요 시간
        4. 맞춤 학습 전략
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"맞춤 학습 경로 생성 실패: {str(e)}"

def generate_adaptive_quiz(username, learning_history, difficulty_level="medium"):
    """사용자 수준에 맞는 적응형 퀴즈 생성"""
    try:
        client = openai.OpenAI()
        
        # 사용자 약점 분석
        weak_topics = analyze_weak_areas(learning_history)
        
        prompt = f"""
        사용자 {username}의 약점을 보완하는 적응형 퀴즈를 생성해주세요.
        
        약점 영역: {weak_topics}
        난이도: {difficulty_level}
        
        5개의 문제를 생성하되, 다음 조건을 만족해주세요:
        1. 약점 영역 집중 (70%)
        2. 복습 문제 (30%)
        3. 단계별 난이도 증가
        4. 상세한 해설 포함
        
        형식:
        Q1: [문제]
        1) 선택지1 2) 선택지2 3) 선택지3 4) 선택지4
        정답: [번호]
        해설: [상세 설명]
        학습 팁: [추가 학습 방향]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"적응형 퀴즈 생성 실패: {str(e)}"

def analyze_weak_areas(learning_history):
    """학습 이력에서 약점 영역 분석"""
    weak_areas = []
    for record in learning_history:
        question = record.get('question', '').lower()
        if any(word in question for word in ['어려워', '모르겠', '이해 안', '헷갈려']):
            # 키워드 추출
            keywords = question.split()[:3]  # 앞의 3단어
            weak_areas.extend(keywords)
    
    # 빈도수 기반 약점 영역 반환
    from collections import Counter
    common_weak = Counter(weak_areas).most_common(5)
    return [item[0] for item in common_weak]

# 2. 강사용 서비스 - PDF → 챗봇 → 공유 링크
def create_instructor_chatbot(instructor_name, pdf_content, course_name):
    """강사용 챗봇 생성"""
    try:
        # 고유 챗봇 ID 생성
        chatbot_id = str(uuid.uuid4())[:8]
        
        # 챗봇 데이터 저장
        os.makedirs("instructor_bots", exist_ok=True)
        bot_data = {
            "chatbot_id": chatbot_id,
            "instructor_name": instructor_name,
            "course_name": course_name,
            "pdf_content": pdf_content[:5000],  # 미리보기용
            "created_at": datetime.datetime.now().isoformat(),
            "access_count": 0,
            "student_interactions": [],
            "share_link": f"https://your-domain.com/chatbot/{chatbot_id}",
            "is_active": True
        }
        
        with open(f"instructor_bots/{chatbot_id}.json", 'w', encoding='utf-8') as f:
            json.dump(bot_data, f, ensure_ascii=False, indent=2)
        
        return chatbot_id, bot_data["share_link"]
    except Exception as e:
        return None, f"챗봇 생성 실패: {str(e)}"

def generate_shareable_quiz_link(quiz_content, instructor_name, course_name):
    """공유 가능한 퀴즈 링크 생성"""
    try:
        quiz_id = str(uuid.uuid4())[:8]
        
        os.makedirs("shared_quizzes", exist_ok=True)
        quiz_data = {
            "quiz_id": quiz_id,
            "instructor_name": instructor_name,
            "course_name": course_name,
            "quiz_content": quiz_content,
            "created_at": datetime.datetime.now().isoformat(),
            "access_count": 0,
            "student_results": [],
            "share_link": f"https://your-domain.com/quiz/{quiz_id}",
            "is_active": True,
            "time_limit": 30  # 분
        }
        
        with open(f"shared_quizzes/{quiz_id}.json", 'w', encoding='utf-8') as f:
            json.dump(quiz_data, f, ensure_ascii=False, indent=2)
        
        return quiz_id, quiz_data["share_link"]
    except Exception as e:
        return None, f"퀴즈 링크 생성 실패: {str(e)}"

# 3. 예상문제 생성 (유료 기능)
def generate_premium_exam_questions(pdf_content, exam_type="midterm", num_questions=20):
    """프리미엄 예상문제 생성"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 교재 내용을 바탕으로 {exam_type} 시험 예상문제 {num_questions}개를 생성해주세요.
        
        교재 내용:
        {pdf_content[:4000]}
        
        요구사항:
        1. 실제 시험과 유사한 난이도
        2. 다양한 문제 유형 (객관식, 단답형, 서술형)
        3. 출제 빈도가 높은 핵심 개념 위주
        4. 상세한 해설과 채점 기준
        5. 예상 출제 확률 표시
        
        형식:
        [문제 번호] (출제확률: ★★★☆☆)
        문제: [내용]
        정답: [답안]
        해설: [상세 설명]
        채점 기준: [부분점수 기준]
        관련 개념: [연관 학습 내용]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"예상문제 생성 실패: {str(e)}"

def create_premium_study_package(username, pdf_content, package_type="complete"):
    """프리미엄 학습 패키지 생성"""
    try:
        package_id = str(uuid.uuid4())[:8]
        
        # 패키지 구성 요소 생성
        components = {}
        
        if package_type in ["complete", "quiz"]:
            components["adaptive_quiz"] = generate_adaptive_quiz(username, [], "advanced")
            components["exam_questions"] = generate_premium_exam_questions(pdf_content)
        
        if package_type in ["complete", "summary"]:
            components["detailed_summary"] = generate_detailed_summary(pdf_content)
            components["concept_map"] = generate_concept_map(pdf_content)
        
        if package_type in ["complete", "practice"]:
            components["practice_problems"] = generate_practice_problems(pdf_content)
            components["solution_guide"] = generate_solution_guide(pdf_content)
        
        # 패키지 저장
        os.makedirs("premium_packages", exist_ok=True)
        package_data = {
            "package_id": package_id,
            "username": username,
            "package_type": package_type,
            "components": components,
            "created_at": datetime.datetime.now().isoformat(),
            "expires_at": (datetime.datetime.now() + datetime.timedelta(days=30)).isoformat(),
            "download_count": 0,
            "price": get_package_price(package_type)
        }
        
        with open(f"premium_packages/{package_id}.json", 'w', encoding='utf-8') as f:
            json.dump(package_data, f, ensure_ascii=False, indent=2)
        
        return package_id, package_data
    except Exception as e:
        return None, f"프리미엄 패키지 생성 실패: {str(e)}"

def get_package_price(package_type):
    """패키지 타입별 가격 반환"""
    prices = {
        "quiz": 5000,      # 5,000원
        "summary": 3000,   # 3,000원
        "practice": 7000,  # 7,000원
        "complete": 12000  # 12,000원 (할인가)
    }
    return prices.get(package_type, 5000)

# 4. 학원 제휴 기능
def create_academy_dashboard(academy_name, instructor_list):
    """학원용 대시보드 생성"""
    try:
        academy_id = str(uuid.uuid4())[:8]
        
        os.makedirs("academy_accounts", exist_ok=True)
        academy_data = {
            "academy_id": academy_id,
            "academy_name": academy_name,
            "instructors": instructor_list,
            "created_at": datetime.datetime.now().isoformat(),
            "subscription_plan": "basic",  # basic, premium, enterprise
            "monthly_usage": {
                "chatbots_created": 0,
                "students_served": 0,
                "quizzes_generated": 0,
                "api_calls": 0
            },
            "features": {
                "max_instructors": 5,
                "max_chatbots": 20,
                "max_students_per_bot": 100,
                "analytics": True,
                "white_label": False
            }
        }
        
        with open(f"academy_accounts/{academy_id}.json", 'w', encoding='utf-8') as f:
            json.dump(academy_data, f, ensure_ascii=False, indent=2)
        
        return academy_id, academy_data
    except Exception as e:
        return None, f"학원 계정 생성 실패: {str(e)}"

def generate_academy_analytics(academy_id):
    """학원용 분석 리포트 생성"""
    try:
        # 학원 데이터 로드
        with open(f"academy_accounts/{academy_id}.json", 'r', encoding='utf-8') as f:
            academy_data = json.load(f)
        
        # 분석 데이터 수집
        analytics = {
            "period": "last_30_days",
            "total_students": academy_data["monthly_usage"]["students_served"],
            "active_chatbots": academy_data["monthly_usage"]["chatbots_created"],
            "quiz_completion_rate": 85.2,  # 예시 데이터
            "student_satisfaction": 4.3,   # 5점 만점
            "most_popular_subjects": ["수학", "영어", "과학"],
            "peak_usage_hours": ["19:00-21:00", "14:00-16:00"],
            "revenue_generated": academy_data["monthly_usage"]["students_served"] * 1000
        }
        
        return analytics
    except Exception as e:
        return f"분석 리포트 생성 실패: {str(e)}"

# 5. 부가 기능들
def generate_detailed_summary(pdf_content):
    """상세 요약 생성"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 내용을 체계적으로 요약해주세요:
        
        {pdf_content[:3000]}
        
        다음 구조로 요약해주세요:
        1. 핵심 개념 (5개)
        2. 주요 내용 정리
        3. 중요 공식/법칙
        4. 실제 적용 사례
        5. 연관 학습 주제
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"상세 요약 생성 실패: {str(e)}"

def generate_concept_map(pdf_content):
    """개념 맵 생성"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 내용의 개념 맵을 텍스트 형태로 생성해주세요:
        
        {pdf_content[:2000]}
        
        형식:
        중심 개념: [메인 주제]
        ├── 하위 개념 1
        │   ├── 세부 내용 1-1
        │   └── 세부 내용 1-2
        ├── 하위 개념 2
        │   ├── 세부 내용 2-1
        │   └── 세부 내용 2-2
        └── 하위 개념 3
            ├── 세부 내용 3-1
            └── 세부 내용 3-2
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"개념 맵 생성 실패: {str(e)}"

def generate_practice_problems(pdf_content):
    """연습 문제 생성"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 내용을 바탕으로 연습 문제 10개를 생성해주세요:
        
        {pdf_content[:3000]}
        
        문제 유형:
        - 기초 문제 (3개)
        - 응용 문제 (4개)  
        - 심화 문제 (3개)
        
        각 문제마다 난이도와 예상 소요 시간을 표시해주세요.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"연습 문제 생성 실패: {str(e)}"

def generate_solution_guide(pdf_content):
    """해설 가이드 생성"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 내용에 대한 문제 해결 가이드를 작성해주세요:
        
        {pdf_content[:2000]}
        
        포함 내용:
        1. 문제 접근 방법
        2. 단계별 해결 과정
        3. 자주 하는 실수
        4. 확인 방법
        5. 유사 문제 해결 팁
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"해설 가이드 생성 실패: {str(e)}"

# 누락된 함수들 추가
def create_qa_chain(vectorstore):
    """QA 체인 생성"""
    if not LANGCHAIN_AVAILABLE:
        print("LangChain이 설치되지 않았습니다.")
        return None
    
    if vectorstore is None:
        print("벡터스토어가 없습니다.")
        return None
    
    try:
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API 키가 설정되지 않았습니다.")
            return None
            
        retriever = vectorstore.as_retriever()
        
        # 최신 모델 사용
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        from langchain.chains import RetrievalQA
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return chain
    except Exception as e:
        print(f"QA 체인 생성 중 오류: {str(e)}")
        return None

def authenticate_user(username, password):
    """사용자 인증"""
    try:
        users_file = "users/users.json"
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        if username in users:
            stored_password = users[username]["password"]
            if stored_password == hash_password(password):
                # 로그인 시간 업데이트
                users[username]["last_login"] = datetime.datetime.now().isoformat()
                
                with open(users_file, 'w', encoding='utf-8') as f:
                    json.dump(users, f, ensure_ascii=False, indent=2)
                
                return users[username]
        
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"인증 오류: {e}")
        return None

def hash_password(password):
    """비밀번호 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, plan="free"):
    """새 사용자 생성"""
    try:
        os.makedirs("users", exist_ok=True)
        users_file = "users/users.json"
        
        # 기존 사용자 목록 로드
        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                users = json.load(f)
        except FileNotFoundError:
            users = {}
        
        # 사용자명 중복 확인
        if username in users:
            return False
        
        # 새 사용자 추가
        users[username] = {
            "username": username,
            "password": hash_password(password),
            "plan": plan,
            "created_at": datetime.datetime.now().isoformat(),
            "last_login": None,
            "usage_stats": {
                "total_questions": 0,
                "total_pdfs": 0,
                "total_quizzes": 0,
                "api_calls_today": 0,
                "last_api_call": None
            }
        }
        
        # 저장
        with open(users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 생성 오류: {e}")
        return False

def load_user_documents(username):
    """사용자별 선택된 문서 목록 로드"""
    try:
        user_docs_file = f"users/{username}_documents.json"
        with open(user_docs_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        return doc_data.get("selected_documents", [])
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"사용자 문서 로드 오류: {e}")
        return []

def get_document_summary(pdf_name, text):
    """문서별 간단한 요약 생성"""
    try:
        # 텍스트가 너무 길면 앞부분만 사용
        preview_text = text[:1000] if len(text) > 1000 else text
        
        client = openai.OpenAI()
        prompt = f"""
        다음 문서의 핵심 내용을 2-3줄로 요약해주세요:
        
        문서명: {pdf_name}
        내용: {preview_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"요약 생성 실패: {str(e)}"

def save_user_documents(username, selected_pdfs):
    """사용자별 선택된 문서 목록 저장"""
    try:
        user_docs_file = f"users/{username}_documents.json"
        os.makedirs("users", exist_ok=True)
        
        doc_data = {
            "username": username,
            "selected_documents": selected_pdfs,
            "last_updated": datetime.datetime.now().isoformat(),
            "document_count": len(selected_pdfs)
        }
        
        with open(user_docs_file, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 문서 저장 오류: {e}")
        return False

def check_plan_limits(username, feature_type):
    """플랜별 기능 제한 확인"""
    try:
        users_file = "users/users.json"
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        if username not in users:
            return False, "사용자를 찾을 수 없습니다."
        
        user = users[username]
        plan = user.get("plan", "free")
        usage = user.get("usage_stats", {})
        
        # 무료 플랜 제한
        if plan == "free":
            if feature_type == "pdf_upload":
                if usage.get("total_pdfs", 0) >= 10:  # 일일 10개 제한
                    return False, "무료 플랜은 일일 10개 PDF 제한입니다."
            elif feature_type == "quiz_generation":
                if usage.get("total_quizzes", 0) >= 5:  # 일일 5개 제한
                    return False, "무료 플랜은 일일 5개 퀴즈 제한입니다."
            elif feature_type == "multi_document":
                return False, "다중 문서 기능은 프리미엄 플랜이 필요합니다."
            elif feature_type == "api_calls":
                today = datetime.datetime.now().date().isoformat()
                last_call = usage.get("last_api_call", "")
                if last_call.startswith(today):
                    if usage.get("api_calls_today", 0) >= 50:  # 일일 50회 제한
                        return False, "무료 플랜은 일일 50회 API 호출 제한입니다."
        
        return True, "사용 가능"
    except Exception as e:
        print(f"플랜 제한 확인 오류: {e}")
        return True, "확인 불가"

def update_user_activity(username, activity_type, data=None):
    """사용자 활동 업데이트"""
    try:
        os.makedirs("users", exist_ok=True)
        activity_file = f"users/{username}_activity.json"
        
        # 기존 활동 로드
        try:
            with open(activity_file, 'r', encoding='utf-8') as f:
                activity = json.load(f)
        except FileNotFoundError:
            activity = {
                "username": username,
                "total_questions": 0,
                "total_pdfs": 0,
                "total_quizzes": 0,
                "last_activity": None,
                "activities": []
            }
        
        # 활동 업데이트
        if activity_type == "question_asked":
            activity["total_questions"] += 1
        elif activity_type == "pdf_processed":
            activity["total_pdfs"] += 1
        elif activity_type == "quiz_completed":
            activity["total_quizzes"] += 1
        
        activity["last_activity"] = datetime.datetime.now().isoformat()
        
        # 활동 기록 추가
        activity["activities"].append({
            "type": activity_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data
        })
        
        # 최근 50개 활동만 유지
        if len(activity["activities"]) > 50:
            activity["activities"] = activity["activities"][-50:]
        
        # 저장
        with open(activity_file, 'w', encoding='utf-8') as f:
            json.dump(activity, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 활동 업데이트 오류: {e}")
        return False

# 🆕 다중 문서 지원 기능
def create_multi_vectorstore(texts_dict):
    """여러 PDF의 텍스트로 통합 벡터스토어 생성"""
    try:
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        
        all_chunks = []
        metadata_list = []
        
        for pdf_name, text in texts_dict.items():
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            
            # 각 청크에 출처 정보 추가
            for chunk in chunks:
                metadata_list.append({"source": pdf_name})
        
        # HuggingFace 임베딩 모델 사용
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 메타데이터와 함께 벡터스토어 생성
        vectorstore = FAISS.from_texts(all_chunks, embeddings, metadatas=metadata_list)
        
        return vectorstore
    except Exception as e:
        print(f"다중 벡터스토어 생성 오류: {e}")
        return None

def save_user_documents(username, selected_pdfs):
    """사용자별 선택된 문서 목록 저장"""
    try:
        user_docs_file = f"users/{username}_documents.json"
        os.makedirs("users", exist_ok=True)
        
        doc_data = {
            "username": username,
            "selected_documents": selected_pdfs,
            "last_updated": datetime.datetime.now().isoformat(),
            "document_count": len(selected_pdfs)
        }
        
        with open(user_docs_file, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 문서 저장 오류: {e}")
        return False

def load_user_documents(username):
    """사용자별 선택된 문서 목록 로드"""
    try:
        user_docs_file = f"users/{username}_documents.json"
        with open(user_docs_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        return doc_data.get("selected_documents", [])
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"사용자 문서 로드 오류: {e}")
        return []

def get_document_summary(pdf_name, text):
    """문서별 간단한 요약 생성"""
    try:
        # 텍스트가 너무 길면 앞부분만 사용
        preview_text = text[:1000] if len(text) > 1000 else text
        
        client = openai.OpenAI()
        prompt = f"""
        다음 문서의 핵심 내용을 2-3줄로 요약해주세요:
        
        문서명: {pdf_name}
        내용: {preview_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"요약 생성 실패: {str(e)}"

def create_cross_document_qa_chain(vectorstore):
    """다중 문서 질의응답 체인 생성"""
    try:
        if vectorstore is None:
            print("벡터스토어가 없습니다.")
            return None
            
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API 키가 설정되지 않았습니다.")
            return None
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 더 많은 문서에서 검색
        
        # 최신 모델 사용
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        # 커스텀 프롬프트로 출처 정보 포함
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        template = """
        다음 문서들의 내용을 바탕으로 질문에 답변해주세요. 
        답변 시 어떤 문서에서 정보를 가져왔는지 출처를 명시해주세요.
        
        관련 문서 내용:
        {context}
        
        질문: {question}
        
        답변 (출처 포함):
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
    except Exception as e:
        print(f"다중 문서 QA 체인 생성 오류: {e}")
        return None

# 질의응답 체인 구성 (RAG)
from langchain.chains import RetrievalQA
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
import openai
import json
import random

def create_qa_chain(vectorstore):
    if not LANGCHAIN_AVAILABLE:
        print("LangChain이 설치되지 않았습니다.")
        return None
    
    if vectorstore is None:
        print("벡터스토어가 없습니다.")
        return None
    
    try:
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API 키가 설정되지 않았습니다.")
            return None
            
        retriever = vectorstore.as_retriever()
        
        # 최신 모델 사용
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return chain
    except Exception as e:
        print(f"QA 체인 생성 중 오류: {str(e)}")
        return None

# 텍스트 요약 기능
def summarize_text(text, max_length=500):
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 {max_length}자 이내로 요약해주세요. 
        주요 개념과 핵심 내용을 포함하여 학습에 도움이 되도록 요약해주세요.
        
        텍스트:
        {text[:3000]}  # 너무 긴 텍스트는 잘라서 처리
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"요약 생성 중 오류가 발생했습니다: {str(e)}"

# 퀴즈 생성 기능
def generate_quiz(text, num_questions=5):
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 바탕으로 {num_questions}개의 객관식 퀴즈를 생성해주세요.
        각 문제는 4개의 선택지를 가지고, 정답은 1개입니다.
        
        형식:
        Q1: [질문]
        1) [선택지1]
        2) [선택지2] 
        3) [선택지3]
        4) [선택지4]
        정답: [번호]
        해설: [간단한 설명]
        
        텍스트:
        {text[:2000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}"

# 학습 이력 관리
import datetime
import json
import hashlib

def save_study_history(question, answer, filename="study_history.json"):
    try:
        # 기존 이력 로드
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # 새 기록 추가
        new_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer
        }
        
        history.append(new_record)
        
        # 최근 50개만 유지
        if len(history) > 50:
            history = history[-50:]
        
        # 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        print(f"이력 저장 오류: {e}")
        return False

def load_study_history(filename="study_history.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"이력 로드 오류: {e}")
        return []

# 🆕 단답형 퀴즈 생성 기능
def generate_short_answer_quiz(text, num_questions=5):
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 바탕으로 {num_questions}개의 단답형 퀴즈를 생성해주세요.
        각 문제는 간단한 단어나 구문으로 답할 수 있어야 합니다.
        
        형식:
        Q1: [질문]
        정답: [단답]
        해설: [간단한 설명]
        
        텍스트:
        {text[:2000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"단답형 퀴즈 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 챕터별 분석 및 요약
def analyze_chapters(text):
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 분석하여 챕터나 주제별로 나누고, 각 부분의 핵심 내용을 요약해주세요.
        
        형식:
        ## 챕터 1: [제목]
        - 핵심 내용: [요약]
        - 중요 키워드: [키워드들]
        - 학습 포인트: [학습해야 할 점]
        
        텍스트:
        {text[:4000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"챕터 분석 중 오류가 발생했습니다: {str(e)}"

# 🆕 학습 노트 자동 생성
def generate_study_notes(text, style="bullet"):
    try:
        client = openai.OpenAI()
        
        if style == "bullet":
            format_instruction = "불릿 포인트 형식으로 정리"
        elif style == "outline":
            format_instruction = "아웃라인 형식으로 정리"
        else:
            format_instruction = "마인드맵 형식으로 정리"
        
        prompt = f"""
        다음 텍스트를 학습 노트로 {format_instruction}해주세요.
        학생이 복습하기 쉽도록 핵심 개념, 정의, 예시를 포함해주세요.
        
        텍스트:
        {text[:3000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"학습 노트 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 학습 진행률 계산
def calculate_progress(history, total_chapters=10):
    if not history:
        return 0, {}
    
    # 질문 키워드 분석으로 학습 영역 파악
    topics = {}
    for record in history:
        question = record['question'].lower()
        # 간단한 키워드 매칭으로 주제 분류
        if any(word in question for word in ['1장', '첫번째', '처음', 'chapter 1']):
            topics['Chapter 1'] = topics.get('Chapter 1', 0) + 1
        elif any(word in question for word in ['2장', '두번째', 'chapter 2']):
            topics['Chapter 2'] = topics.get('Chapter 2', 0) + 1
        # ... 더 많은 챕터 매칭 로직
    
    progress_percentage = min(len(topics) / total_chapters * 100, 100)
    return progress_percentage, topics

# 🆕 TTS 기능 (gTTS 사용)
def text_to_speech(text, lang='ko'):
    try:
        from gtts import gTTS
        import io
        
        # 텍스트가 너무 길면 요약
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # 메모리에 저장
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
    except Exception as e:
        return None

# 🆕 코넬 노트 필기법 생성
def generate_cornell_notes(text):
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 코넬 노트 필기법 형식으로 정리해주세요.
        코넬 노트는 3개 영역으로 구성됩니다:
        
        1. 노트 영역 (Note-taking Area): 주요 내용과 세부사항
        2. 단서 영역 (Cue Column): 핵심 키워드, 질문, 중요 포인트
        3. 요약 영역 (Summary): 전체 내용의 핵심 요약
        
        다음 형식으로 작성해주세요:
        
        # 📝 코넬 노트
        
        ## 📋 노트 영역 (Note-taking Area)
        ### 주제 1: [제목]
        - [상세 내용]
        - [예시나 설명]
        - [중요한 개념]
        
        ### 주제 2: [제목]
        - [상세 내용]
        - [예시나 설명]
        
        ## 🔑 단서 영역 (Cue Column)
        - **핵심 키워드**: [키워드1, 키워드2, ...]
        - **중요 질문**: 
          - [질문1]
          - [질문2]
        - **기억할 점**: [중요 포인트]
        - **연관 개념**: [관련 개념들]
        
        ## 📌 요약 영역 (Summary)
        [전체 내용을 2-3문장으로 핵심 요약]
        
        텍스트:
        {text[:3500]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"코넬 노트 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 개선된 코넬 노트 생성 함수
def generate_cornell_notes_advanced(text, style="standard", include_questions=True):
    """
    코넬 노트 필기법에 따른 고급 노트 생성
    - style: standard, detailed, exam_focused
    - include_questions: 복습 질문 포함 여부
    """
    try:
        client = openai.OpenAI()
        
        # 스타일별 프롬프트 설정
        style_instructions = {
            "standard": "균형잡힌 구성으로 핵심 내용과 세부사항을 적절히 포함",
            "detailed": "상세한 설명과 예시를 중심으로 깊이 있는 내용 구성",
            "exam_focused": "시험에 나올 만한 핵심 개념과 중요 포인트 중심으로 구성"
        }
        
        questions_instruction = """
        - **복습 질문**: 
          - [내용을 이해했는지 확인하는 질문]
          - [응용 문제나 사고 질문]
          - [연관 개념과의 관계를 묻는 질문]
        """ if include_questions else ""
        
        prompt = f"""
        다음 텍스트를 코넬 노트 필기법(Cornell Note-Taking System)에 따라 정리해주세요.
        
        **작성 스타일**: {style_instructions[style]}
        
        **코넬 노트 구성 원칙**:
        1. 노트 영역(60%): 강의/읽기 중 기록하는 주요 내용
        2. 단서 영역(20%): 복습 시 사용할 키워드와 힌트
        3. 요약 영역(20%): 전체 내용의 핵심 요약
        
        **출력 형식**:
        
        # 📝 코넬 노트 - [주제명]
        
        ---
        
        ## 📋 노트 영역 (Note-taking Area)
        
        ### 1. [주요 주제 1]
        - **정의**: [핵심 개념 정의]
        - **특징**: [주요 특징들]
        - **예시**: [구체적 예시]
        - **중요사항**: [기억해야 할 점]
        
        ### 2. [주요 주제 2]
        - **개념**: [핵심 개념]
        - **원리**: [작동 원리나 과정]
        - **응용**: [실제 적용 사례]
        
        ### 3. [주요 주제 3]
        - **내용**: [상세 내용]
        - **관련성**: [다른 개념과의 연관성]
        
        ---
        
        ## 🔑 단서 영역 (Cue Column)
        
        **핵심 키워드**: [키워드1], [키워드2], [키워드3]
        
        **기억 단서**: 
        - [기억하기 쉬운 단서나 연상법]
        - [중요한 공식이나 법칙]
        
        **중요 포인트**: 
        - [시험에 나올 만한 내용]
        - [반드시 기억해야 할 사실]
        
        {questions_instruction}
        
        ---
        
        ## 📌 요약 영역 (Summary)
        
        [전체 내용을 2-3문장으로 핵심만 간결하게 요약. 나중에 빠른 복습용으로 사용할 수 있도록 가장 중요한 내용만 포함]
        
        ---
        
        텍스트:
        {text[:4000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.2
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"코넬 노트 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 코넬 노트 HTML 템플릿 생성 (인쇄용)
def generate_cornell_notes_html(cornell_content, title="학습 노트"):
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} - 코넬 노트</title>
        <style>
            body {{
                font-family: 'Malgun Gothic', Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                color: #333;
            }}
            .cornell-container {{
                max-width: 800px;
                margin: 0 auto;
                border: 2px solid #333;
                min-height: 600px;
            }}
            .header {{
                border-bottom: 2px solid #333;
                padding: 10px;
                background-color: #f8f9fa;
                text-align: center;
            }}
            .main-content {{
                display: flex;
                min-height: 500px;
            }}
            .cue-column {{
                width: 30%;
                border-right: 2px solid #333;
                padding: 15px;
                background-color: #fff8dc;
            }}
            .note-area {{
                width: 70%;
                padding: 15px;
                background-color: white;
            }}
            .summary-area {{
                border-top: 2px solid #333;
                padding: 15px;
                background-color: #f0f8ff;
                min-height: 80px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
                margin-top: 0;
            }}
            .cue-column h3 {{
                color: #e74c3c;
                font-size: 14px;
                margin-bottom: 8px;
            }}
            .note-area h3 {{
                color: #3498db;
                border-bottom: 1px solid #3498db;
                padding-bottom: 5px;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            .keyword {{
                background-color: #fff3cd;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .question {{
                color: #dc3545;
                font-style: italic;
            }}
            @media print {{
                body {{ margin: 0; }}
                .cornell-container {{ border: 1px solid #000; }}
            }}
        </style>
    </head>
    <body>
        <div class="cornell-container">
            <div class="header">
                <h1>📝 {title}</h1>
                <p>날짜: ___________  과목: ___________</p>
            </div>
            
            <div class="main-content">
                <div class="cue-column">
                    <h2>🔑 단서</h2>
                    <div id="cue-content">
                        <!-- 단서 내용이 여기에 들어갑니다 -->
                    </div>
                </div>
                
                <div class="note-area">
                    <h2>📋 노트</h2>
                    <div id="note-content">
                        <!-- 노트 내용이 여기에 들어갑니다 -->
                    </div>
                </div>
            </div>
            
            <div class="summary-area">
                <h2>📌 요약</h2>
                <div id="summary-content">
                    <!-- 요약 내용이 여기에 들어갑니다 -->
                </div>
            </div>
        </div>
        
        <script>
            // 마크다운 내용을 파싱하여 적절한 영역에 배치
            const content = `{cornell_content}`;
            
            // 간단한 마크다운 파싱
            function parseContent(content) {{
                const lines = content.split('\\n');
                let currentSection = '';
                let noteContent = '';
                let cueContent = '';
                let summaryContent = '';
                
                for (let line of lines) {{
                    if (line.includes('노트 영역') || line.includes('Note-taking Area')) {{
                        currentSection = 'note';
                    }} else if (line.includes('단서 영역') || line.includes('Cue Column')) {{
                        currentSection = 'cue';
                    }} else if (line.includes('요약 영역') || line.includes('Summary')) {{
                        currentSection = 'summary';
                    }} else if (line.trim() && !line.startsWith('#')) {{
                        if (currentSection === 'note') {{
                            noteContent += line + '<br>';
                        }} else if (currentSection === 'cue') {{
                            cueContent += line + '<br>';
                        }} else if (currentSection === 'summary') {{
                            summaryContent += line + '<br>';
                        }}
                    }}
                }}
                
                document.getElementById('note-content').innerHTML = noteContent;
                document.getElementById('cue-content').innerHTML = cueContent;
                document.getElementById('summary-content').innerHTML = summaryContent;
            }}
            
            parseContent(content);
        </script>
    </body>
    </html>
    """
    
    return html_template

# 🆕 개선된 코넬 노트 HTML 생성
def generate_cornell_notes_html_advanced(cornell_content, title="코넬 노트"):
    """실제 코넬 노트 형식에 맞는 HTML 생성"""
    
    # 내용 파싱
    sections = {'notes': '', 'cue': '', 'summary': ''}
    lines = cornell_content.split('\n')
    current_section = None
    
    for line in lines:
        if '노트 영역' in line or 'Note-taking Area' in line:
            current_section = 'notes'
        elif '단서 영역' in line or 'Cue Column' in line:
            current_section = 'cue'
        elif '요약 영역' in line or 'Summary' in line:
            current_section = 'summary'
        elif current_section and line.strip() and not line.startswith('#') and not line.startswith('---'):
            sections[current_section] += line + '\n'
    
    # HTML 템플릿
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            @page {{
                size: A4;
                margin: 1cm;
            }}
            
            body {{
                font-family: 'Malgun Gothic', Arial, sans-serif;
                margin: 0;
                padding: 0;
                line-height: 1.6;
                color: #333;
            }}
            
            .cornell-container {{
                width: 100%;
                height: 100vh;
                border: 3px solid #000;
                display: flex;
                flex-direction: column;
            }}
            
            .cornell-header {{
                background-color: #f8f9fa;
                border-bottom: 2px solid #000;
                padding: 15px;
                text-align: center;
            }}
            
            .cornell-main {{
                display: flex;
                flex: 1;
            }}
            
            .cornell-cue {{
                width: 25%;
                border-right: 2px solid #000;
                padding: 15px;
                background-color: #fff8dc;
                font-size: 0.9rem;
            }}
            
            .cornell-notes {{
                width: 75%;
                padding: 15px;
                background-color: white;
            }}
            
            .cornell-summary {{
                border-top: 2px solid #000;
                padding: 15px;
                background-color: #e7f3ff;
                min-height: 100px;
            }}
        </style>
    </head>
    <body>
        <div class="cornell-container">
            <div class="cornell-header">
                <h1>📝 {title}</h1>
                <div>생성일: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}</div>
            </div>
            
            <div class="cornell-main">
                <div class="cornell-cue">
                    <h3>🔑 단서 영역</h3>
                    <div>{sections['cue']}</div>
                </div>
                
                <div class="cornell-notes">
                    <h3>📋 노트 영역</h3>
                    <div>{sections['notes']}</div>
                </div>
            </div>
            
            <div class="cornell-summary">
                <h3>📌 요약 영역</h3>
                <div>{sections['summary']}</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template

# 🆕 플래시카드 생성 기능
def generate_flashcards(text, num_cards=10):
    try:
        # 텍스트 유효성 검사
        if not text or len(text.strip()) < 100:
            return "플래시카드를 생성하기에 텍스트가 너무 짧습니다. 더 많은 내용이 필요합니다."
        
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OpenAI API 키가 설정되지 않았습니다."
        
        client = openai.OpenAI(api_key=api_key)
        
        # 텍스트 길이 제한 (안전하게)
        safe_text = text[:3000] if len(text) > 3000 else text
        
        prompt = f"""
        다음 텍스트를 바탕으로 {num_cards}개의 학습 플래시카드를 생성해주세요.
        각 카드는 앞면(질문/개념)과 뒷면(답변/설명)으로 구성됩니다.
        
        형식:
        카드 1:
        앞면: [핵심 개념이나 질문]
        뒷면: [상세한 설명이나 답변]
        
        카드 2:
        앞면: [핵심 개념이나 질문]
        뒷면: [상세한 설명이나 답변]
        
        플래시카드는 다음과 같은 유형으로 만들어주세요:
        - 정의 암기 (개념 → 정의)
        - 공식 암기 (공식명 → 공식)
        - 예시 문제 (문제 → 해답)
        - 핵심 키워드 (키워드 → 설명)
        
        텍스트:
        {safe_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"플래시카드 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 직접 답변 생성 (체인 실패 시 대안)
def generate_direct_answer(text, question):
    """벡터스토어 없이 직접 텍스트 기반 답변 생성"""
    try:
        client = openai.OpenAI()
        
        # 텍스트가 너무 길면 관련 부분만 추출
        relevant_text = text[:4000] if len(text) > 4000 else text
        
        prompt = f"""
        다음 텍스트를 바탕으로 질문에 답변해주세요.
        
        텍스트:
        {relevant_text}
        
        질문: {question}
        
        답변:
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 간단한 답변 생성 (최종 대안)
def generate_simple_answer(text, question):
    """가장 기본적인 텍스트 매칭 기반 답변"""
    try:
        # 질문에서 키워드 추출
        question_lower = question.lower()
        text_lower = text.lower()
        
        # 간단한 키워드 매칭
        keywords = [word for word in question_lower.split() if len(word) > 2]
        
        # 관련 문장 찾기
        sentences = text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= 3:  # 최대 3개 문장
                    break
        
        if relevant_sentences:
            return " ".join(relevant_sentences)
        else:
            return "죄송합니다. 해당 질문에 대한 관련 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요."
    
    except Exception as e:
        return f"기본 답변 생성 실패: {str(e)}"

# 🆕 사용자별 학습 이력 관리
def save_user_study_history(username, question, answer, topic="일반"):
    """사용자별 학습 이력 저장"""
    try:
        os.makedirs("users", exist_ok=True)
        history_file = f"users/{username}_history.json"
        
        # 기존 이력 로드
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # 새 기록 추가
        new_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "topic": topic
        }
        
        history.append(new_record)
        
        # 최근 100개만 유지
        if len(history) > 100:
            history = history[-100:]
        
        # 저장
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 이력 저장 오류: {e}")
        return False

def load_user_study_history(username):
    """사용자별 학습 이력 로드"""
    try:
        history_file = f"users/{username}_history.json"
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"사용자 이력 로드 오류: {e}")
        return []

def update_user_activity(username, activity_type, data=None):
    """사용자 활동 업데이트"""
    try:
        os.makedirs("users", exist_ok=True)
        activity_file = f"users/{username}_activity.json"
        
        # 기존 활동 로드
        try:
            with open(activity_file, 'r', encoding='utf-8') as f:
                activity = json.load(f)
        except FileNotFoundError:
            activity = {
                "username": username,
                "total_questions": 0,
                "total_pdfs": 0,
                "total_quizzes": 0,
                "last_activity": None,
                "activities": []
            }
        
        # 활동 업데이트
        if activity_type == "question_asked":
            activity["total_questions"] += 1
        elif activity_type == "pdf_processed":
            activity["total_pdfs"] += 1
        elif activity_type == "quiz_completed":
            activity["total_quizzes"] += 1
        
        activity["last_activity"] = datetime.datetime.now().isoformat()
        
        # 활동 기록 추가
        activity["activities"].append({
            "type": activity_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data
        })
        
        # 최근 50개 활동만 유지
        if len(activity["activities"]) > 50:
            activity["activities"] = activity["activities"][-50:]
        
        # 저장
        with open(activity_file, 'w', encoding='utf-8') as f:
            json.dump(activity, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 활동 업데이트 오류: {e}")
        return False

def calculate_user_progress(history, username):
    """사용자별 학습 진행률 계산"""
    if not history:
        return 0, {}, {}
    
    # 주제별 질문 수 계산
    topics = {}
    for record in history:
        topic = record.get('topic', '일반')
        topics[topic] = topics.get(topic, 0) + 1
    
    # 진행률 계산 (질문 수 기반)
    total_questions = len(history)
    progress_percentage = min(total_questions * 2, 100)  # 50개 질문 = 100%
    
    # 학습 패턴 분석
    study_patterns = {
        "most_active_topic": max(topics.items(), key=lambda x: x[1])[0] if topics else "없음",
        "total_topics": len(topics),
        "avg_questions_per_topic": total_questions / len(topics) if topics else 0
    }
    
    return progress_percentage, topics, study_patterns

def check_plan_limits(username, feature_type):
    """플랜별 기능 제한 확인"""
    # 기본적으로 모든 기능 허용 (실제 구현에서는 사용자 플랜 확인)
    return True, "사용 가능"

def generate_learning_recommendations(username, history):
    """학습 추천 생성"""
    try:
        if not history:
            return "아직 학습 이력이 없습니다. 질문을 시작해보세요!"
        
        # 최근 질문 분석
        recent_questions = [record['question'] for record in history[-5:]]
        
        client = openai.OpenAI()
        prompt = f"""
        다음 최근 질문들을 분석하여 학습자에게 맞춤 추천을 해주세요:
        
        최근 질문들:
        {chr(10).join(recent_questions)}
        
        다음 형식으로 추천해주세요:
        1. 학습 패턴 분석
        2. 추천 학습 주제
        3. 다음 단계 제안
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"추천 생성 실패: {str(e)}"

# 🆕 사용자 관리 시스템
def hash_password(password):
    """비밀번호 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, plan="free"):
    """새 사용자 생성"""
    try:
        os.makedirs("users", exist_ok=True)
        users_file = "users/users.json"
        
        # 기존 사용자 목록 로드
        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                users = json.load(f)
        except FileNotFoundError:
            users = {}
        
        # 사용자명 중복 확인
        if username in users:
            return False
        
        # 새 사용자 추가
        users[username] = {
            "username": username,
            "password": hash_password(password),
            "plan": plan,
            "created_at": datetime.datetime.now().isoformat(),
            "last_login": None,
            "usage_stats": {
                "total_questions": 0,
                "total_pdfs": 0,
                "total_quizzes": 0,
                "api_calls_today": 0,
                "last_api_call": None
            }
        }
        
        # 저장
        with open(users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 생성 오류: {e}")
        return False

def authenticate_user(username, password):
    """사용자 인증"""
    try:
        users_file = "users/users.json"
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        if username in users:
            stored_password = users[username]["password"]
            if stored_password == hash_password(password):
                # 로그인 시간 업데이트
                users[username]["last_login"] = datetime.datetime.now().isoformat()
                
                with open(users_file, 'w', encoding='utf-8') as f:
                    json.dump(users, f, ensure_ascii=False, indent=2)
                
                return users[username]
        
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"인증 오류: {e}")
        return None

def check_plan_limits(username, feature_type):
    """플랜별 기능 제한 확인"""
    try:
        users_file = "users/users.json"
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        if username not in users:
            return False, "사용자를 찾을 수 없습니다."
        
        user = users[username]
        plan = user.get("plan", "free")
        usage = user.get("usage_stats", {})
        
        # 무료 플랜 제한
        if plan == "free":
            if feature_type == "pdf_upload":
                if usage.get("total_pdfs", 0) >= 10:  # 일일 10개 제한
                    return False, "무료 플랜은 일일 10개 PDF 제한입니다."
            elif feature_type == "quiz_generation":
                if usage.get("total_quizzes", 0) >= 5:  # 일일 5개 제한
                    return False, "무료 플랜은 일일 5개 퀴즈 제한입니다."
            elif feature_type == "multi_document":
                return False, "다중 문서 기능은 프리미엄 플랜이 필요합니다."
            elif feature_type == "api_calls":
                today = datetime.datetime.now().date().isoformat()
                last_call = usage.get("last_api_call", "")
                if last_call.startswith(today):
                    if usage.get("api_calls_today", 0) >= 50:  # 일일 50회 제한
                        return False, "무료 플랜은 일일 50회 API 호출 제한입니다."
        
        return True, "사용 가능"
    except Exception as e:
        print(f"플랜 제한 확인 오류: {e}")
        return True, "확인 불가"

def update_user_usage(username, feature_type):
    """사용자 사용량 업데이트"""
    try:
        users_file = "users/users.json"
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        if username in users:
            usage = users[username].get("usage_stats", {})
            today = datetime.datetime.now().date().isoformat()
            
            # API 호출 카운트 업데이트
            last_call = usage.get("last_api_call", "")
            if last_call.startswith(today):
                usage["api_calls_today"] = usage.get("api_calls_today", 0) + 1
            else:
                usage["api_calls_today"] = 1
            
            usage["last_api_call"] = datetime.datetime.now().isoformat()
            
            # 기능별 카운트 업데이트
            if feature_type == "pdf_upload":
                usage["total_pdfs"] = usage.get("total_pdfs", 0) + 1
            elif feature_type == "quiz_generation":
                usage["total_quizzes"] = usage.get("total_quizzes", 0) + 1
            elif feature_type == "question_asked":
                usage["total_questions"] = usage.get("total_questions", 0) + 1
            
            users[username]["usage_stats"] = usage
            
            with open(users_file, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용량 업데이트 오류: {e}")
        return False

# 🆕 챗 기록 관리
def save_chat_message(username, message, response, message_type="qa"):
    """챗 메시지 저장"""
    try:
        os.makedirs("users", exist_ok=True)
        chat_file = f"users/{username}_chat.json"
        
        # 기존 챗 기록 로드
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
        except FileNotFoundError:
            chat_history = []
        
        # 새 메시지 추가
        new_message = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "type": message_type
        }
        
        chat_history.append(new_message)
        
        # 최근 100개만 유지
        if len(chat_history) > 100:
            chat_history = chat_history[-100:]
        
        # 저장
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"챗 기록 저장 오류: {e}")
        return False

def load_chat_history(username):
    """챗 기록 로드"""
    try:
        chat_file = f"users/{username}_chat.json"
        with open(chat_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"챗 기록 로드 오류: {e}")
        return []

# 🆕 다중 벡터스토어 관리
class MultiVectorStoreManager:
    def __init__(self):
        self.vectorstores = {}
        self.document_mapping = {}
    
    def add_document(self, doc_name, text):
        """개별 문서의 벡터스토어 생성"""
        try:
            vectorstore = create_vectorstore(text)
            if vectorstore:
                self.vectorstores[doc_name] = vectorstore
                self.document_mapping[doc_name] = len(text)
                return True
            return False
        except Exception as e:
            print(f"문서 추가 오류: {e}")
            return False
    
    def search_across_documents(self, query, k=5):
        """모든 문서에서 검색"""
        results = []
        for doc_name, vectorstore in self.vectorstores.items():
            try:
                docs = vectorstore.similarity_search(query, k=k//len(self.vectorstores) + 1)
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "source": doc_name,
                        "score": 0  # 실제 구현에서는 유사도 점수 계산
                    })
            except Exception as e:
                print(f"검색 오류 ({doc_name}): {e}")
        
        return results[:k]
    
    def get_document_stats(self):
        """문서 통계 반환"""
        return {
            "total_documents": len(self.vectorstores),
            "document_sizes": self.document_mapping,
            "total_size": sum(self.document_mapping.values())
        }

# 전역 벡터스토어 매니저
vector_manager = MultiVectorStoreManager()

# 🆕 플래시카드 HTML 생성 (인터랙티브)
def generate_flashcards_html(flashcards_content, title="학습 플래시카드"):
    # 플래시카드 내용 파싱
    cards = []
    lines = flashcards_content.split('\n')
    current_card = {}
    
    try:
        for line in lines:
            line = line.strip()
            if line.startswith('카드') and ':' in line:
                if current_card and 'front' in current_card and 'back' in current_card:
                    cards.append(current_card)
                current_card = {}
            elif line.startswith('앞면:'):
                current_card['front'] = line.replace('앞면:', '').strip()
            elif line.startswith('뒷면:'):
                current_card['back'] = line.replace('뒷면:', '').strip()
        
        # 마지막 카드 추가
        if current_card and 'front' in current_card and 'back' in current_card:
            cards.append(current_card)
        
        # 카드가 없는 경우 기본 카드 생성
        if not cards:
            cards = [{'front': '플래시카드 생성 실패', 'back': '다시 시도해주세요'}]
    
    except Exception as e:
        cards = [{'front': '파싱 오류', 'back': f'오류: {str(e)}'}]
    
    # HTML 템플릿
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Malgun Gothic', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 800px;
                margin: 0 auto;
                text-align: center;
            }}
            
            .header {{
                color: white;
                margin-bottom: 30px;
            }}
            
            .flashcard-container {{
                perspective: 1000px;
                margin: 20px auto;
                width: 400px;
                height: 250px;
            }}
            
            .flashcard {{
                position: relative;
                width: 100%;
                height: 100%;
                text-align: center;
                transition: transform 0.6s;
                transform-style: preserve-3d;
                cursor: pointer;
            }}
            
            .flashcard.flipped {{
                transform: rotateY(180deg);
            }}
            
            .card-face {{
                position: absolute;
                width: 100%;
                height: 100%;
                backface-visibility: hidden;
                border-radius: 15px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                box-sizing: border-box;
            }}
            
            .card-front {{
                background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                color: #333;
            }}
            
            .card-back {{
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #333;
                transform: rotateY(180deg);
            }}
            
            .card-content {{
                font-size: 18px;
                line-height: 1.4;
                word-break: keep-all;
            }}
            
            .controls {{
                margin: 30px 0;
            }}
            
            .btn {{
                background: rgba(255,255,255,0.2);
                color: white;
                border: 2px solid white;
                padding: 12px 24px;
                margin: 0 10px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s;
            }}
            
            .btn:hover {{
                background: white;
                color: #667eea;
            }}
            
            .progress {{
                color: white;
                font-size: 18px;
                margin: 20px 0;
            }}
            
            .card-indicator {{
                color: white;
                font-size: 14px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎴 {title}</h1>
                <p>카드를 클릭하면 뒤집어집니다</p>
            </div>
            
            <div class="progress">
                <span id="current-card">1</span> / <span id="total-cards">{len(cards)}</span>
            </div>
            
            <div class="flashcard-container">
                <div class="flashcard" id="flashcard" onclick="flipCard()">
                    <div class="card-face card-front">
                        <div class="card-content" id="front-content">
                            {cards[0]['front'] if cards else '카드가 없습니다'}
                        </div>
                    </div>
                    <div class="card-face card-back">
                        <div class="card-content" id="back-content">
                            {cards[0]['back'] if cards else ''}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card-indicator">
                💡 카드를 클릭해서 답을 확인하세요
            </div>
            
            <div class="controls">
                <button class="btn" onclick="previousCard()">⬅️ 이전</button>
                <button class="btn" onclick="flipCard()">🔄 뒤집기</button>
                <button class="btn" onclick="nextCard()">다음 ➡️</button>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="shuffleCards()">🔀 섞기</button>
                <button class="btn" onclick="resetCards()">🔄 처음부터</button>
            </div>
        </div>
        
        <script>
            const cards = {cards};
            let currentIndex = 0;
            let isFlipped = false;
            
            function updateCard() {{
                const frontContent = document.getElementById('front-content');
                const backContent = document.getElementById('back-content');
                const currentCardSpan = document.getElementById('current-card');
                
                if (cards[currentIndex]) {{
                    frontContent.textContent = cards[currentIndex].front;
                    backContent.textContent = cards[currentIndex].back;
                    currentCardSpan.textContent = currentIndex + 1;
                }}
                
                // 카드 뒤집기 상태 초기화
                const flashcard = document.getElementById('flashcard');
                flashcard.classList.remove('flipped');
                isFlipped = false;
            }}
            
            function flipCard() {{
                const flashcard = document.getElementById('flashcard');
                flashcard.classList.toggle('flipped');
                isFlipped = !isFlipped;
            }}
            
            function nextCard() {{
                if (currentIndex < cards.length - 1) {{
                    currentIndex++;
                    updateCard();
                }}
            }}
            
            function previousCard() {{
                if (currentIndex > 0) {{
                    currentIndex--;
                    updateCard();
                }}
            }}
            
            function shuffleCards() {{
                for (let i = cards.length - 1; i > 0; i--) {{
                    const j = Math.floor(Math.random() * (i + 1));
                    [cards[i], cards[j]] = [cards[j], cards[i]];
                }}
                currentIndex = 0;
                updateCard();
            }}
            
            function resetCards() {{
                currentIndex = 0;
                updateCard();
            }}
            
            // 키보드 단축키
            document.addEventListener('keydown', function(event) {{
                switch(event.key) {{
                    case 'ArrowLeft':
                        previousCard();
                        break;
                    case 'ArrowRight':
                        nextCard();
                        break;
                    case ' ':
                        event.preventDefault();
                        flipCard();
                        break;
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_template

# 🆕 사용자 맞춤 학습 이력 관리
def load_user_study_history(username, filename_prefix="user_history"):
    """지식 카드 업데이트"""
    try:
        # 순환 import 방지를 위해 함수들을 여기서 직접 구현
        def create_knowledge_card_local(pdf_name: str, upload_date: str, question_count: int, last_study: str):
            return {
                "pdf_name": pdf_name,
                "upload_date": upload_date,
                "question_count": question_count,
                "last_study": last_study,
                "card_id": f"card_{pdf_name.replace('.pdf', '').replace(' ', '_')}"
            }
        
        def load_knowledge_cards_local(username: str = None):
            try:
                if username:
                    cards_file = f"users/{username}_knowledge_cards.json"
                else:
                    cards_file = "knowledge_cards.json"
                
                if os.path.exists(cards_file):
                    with open(cards_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return []
            except:
                return []
        
        def save_knowledge_cards_local(cards, username: str = None):
            try:
                if username:
                    cards_file = f"users/{username}_knowledge_cards.json"
                    os.makedirs("users", exist_ok=True)
                else:
                    cards_file = "knowledge_cards.json"
                
                with open(cards_file, 'w', encoding='utf-8') as f:
                    json.dump(cards, f, ensure_ascii=False, indent=2)
                return True
            except:
                return False
        
        cards = load_knowledge_cards_local(username)
        
        # 기존 카드 찾기
        existing_card = None
        for i, card in enumerate(cards):
            if card['pdf_name'] == pdf_name:
                existing_card = i
                break
        
        # 카드 업데이트 또는 생성
        if existing_card is not None:
            cards[existing_card]['question_count'] += 1
            cards[existing_card]['last_study'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        else:
            new_card = create_knowledge_card_local(
                pdf_name=pdf_name,
                upload_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                question_count=1,
                last_study=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            )
            cards.append(new_card)
        
        save_knowledge_cards_local(cards, username)
        return True
    except Exception as e:
        print(f"지식 카드 업데이트 오류: {e}")
        return False

# 🆕 사용자 맞춤 학습 이력 관리
def load_user_study_history(username, filename_prefix="user_history"):
    """사용자별 학습 이력 로드"""
    filename = f"{filename_prefix}_{username}.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"사용자 이력 로드 오류: {e}")
        return []

def save_user_study_history(username, question, answer, topic=None, difficulty=None, filename_prefix="user_history"):
    """사용자별 학습 이력 저장"""
    filename = f"{filename_prefix}_{username}.json"
    try:
        # 기존 이력 로드
        history = load_user_study_history(username, filename_prefix)
        
        # 새 기록 추가
        new_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "topic": topic,
            "difficulty": difficulty,
            "study_time": random.randint(30, 300)  # 30초-5분 랜덤
        }
        
        history.append(new_record)
        
        # 최근 100개만 유지
        if len(history) > 100:
            history = history[-100:]
        
        # 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        print(f"사용자 이력 저장 오류: {e}")
        return False

def calculate_user_progress(history, username):
    """사용자별 상세 진행률 계산"""
    if not history:
        return 0, {}, {}
    
    # 기본 진행률 계산
    topics = {}
    study_patterns = {
        'time_distribution': {},
        'difficulty_preference': {},
        'topic_interest': {}
    }
    
    for record in history:
        # 주제별 분류
        question = record['question'].lower()
        topic_found = False
        
        # 간단한 키워드 매칭으로 주제 분류
        topic_keywords = {
            'Chapter 1': ['1장', '첫번째', '처음', 'chapter 1', '기초'],
            'Chapter 2': ['2장', '두번째', 'chapter 2', '중급'],
            'Chapter 3': ['3장', '세번째', 'chapter 3', '고급'],
            '개념': ['개념', '정의', '의미', '뜻'],
            '공식': ['공식', '수식', '계산', '식'],
            '예제': ['예제', '문제', '풀이', '해답']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in question for keyword in keywords):
                topics[topic] = topics.get(topic, 0) + 1
                topic_found = True
                break
        
        if not topic_found:
            topics['기타'] = topics.get('기타', 0) + 1
        
        # 시간 패턴 분석
        timestamp = record['timestamp']
        hour = datetime.datetime.fromisoformat(timestamp).hour
        time_slot = f"{hour:02d}:00"
        study_patterns['time_distribution'][time_slot] = study_patterns['time_distribution'].get(time_slot, 0) + 1
        
        # 난이도 선호도
        difficulty = record.get('difficulty', 'medium')
        study_patterns['difficulty_preference'][difficulty] = study_patterns['difficulty_preference'].get(difficulty, 0) + 1
    
    # 진행률 계산 (더 정교한 방식)
    total_chapters = 10
    progress_percentage = min(len(topics) / total_chapters * 100, 100)
    
    # 학습 일관성 보너스
    if len(history) > 20:
        progress_percentage += 10
    if len(set([record['timestamp'][:10] for record in history])) > 7:  # 7일 이상 학습
        progress_percentage += 15
    
    progress_percentage = min(progress_percentage, 100)
    
    return progress_percentage, topics, study_patterns

def generate_learning_recommendations(username, history, current_topic=None):
    """AI 기반 맞춤 학습 추천"""
    try:
        client = openai.OpenAI()
        
        # 사용자 학습 패턴 분석
        recent_questions = [record['question'] for record in history[-10:]]
        topics_studied = list(set([record.get('topic', '일반') for record in history]))
        
        prompt = f"""
        사용자 {username}의 학습 패턴을 분석하여 맞춤 추천을 생성해주세요.
        
        최근 질문들:
        {chr(10).join(recent_questions)}
        
        학습한 주제들: {', '.join(topics_studied)}
        현재 학습 중인 주제: {current_topic or '없음'}
        
        다음 형식으로 추천해주세요:
        1. 복습이 필요한 영역
        2. 다음 학습 추천 주제
        3. 학습 방법 제안
        4. 예상 소요 시간
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"추천 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 사용자 관리 시스템
import hashlib
import datetime

def create_user_profile(username, email, plan="free"):
    """사용자 프로필 생성"""
    user_data = {
        "username": username,
        "email": email,
        "plan": plan,  # free, premium, instructor
        "created_date": datetime.datetime.now().isoformat(),
        "pdf_count": 0,
        "quiz_count": 0,
        "study_time": 0,
        "achievements": [],
        "learning_streak": 0,
        "last_login": datetime.datetime.now().isoformat()
    }
    
    # 사용자 파일 저장
    user_file = f"users/{username}.json"
    os.makedirs("users", exist_ok=True)
    
    try:
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"사용자 생성 오류: {e}")
        return False

def load_user_profile(username):
    """사용자 프로필 로드"""
    user_file = f"users/{username}.json"
    try:
        with open(user_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"사용자 로드 오류: {e}")
        return None

def update_user_activity(username, activity_type, data=None):
    """사용자 활동 업데이트"""
    user_profile = load_user_profile(username)
    if not user_profile:
        return False
    
    # 활동별 업데이트
    if activity_type == "pdf_processed":
        user_profile["pdf_count"] += 1
    elif activity_type == "multi_document_processed":
        user_profile["multi_doc_count"] = user_profile.get("multi_doc_count", 0) + 1
        user_profile["total_documents"] = user_profile.get("total_documents", 0) + data.get("count", 1)
    elif activity_type == "quiz_completed":
        user_profile["quiz_count"] += 1
    elif activity_type == "flashcard_generated":
        user_profile["flashcard_count"] = user_profile.get("flashcard_count", 0) + 1
    elif activity_type == "question_asked":
        user_profile["question_count"] = user_profile.get("question_count", 0) + 1
    elif activity_type == "premium_quiz_generated":
        user_profile["premium_quiz_count"] = user_profile.get("premium_quiz_count", 0) + 1
    elif activity_type == "study_session":
        user_profile["study_time"] += data.get("duration", 0)
    
    # 마지막 활동 시간 업데이트
    user_profile["last_activity"] = datetime.datetime.now().isoformat()
    
    # 연속 학습일 계산
    today = datetime.datetime.now().date()
    last_login = datetime.datetime.fromisoformat(user_profile.get("last_login", user_profile["created_date"])).date()
    
    if (today - last_login).days == 1:
        user_profile["learning_streak"] = user_profile.get("learning_streak", 0) + 1
    elif (today - last_login).days > 1:
        user_profile["learning_streak"] = 1
    
    user_profile["last_login"] = datetime.datetime.now().isoformat()
    
    # 업적 시스템
    check_achievements(user_profile)
    
    # 저장
    user_file = f"users/{username}.json"
    try:
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(user_profile, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"사용자 업데이트 오류: {e}")
        return False

def check_achievements(user_profile):
    """업적 확인 및 추가"""
    achievements = user_profile.get("achievements", [])
    
    # 기본 업적들
    if user_profile["pdf_count"] >= 1 and "📚 첫 PDF 처리" not in achievements:
        achievements.append("📚 첫 PDF 처리")
    if user_profile["pdf_count"] >= 5 and "📖 PDF 마스터" not in achievements:
        achievements.append("📖 PDF 마스터")
    if user_profile["quiz_count"] >= 5 and "🧩 퀴즈 초보자" not in achievements:
        achievements.append("🧩 퀴즈 초보자")
    if user_profile["quiz_count"] >= 20 and "🎯 퀴즈 마스터" not in achievements:
        achievements.append("🎯 퀴즈 마스터")
    if user_profile.get("flashcard_count", 0) >= 3 and "🎴 플래시카드 수집가" not in achievements:
        achievements.append("🎴 플래시카드 수집가")
    if user_profile.get("question_count", 0) >= 50 and "💬 질문왕" not in achievements:
        achievements.append("💬 질문왕")
    
    # 연속 학습 업적
    streak = user_profile.get("learning_streak", 0)
    if streak >= 3 and "🔥 3일 연속 학습" not in achievements:
        achievements.append("🔥 3일 연속 학습")
    if streak >= 7 and "⭐ 일주일 연속 학습" not in achievements:
        achievements.append("⭐ 일주일 연속 학습")
    if streak >= 30 and "👑 한달 연속 학습" not in achievements:
        achievements.append("👑 한달 연속 학습")
    
    # 시간 기반 업적
    study_time = user_profile.get("study_time", 0)
    if study_time >= 1800 and "⏰ 30분 학습" not in achievements:  # 30분
        achievements.append("⏰ 30분 학습")
    if study_time >= 3600 and "💪 1시간 집중" not in achievements:  # 1시간
        achievements.append("💪 1시간 집중")
    if study_time >= 18000 and "🏆 5시간 마라톤" not in achievements:  # 5시간
        achievements.append("🏆 5시간 마라톤")
    
    # 프리미엄 업적
    if user_profile.get("premium_quiz_count", 0) >= 5 and "💎 프리미엄 사용자" not in achievements:
        achievements.append("💎 프리미엄 사용자")
    
    # 다중 문서 업적
    multi_doc_count = user_profile.get("multi_doc_count", 0)
    total_documents = user_profile.get("total_documents", 0)
    
    if multi_doc_count >= 1 and "📚 다중 문서 입문" not in achievements:
        achievements.append("📚 다중 문서 입문")
    if multi_doc_count >= 5 and "🔗 통합 학습자" not in achievements:
        achievements.append("🔗 통합 학습자")
    if total_documents >= 20 and "📖 문서 컬렉터" not in achievements:
        achievements.append("📖 문서 컬렉터")
    
    # 특별 업적
    total_activities = (user_profile["pdf_count"] + user_profile["quiz_count"] + 
                       user_profile.get("flashcard_count", 0) + user_profile.get("question_count", 0) +
                       multi_doc_count)
    if total_activities >= 100 and "🌟 올라운더" not in achievements:
        achievements.append("🌟 올라운더")
    
    user_profile["achievements"] = achievements

# 🆕 수익화 기능들
def check_plan_limits(username, feature):
    """플랜별 제한 확인"""
    user_profile = load_user_profile(username)
    if not user_profile:
        return False, "사용자를 찾을 수 없습니다."
    
    plan = user_profile.get("plan", "free")
    
    if plan == "free":
        if feature == "pdf_upload" and user_profile["pdf_count"] >= 1:
            return False, "🚫 무료 플랜은 1개 PDF만 처리 가능합니다. 프리미엄으로 업그레이드하세요!"
        if feature == "quiz_generation" and user_profile["quiz_count"] >= 3:
            return False, "🚫 무료 플랜은 3개 퀴즈만 생성 가능합니다. 프리미엄으로 업그레이드하세요!"
        if feature == "flashcard_generation" and user_profile.get("flashcard_count", 0) >= 2:
            return False, "🚫 무료 플랜은 2개 플래시카드만 생성 가능합니다."
        if feature == "multi_document":
            return False, "🚫 다중 문서 기능은 프리미엄 플랜이 필요합니다."
        if feature == "premium_features":
            return False, "🚫 프리미엄 기능입니다. 업그레이드가 필요합니다."
    
    elif plan == "premium":
        # 프리미엄은 대부분 제한 없음, 단 일부 고급 기능은 제한
        if feature == "instructor_features":
            return False, "🚫 강사 플랜 전용 기능입니다."
    
    elif plan == "instructor":
        # 강사 플랜은 모든 기능 사용 가능
        pass
    
    return True, "사용 가능"

def generate_share_link(pdf_name, username):
    """강사용 공유 링크 생성"""
    # 간단한 해시 기반 링크 생성
    link_data = f"{pdf_name}_{username}_{datetime.datetime.now().isoformat()}"
    link_hash = hashlib.md5(link_data.encode()).hexdigest()[:10]
    
    share_link = f"https://your-domain.com/shared/{link_hash}"
    
    # 공유 정보 저장
    share_data = {
        "pdf_name": pdf_name,
        "creator": username,
        "created_date": datetime.datetime.now().isoformat(),
        "access_count": 0,
        "link_hash": link_hash
    }
    
    os.makedirs("shared_links", exist_ok=True)
    with open(f"shared_links/{link_hash}.json", 'w', encoding='utf-8') as f:
        json.dump(share_data, f, ensure_ascii=False, indent=2)
    
    return share_link

def generate_premium_quiz(text, difficulty="medium", num_questions=10):
    """프리미엄 예상문제 생성"""
    try:
        client = openai.OpenAI()
        
        difficulty_prompts = {
            "easy": "기본적인 개념 이해를 확인하는",
            "medium": "응용력을 요구하는",
            "hard": "심화 사고력을 평가하는"
        }
        
        prompt = f"""
        다음 텍스트를 바탕으로 {difficulty_prompts[difficulty]} {num_questions}개의 고품질 예상문제를 생성해주세요.
        
        문제 유형:
        1. 객관식 (4지선다)
        2. 단답형
        3. 서술형 (간단한 설명)
        
        각 문제는 다음 형식으로:
        
        [문제 유형] Q1: [문제]
        1) [선택지1] (객관식인 경우)
        2) [선택지2]
        3) [선택지3]
        4) [선택지4]
        정답: [답]
        해설: [상세한 해설과 관련 개념 설명]
        난이도: {difficulty}
        
        텍스트:
        {text[:4000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"프리미엄 문제 생성 중 오류가 발생했습니다: {str(e)}"

# 🆕 학습 분석 리포트
def generate_learning_report(username):
    """개인 맞춤 학습 분석 리포트"""
    user_profile = load_user_profile(username)
    history = load_study_history(f"users/{username}_history.json")
    
    if not user_profile or not history:
        return "데이터가 부족합니다."
    
    try:
        client = openai.OpenAI()
        
        # 학습 패턴 분석
        recent_questions = [h['question'] for h in history[-10:]]
        study_topics = analyze_study_topics(recent_questions)
        
        prompt = f"""
        다음 학습자의 데이터를 분석하여 개인 맞춤 학습 리포트를 작성해주세요:
        
        학습자 정보:
        - 총 PDF 처리: {user_profile['pdf_count']}개
        - 총 퀴즈 완료: {user_profile['quiz_count']}개
        - 총 학습 시간: {user_profile['study_time']//60}분
        - 최근 질문들: {recent_questions}
        
        리포트 구성:
        1. 학습 현황 요약
        2. 강점 분야
        3. 보완이 필요한 영역
        4. 맞춤 학습 추천
        5. 다음 단계 제안
        
        친근하고 격려하는 톤으로 작성해주세요.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"리포트 생성 중 오류가 발생했습니다: {str(e)}"

def analyze_study_topics(questions):
    """질문에서 학습 주제 분석"""
    topics = {}
    for question in questions:
        # 간단한 키워드 분석
        words = question.lower().split()
        for word in words:
            if len(word) > 2:  # 2글자 이상 단어만
                topics[word] = topics.get(word, 0) + 1
    
    # 상위 5개 주제 반환
    return dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5])# 추가 누락된 함수들
def load_chat_history(username):
    """챗 기록 로드"""
    try:
        chat_file = f"users/{username}_chat.json"
        with open(chat_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"챗 기록 로드 오류: {e}")
        return []

def update_user_usage(username, feature_type):
    """사용자 사용량 업데이트"""
    try:
        users_file = "users/users.json"
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        if username in users:
            usage = users[username].get("usage_stats", {})
            today = datetime.datetime.now().date().isoformat()
            
            # API 호출 카운트 업데이트
            last_call = usage.get("last_api_call", "")
            if last_call.startswith(today):
                usage["api_calls_today"] = usage.get("api_calls_today", 0) + 1
            else:
                usage["api_calls_today"] = 1
            
            usage["last_api_call"] = datetime.datetime.now().isoformat()
            
            # 기능별 카운트 업데이트
            if feature_type == "pdf_upload":
                usage["total_pdfs"] = usage.get("total_pdfs", 0) + 1
            elif feature_type == "quiz_generation":
                usage["total_quizzes"] = usage.get("total_quizzes", 0) + 1
            elif feature_type == "question_asked":
                usage["total_questions"] = usage.get("total_questions", 0) + 1
            
            users[username]["usage_stats"] = usage
            
            with open(users_file, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용량 업데이트 오류: {e}")
        return False

def generate_direct_answer(text, question):
    """벡터스토어 없이 직접 텍스트 기반 답변 생성"""
    try:
        client = openai.OpenAI()
        
        # 텍스트가 너무 길면 관련 부분만 추출
        relevant_text = text[:4000] if len(text) > 4000 else text
        
        prompt = f"""
        다음 텍스트를 바탕으로 질문에 답변해주세요.
        
        텍스트:
        {relevant_text}
        
        질문: {question}
        
        답변:
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

def save_chat_message(username, message, response, message_type="qa"):
    """챗 메시지 저장"""
    try:
        os.makedirs("users", exist_ok=True)
        chat_file = f"users/{username}_chat.json"
        
        # 기존 챗 기록 로드
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
        except FileNotFoundError:
            chat_history = []
        
        # 새 메시지 추가
        new_message = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "type": message_type
        }
        
        chat_history.append(new_message)
        
        # 최근 100개만 유지
        if len(chat_history) > 100:
            chat_history = chat_history[-100:]
        
        # 저장
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"챗 기록 저장 오류: {e}")
        return False

def save_user_study_history(username, question, answer, topic="일반"):
    """사용자별 학습 이력 저장"""
    try:
        os.makedirs("users", exist_ok=True)
        history_file = f"users/{username}_history.json"
        
        # 기존 이력 로드
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # 새 기록 추가
        new_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "topic": topic
        }
        
        history.append(new_record)
        
        # 최근 100개만 유지
        if len(history) > 100:
            history = history[-100:]
        
        # 저장
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"사용자 이력 저장 오류: {e}")
        return False

def save_study_history(question, answer, filename="study_history.json"):
    """기본 학습 이력 저장"""
    try:
        # 기존 이력 로드
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # 새 기록 추가
        new_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer
        }
        
        history.append(new_record)
        
        # 최근 50개만 유지
        if len(history) > 50:
            history = history[-50:]
        
        # 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        print(f"이력 저장 오류: {e}")
        return False

def generate_simple_answer(text, question):
    """가장 기본적인 텍스트 매칭 기반 답변"""
    try:
        # 질문에서 키워드 추출
        question_lower = question.lower()
        text_lower = text.lower()
        
        # 간단한 키워드 매칭
        keywords = [word for word in question_lower.split() if len(word) > 2]
        
        # 관련 문장 찾기
        sentences = text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= 3:  # 최대 3개 문장
                    break
        
        if relevant_sentences:
            return " ".join(relevant_sentences)
        else:
            return "죄송합니다. 해당 질문에 대한 관련 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요."
    
    except Exception as e:
        return f"기본 답변 생성 실패: {str(e)}"

def summarize_text(text, max_length=500):
    """텍스트 요약 기능"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 {max_length}자 이내로 요약해주세요. 
        주요 개념과 핵심 내용을 포함하여 학습에 도움이 되도록 요약해주세요.
        
        텍스트:
        {text[:3000]}  # 너무 긴 텍스트는 잘라서 처리
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"요약 생성 중 오류가 발생했습니다: {str(e)}"

def generate_quiz(text, num_questions=5):
    """퀴즈 생성 기능"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 바탕으로 {num_questions}개의 객관식 퀴즈를 생성해주세요.
        각 문제는 4개의 선택지를 가지고, 정답은 1개입니다.
        
        형식:
        Q1: [질문]
        1) [선택지1]
        2) [선택지2] 
        3) [선택지3]
        4) [선택지4]
        정답: [번호]
        해설: [간단한 설명]
        
        텍스트:
        {text[:2000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}"

def generate_short_answer_quiz(text, num_questions=5):
    """단답형 퀴즈 생성 기능"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 바탕으로 {num_questions}개의 단답형 퀴즈를 생성해주세요.
        각 문제는 간단한 단어나 구문으로 답할 수 있어야 합니다.
        
        형식:
        Q1: [질문]
        정답: [단답]
        해설: [간단한 설명]
        
        텍스트:
        {text[:2000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"단답형 퀴즈 생성 중 오류가 발생했습니다: {str(e)}"

def generate_flashcards(text, num_cards=10):
    """플래시카드 생성 기능"""
    try:
        # 텍스트 유효성 검사
        if not text or len(text.strip()) < 100:
            return "플래시카드를 생성하기에 텍스트가 너무 짧습니다. 더 많은 내용이 필요합니다."
        
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OpenAI API 키가 설정되지 않았습니다."
        
        client = openai.OpenAI(api_key=api_key)
        
        # 텍스트 길이 제한 (안전하게)
        safe_text = text[:3000] if len(text) > 3000 else text
        
        prompt = f"""
        다음 텍스트를 바탕으로 {num_cards}개의 학습 플래시카드를 생성해주세요.
        각 카드는 앞면(질문/개념)과 뒷면(답변/설명)으로 구성됩니다.
        
        형식:
        카드 1:
        앞면: [핵심 개념이나 질문]
        뒷면: [상세한 설명이나 답변]
        
        카드 2:
        앞면: [핵심 개념이나 질문]
        뒷면: [상세한 설명이나 답변]
        
        플래시카드는 다음과 같은 유형으로 만들어주세요:
        - 정의 암기 (개념 → 정의)
        - 공식 암기 (공식명 → 공식)
        - 예시 문제 (문제 → 해답)
        - 핵심 키워드 (키워드 → 설명)
        
        텍스트:
        {safe_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"플래시카드 생성 중 오류가 발생했습니다: {str(e)}"

def load_user_study_history(username):
    """사용자별 학습 이력 로드"""
    try:
        history_file = f"users/{username}_history.json"
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"사용자 이력 로드 오류: {e}")
        return []

def load_study_history(filename="study_history.json"):
    """기본 학습 이력 로드"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"이력 로드 오류: {e}")
        return []

def calculate_progress(history, total_chapters=10):
    """학습 진행률 계산"""
    if not history:
        return 0, {}
    
    # 질문 키워드 분석으로 학습 영역 파악
    topics = {}
    for record in history:
        question = record['question'].lower()
        # 간단한 키워드 매칭으로 주제 분류
        if any(word in question for word in ['1장', '첫번째', '처음', 'chapter 1']):
            topics['Chapter 1'] = topics.get('Chapter 1', 0) + 1
        elif any(word in question for word in ['2장', '두번째', 'chapter 2']):
            topics['Chapter 2'] = topics.get('Chapter 2', 0) + 1
        # ... 더 많은 챕터 매칭 로직
    
    progress_percentage = min(len(topics) / total_chapters * 100, 100)
    return progress_percentage, topics

def calculate_user_progress(history, username):
    """사용자별 학습 진행률 계산"""
    if not history:
        return 0, {}, {}
    
    # 주제별 질문 수 계산
    topics = {}
    for record in history:
        topic = record.get('topic', '일반')
        topics[topic] = topics.get(topic, 0) + 1
    
    # 진행률 계산 (질문 수 기반)
    total_questions = len(history)
    progress_percentage = min(total_questions * 2, 100)  # 50개 질문 = 100%
    
    # 학습 패턴 분석
    study_patterns = {
        "most_active_topic": max(topics.items(), key=lambda x: x[1])[0] if topics else "없음",
        "total_topics": len(topics),
        "avg_questions_per_topic": total_questions / len(topics) if topics else 0
    }
    
    return progress_percentage, topics, study_patterns

def generate_learning_recommendations(username, history):
    """학습 추천 생성"""
    try:
        if not history:
            return "아직 학습 이력이 없습니다. 질문을 시작해보세요!"
        
        # 최근 질문 분석
        recent_questions = [record['question'] for record in history[-5:]]
        
        client = openai.OpenAI()
        prompt = f"""
        다음 최근 질문들을 분석하여 학습자에게 맞춤 추천을 해주세요:
        
        최근 질문들:
        {chr(10).join(recent_questions)}
        
        다음 형식으로 추천해주세요:
        1. 학습 패턴 분석
        2. 추천 학습 주제
        3. 다음 단계 제안
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"추천 생성 실패: {str(e)}"

# 다중 벡터스토어 관리 클래스
class MultiVectorStoreManager:
    def __init__(self):
        self.vectorstores = {}
        self.document_mapping = {}
    
    def add_document(self, doc_name, text):
        """개별 문서의 벡터스토어 생성"""
        try:
            vectorstore = create_vectorstore(text)
            if vectorstore:
                self.vectorstores[doc_name] = vectorstore
                self.document_mapping[doc_name] = len(text)
                return True
            return False
        except Exception as e:
            print(f"문서 추가 오류: {e}")
            return False
    
    def search_across_documents(self, query, k=5):
        """모든 문서에서 검색"""
        results = []
        for doc_name, vectorstore in self.vectorstores.items():
            try:
                docs = vectorstore.similarity_search(query, k=k//len(self.vectorstores) + 1)
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "source": doc_name,
                        "score": 0  # 실제 구현에서는 유사도 점수 계산
                    })
            except Exception as e:
                print(f"검색 오류 ({doc_name}): {e}")
        
        return results[:k]
    
    def get_document_stats(self):
        """문서 통계 반환"""
        return {
            "total_documents": len(self.vectorstores),
            "document_sizes": self.document_mapping,
            "total_size": sum(self.document_mapping.values())
        }

def create_cross_document_qa_chain(vectorstore):
    """다중 문서 질의응답 체인 생성"""
    try:
        if vectorstore is None:
            print("벡터스토어가 없습니다.")
            return None
            
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API 키가 설정되지 않았습니다.")
            return None
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 더 많은 문서에서 검색
        
        # 최신 모델 사용
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        # 커스텀 프롬프트로 출처 정보 포함
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        template = """
        다음 문서들의 내용을 바탕으로 질문에 답변해주세요. 
        답변 시 어떤 문서에서 정보를 가져왔는지 출처를 명시해주세요.
        
        관련 문서 내용:
        {context}
        
        질문: {question}
        
        답변 (출처 포함):
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
    except Exception as e:
        print(f"다중 문서 QA 체인 생성 오류: {e}")
        return None

# 추가 기능들
def analyze_chapters(text):
    """챕터별 분석 및 요약"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 분석하여 챕터나 주제별로 나누고, 각 부분의 핵심 내용을 요약해주세요.
        
        형식:
        ## 챕터 1: [제목]
        - 핵심 내용: [요약]
        - 중요 키워드: [키워드들]
        - 학습 포인트: [학습해야 할 점]
        
        텍스트:
        {text[:4000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"챕터 분석 중 오류가 발생했습니다: {str(e)}"

def generate_study_notes(text, style="bullet"):
    """학습 노트 자동 생성"""
    try:
        client = openai.OpenAI()
        
        if style == "bullet":
            format_instruction = "불릿 포인트 형식으로 정리"
        elif style == "outline":
            format_instruction = "아웃라인 형식으로 정리"
        else:
            format_instruction = "마인드맵 형식으로 정리"
        
        prompt = f"""
        다음 텍스트를 학습 노트로 {format_instruction}해주세요.
        학생이 복습하기 쉽도록 핵심 개념, 정의, 예시를 포함해주세요.
        
        텍스트:
        {text[:3000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"학습 노트 생성 중 오류가 발생했습니다: {str(e)}"

def generate_cornell_notes_advanced(text, style="standard", include_questions=True):
    """코넬 노트 필기법에 따른 고급 노트 생성"""
    try:
        client = openai.OpenAI()
        
        # 스타일별 프롬프트 설정
        style_instructions = {
            "standard": "균형잡힌 구성으로 핵심 내용과 세부사항을 적절히 포함",
            "detailed": "상세한 설명과 예시를 중심으로 깊이 있는 내용 구성",
            "exam_focused": "시험에 나올 만한 핵심 개념과 중요 포인트 중심으로 구성"
        }
        
        questions_instruction = """
        - **복습 질문**: 
          - [내용을 이해했는지 확인하는 질문]
          - [응용 문제나 사고 질문]
          - [연관 개념과의 관계를 묻는 질문]
        """ if include_questions else ""
        
        prompt = f"""
        다음 텍스트를 코넬 노트 필기법(Cornell Note-Taking System)에 따라 정리해주세요.
        
        **작성 스타일**: {style_instructions[style]}
        
        **코넬 노트 구성 원칙**:
        1. 노트 영역(60%): 강의/읽기 중 기록하는 주요 내용
        2. 단서 영역(20%): 복습 시 사용할 키워드와 힌트
        3. 요약 영역(20%): 전체 내용의 핵심 요약
        
        **출력 형식**:
        
        # 📝 코넬 노트 - [주제명]
        
        ---
        
        ## 📋 노트 영역 (Note-taking Area)
        
        ### 1. [주요 주제 1]
        - **정의**: [핵심 개념 정의]
        - **특징**: [주요 특징들]
        - **예시**: [구체적 예시]
        - **중요사항**: [기억해야 할 점]
        
        ### 2. [주요 주제 2]
        - **개념**: [핵심 개념]
        - **원리**: [작동 원리나 과정]
        - **응용**: [실제 적용 사례]
        
        ### 3. [주요 주제 3]
        - **내용**: [상세 내용]
        - **관련성**: [다른 개념과의 연관성]
        
        ---
        
        ## 🔑 단서 영역 (Cue Column)
        
        **핵심 키워드**: [키워드1], [키워드2], [키워드3]
        
        **기억 단서**: 
        - [기억하기 쉬운 단서나 연상법]
        - [중요한 공식이나 법칙]
        
        **중요 포인트**: 
        - [시험에 나올 만한 내용]
        - [반드시 기억해야 할 사실]
        
        {questions_instruction}
        
        ---
        
        ## 📌 요약 영역 (Summary)
        
        [전체 내용을 2-3문장으로 핵심만 간결하게 요약. 나중에 빠른 복습용으로 사용할 수 있도록 가장 중요한 내용만 포함]
        
        ---
        
        텍스트:
        {text[:4000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.2
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"코넬 노트 생성 중 오류가 발생했습니다: {str(e)}"

def generate_cornell_notes_html_advanced(cornell_content, title="코넬 노트"):
    """실제 코넬 노트 형식에 맞는 HTML 생성"""
    
    # 내용 파싱
    sections = {'notes': '', 'cue': '', 'summary': ''}
    lines = cornell_content.split('\n')
    current_section = None
    
    for line in lines:
        if '노트 영역' in line or 'Note-taking Area' in line:
            current_section = 'notes'
        elif '단서 영역' in line or 'Cue Column' in line:
            current_section = 'cue'
        elif '요약 영역' in line or 'Summary' in line:
            current_section = 'summary'
        elif current_section and line.strip() and not line.startswith('#') and not line.startswith('---'):
            sections[current_section] += line + '\n'
    
    # HTML 템플릿
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            @page {{
                size: A4;
                margin: 1cm;
            }}
            
            body {{
                font-family: 'Malgun Gothic', Arial, sans-serif;
                margin: 0;
                padding: 0;
                line-height: 1.6;
                color: #333;
            }}
            
            .cornell-container {{
                width: 100%;
                height: 100vh;
                border: 3px solid #000;
                display: flex;
                flex-direction: column;
            }}
            
            .cornell-header {{
                background-color: #f8f9fa;
                border-bottom: 2px solid #000;
                padding: 15px;
                text-align: center;
            }}
            
            .cornell-main {{
                display: flex;
                flex: 1;
            }}
            
            .cornell-cue {{
                width: 25%;
                border-right: 2px solid #000;
                padding: 15px;
                background-color: #fff8dc;
                font-size: 0.9rem;
            }}
            
            .cornell-notes {{
                width: 75%;
                padding: 15px;
                background-color: white;
            }}
            
            .cornell-summary {{
                border-top: 2px solid #000;
                padding: 15px;
                background-color: #e7f3ff;
                min-height: 100px;
            }}
        </style>
    </head>
    <body>
        <div class="cornell-container">
            <div class="cornell-header">
                <h1>📝 {title}</h1>
                <div>생성일: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}</div>
            </div>
            
            <div class="cornell-main">
                <div class="cornell-cue">
                    <h3>🔑 단서 영역</h3>
                    <div>{sections['cue']}</div>
                </div>
                
                <div class="cornell-notes">
                    <h3>📋 노트 영역</h3>
                    <div>{sections['notes']}</div>
                </div>
            </div>
            
            <div class="cornell-summary">
                <h3>📌 요약 영역</h3>
                <div>{sections['summary']}</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template

def text_to_speech(text, lang='ko'):
    """TTS 기능 (gTTS 사용)"""
    try:
        from gtts import gTTS
        import io
        
        # 텍스트가 너무 길면 요약
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # 메모리에 저장
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
    except Exception as e:
        return None

# 누락된 함수들 추가
def generate_premium_quiz(text, difficulty="medium", num_questions=10):
    """프리미엄 퀴즈 생성"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
        다음 텍스트를 바탕으로 {difficulty} 난이도의 프리미엄 퀴즈 {num_questions}개를 생성해주세요.
        
        난이도별 특징:
        - easy: 기본 개념과 정의 중심
        - medium: 응용과 이해 중심  
        - hard: 분석과 종합 중심
        
        각 문제는 다음을 포함해야 합니다:
        1. 4개의 선택지
        2. 정답과 상세 해설
        3. 출제 의도
        4. 관련 개념
        
        텍스트:
        {text[:3000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"프리미엄 퀴즈 생성 실패: {str(e)}"

def generate_share_link(content, filename, username):
    """공유 링크 생성"""
    try:
        share_id = str(uuid.uuid4())[:8]
        
        os.makedirs("shared_content", exist_ok=True)
        share_data = {
            "share_id": share_id,
            "filename": filename,
            "username": username,
            "content": content,
            "created_at": datetime.datetime.now().isoformat(),
            "access_count": 0,
            "share_link": f"https://your-domain.com/share/{share_id}",
            "is_active": True
        }
        
        with open(f"shared_content/{share_id}.json", 'w', encoding='utf-8') as f:
            json.dump(share_data, f, ensure_ascii=False, indent=2)
        
        return share_data["share_link"]
    except Exception as e:
        return f"공유 링크 생성 실패: {str(e)}"