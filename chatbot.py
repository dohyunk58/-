from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.core import ServiceContext

# 멀티 턴 대화를 위한 history 리스트
history = []

# 질문에 대한 응답을 받는 함수 ask_query
def ask_query(query, history):
    # 대화 히스토리에 현재 쿼리를 추가
    history.append({"role": "user", "content": query})
    
    # 대화 히스토리를 문자열로 병합
    formatted_history = "\n".join([f"{item['role']}: {item['content']}" for item in history])
    
    # 쿼리 엔진에 현재 히스토리를 전달하여 응답 생성
    response = query_engine.query(formatted_history)
     
    # 응답을 히스토리에 추가
    history.append({"role": "ai", "content": response})
    
    return response



# 환경변수 가져오기(API key)
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# 학습 데이터 위치 설정 후 불러오기
input_dir = r"data_pdf"
reader = SimpleDirectoryReader(input_dir=input_dir)
doc1 = reader.load_data()

input_dir = r"resources/data"
reader = SimpleDirectoryReader(input_dir=input_dir)
doc2 = reader.load_data()

doc1 += doc2

# 입베딩 다운로드
embed_model_ko = HuggingFaceEmbedding(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr") 

# llama index 설정
llm = Gemini(model_name='models/gemini-1.5-flash', request_timeout=120.0)

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20, embed_model=embed_model_ko)
index = VectorStoreIndex.from_documents(doc1,service_context=service_context,show_progress=True)

index.storage_context.persist()

query_engine = index.as_query_engine()



