import torch
import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from loguru import logger
from langchain_community.document_loaders import UnstructuredPDFLoader
import json
import shutil
import getpass
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from IPython.display import Image, display
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
import os.path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END
from PIL import Image as PILImage
import io
torch.classes.__path__ = []

# ПРОМПТЫ:
# -------------------------------------------
router_instructions = """You are an medical expert at routing a user question to a vectorstore or web search.

    The vectorstore contains documents related ONLY to medicine.

    Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

    Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

doc_grader_prompt = """Below is the retrieved content:

{document}

And here is the user's question:

{question}

Your task is to evaluate whether the document contains any information that could help answer the question — either directly, partially, or through related terminology or meaning.

Do not try to answer the question. Your job is only to check whether the document contains **relevant** content in any form.

Respond in JSON format with a single key, binary_score. The value of binary_score should be either 'yes' or 'no'.

Examples:
- If the document says "Paris is the capital of France" and the question is "What is the capital of France?" → "yes"
- If the document mentions Berlin and Germany, and the question is still "What is the capital of France?" → "no"""


doc_grader_instructions = """You are responsible for evaluating how well a retrieved document matches a user's question.

    If the content includes relevant terms or conveys similar meaning to the question, mark it as relevant."""


rag_prompt = """You are a medical assistant helping with question answering. You are not allowed to use any English words or phrases. Only Russian.

    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, consider the user's question:

    {question}

    Using only the given context, write a response to the question. Always answer exclusively in Russian. Do NOT
    use any English words or terms. Even individual words like "system", "function", or "regulating" must be translated. 
    Any English words will be considered a violation.


    Use four sentences maximum and keep the answer concise. To repeat: The response must be fully in Russian, with absolutely no English words or constructions.

    Answer:"""


hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

hallucination_grader_instructions = """

   You are a teacher grading a student's answer based on provided factual content.

You will receive three inputs:
- FACTS: key factual information extracted from documents.
- STUDENT ANSWER: a generated answer to a user question.

Your goal is to determine whether the STUDENT ANSWER is reasonably supported by the FACTS.

Use the following criteria:

1. The answer does not need to exactly repeat the FACTS, but it should reflect the same meaning or not contradict them.
2. It is acceptable for the answer to include light paraphrasing, summarization, or reasonable inferences as long as they 
are grounded in the provided content.
3. Avoid marking an answer as incorrect unless it clearly introduces new, unsupported, or contradictory information
4. Answer does NOT contain English words. """

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

answer_grader_instructions = """You are a teacher grading a quiz. 

    You will be given a QUESTION and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) The STUDENT ANSWER helps to answer the QUESTION

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""
# -------------------------------------------

load_dotenv()

logger.add(
    "log/my_logs.log",
    format="{time} {level} {message}",
    level="DEBUG",
    rotation="300 KB",                   # когда файл достигнет 100 КБ — создать новый
    compression="zip"                    # сжимать старые логи в .zip
)

# настроим веб-поиск. пользуемся тавили. берем оттуда апи ключ, чтобы модель могла уходить в поиск в интернете
os.environ["TAVILY_API_KEY"]="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"
web_search_tool = TavilySearchResults(max_results=2)


# инициилизируем ллмку. берем лламу на 3b параметров
llm = ChatOllama(model="llama3.2", temperature=0)
# эта же ллама будет возвращать нам json файлы. т.е., например, показывать, что она брала данные из бд или интернета
llm_json_mode = ChatOllama(model="llama3.2", temperature=0, format="json")

def load_pdf_documents(directory: str = "C:/Users/val04/PycharmProjects/streamlit/pdf/new") -> list:
    """
    Загружает все PDF-документы из указанной директории и объединяет текст из нескольких страниц в один блок.
    """

    logger.debug(f"Загрузка PDF-документов из директории: {directory}")
    documents = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                logger.debug(f"Найден PDF: {file}")
                loader = PyPDFLoader(full_path)
                pages = loader.load()

                # Объединяем весь текст из всех страниц в один документ
                full_text = "\n\n".join(page.page_content for page in pages)
                merged_doc = Document(
                    page_content=full_text,
                    metadata={"source": full_path}
                )
                documents.append(merged_doc)

    return documents


def split_documents(documents: list, chunk_size: int = 1500, chunk_overlap: int = 400) -> list:
    """
    Разбивает документы на чанки.
    """
    logger.debug(f"Разбиение на чанки: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)



def indexed_df():
    """
    Обновляет или создаёт FAISS-индекс по новым PDF в папке /pdf/new/.
    После индексации файлы перемещаются в /pdf/processed/.
    """
    logger.debug('...indexed_df')

    model_id = 'intfloat/multilingual-e5-base'
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={'device': 'cpu'}
    )

    df_folder = 'db/db_01'
    index_file = os.path.join(df_folder, "index.faiss")

    # Папки
    new_pdf_folder = "pdf/new"
    processed_pdf_folder = "pdf/processed"

    # Загружаем новые документы
    documents = load_pdf_documents(new_pdf_folder)

    if not documents:
        logger.debug("Нет новых PDF-документов для индексации.")
        if os.path.exists(index_file):
            return FAISS.load_local(df_folder, embeddings, allow_dangerous_deserialization=True)
        else:
            raise ValueError("Нет документов и нет индекса.")

    # Разбиваем на чанки
    split_docs = split_documents(documents)

    # Загружаем или создаем индекс
    if os.path.exists(index_file):
        df = FAISS.load_local(df_folder, embeddings, allow_dangerous_deserialization=True)
        df.add_documents(split_docs)
    else:
        df = FAISS.from_documents(split_docs, embeddings)

    # Сохраняем обновлённый индекс
    df.save_local(df_folder)
    logger.debug("Индекс обновлён и сохранён.")

    # Перемещаем обработанные файлы
    for root, _, files in os.walk(new_pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                src = os.path.join(root, file)
                dst = os.path.join(processed_pdf_folder, file)
                shutil.move(src, dst)
                logger.debug(f"Файл перемещён: {file}")

    return df

# Получаем индекс
df = indexed_df()
logger.debug('Create retriever')
# создаем ретривер от лангчейна - он будет искать релевантные документы
retriever = df.as_retriever(k=3)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class GraphState(TypedDict):
    """
    Описание структуры данных состояния графа.
    Состояние графа - это словарь, содержащий информацию, которую мы хотим передавать и изменять в каждом узле графа.
    """

    question: str     # Вопрос пользователя
    generation: str   # LLM генерация
    web_search: str   # Двоичное решение о запуске веб-поиска
    max_retries: int  # Максимальное количество повторных попыток генерации
    answers: int      # Количество сгенерированных ответов
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # Список найденных документов

def retrieve(state):
    """
    Получение документов из векторного хранилища.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Новый ключ, добавленный в state, documents, который содержит найденные документы
    """
    logger.debug("---Происходит запрос от пользователя---")

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(state["question"])
    logger.debug(f'documents = {documents}')
    return {"documents": documents}

def generate(state):
    """
    Генерация ответа на основе извлеченных документов

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Новый ключ, добавленный в state, generation, которое содержит генерацию LLM
    """
    logger.debug("---СГЕНЕРИРОВАТЬ---")

    loop_step = state.get("loop_step", 0)

    logger.debug('RAG generation')

    rag_prompt_formatted = rag_prompt.format(context=format_docs(state["documents"]), question=state["question"])
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    logger.debug(f'generation={generation}')
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Определяет, имеют ли найденные документы отношение к вопросу.
    Если какой-либо документ не является релевантным, мы установим флаг для запуска веб-поиска

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Отфильтрованые нерелевантные документы и обновлено состояние web_search state
    """

    logger.debug("---ПРОВЕРКА СООТВЕТСТВИЯ ДОКУМЕНТА ВОПРОСУ---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade.lower() == "yes":
            logger.debug("---ОЦЕНКА: ДОКУМЕНТ РЕЛЕВАНТЕН---")
            filtered_docs.append(d)
        else:
            logger.debug("---ОЦЕНКА: ДОКУМЕНТ НЕ РЕЛЕВАНТЕН---")

    #  если ни один документ не был релевантен, активируем веб-поиск
    web_search = "Yes" if not filtered_docs else "No"

    return {"documents": filtered_docs, "web_search": web_search}



def web_search(state):
    """
    Выполнение веб-поиска по запросу пользователя.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Добавленные найденные в web результаты  в documents
    """

    logger.debug("---ПОИСК в ИНТЕРНЕТЕ---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = Document(page_content="\n".join([d["content"] for d in docs]))
    logger.debug(f'web_results={web_results}')
    documents.append(web_results)
    return {"documents": documents}


# Определение логики маршрутизации запросов
def route_question(state):
    """
    Маршрутизация вопроса к веб-поиску или векторному хранилищу.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        str: Следующий узел для вызова
    """

    logger.debug("---ВОПРОС МАРШРУТА---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        logger.debug("---НАПРАВИТЬ ВОПРОС НА ПОИСК В ИНТЕРНЕТЕ---")
        return "websearch"
    elif source == "vectorstore":
        logger.debug("---НАПРАВИТЬ ВОПРОС В RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Решение о том, по какому пути продолжить генерацию ответа.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        str: Двоичное решение для следующего узла вызова
    """

    logger.debug("---РЕЛЕВАНТНЫ ЛИ ДОКУМЕНТЫ?---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.debug(
            "---РЕШЕНИЕ: НЕ ВСЕ ДОКУМЕНТЫ ИМЕЮТ ОТНОШЕНИЕ К ВОПРОСУ, ВКЛЮЧИТЕ ВЕБ-ПОИСК---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        logger.debug("---РЕШЕНИЕ: ГЕНЕРИРОВАТЬ---")
        return "generate"




def grade_generation_v_documents_and_question(state):
    """
    Проверка соответствия сгенерированного ответа документам и вопросу.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        str: Решение для следующего узла для вызова
    """

    logger.debug("---ПРОВЕРИТЬ ГАЛЛЮЦИНАЦИИ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]
    explanation = json.loads(result.content)["explanation"]
    logger.debug(f"---ОЦЕНКА ГАЛЛЮЦИНАЦИИ: {grade} --- ОБЪЯСНЕНИЕ: {explanation}")

    # Check hallucination
    if grade == "yes":
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ ОСНОВАНА НА ДОКУМЕНТАХ---")
        # Check question-answering
        logger.debug("---Оценка: ГЕНЕРАЦИЯ против ВОПРОСА---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            logger.debug("---РЕШЕНИЕ: GENERATION ОБРАЩАЕТСЯ К ВОПРОСУ---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            logger.debug("---РЕШЕНИЕ: GENERATION НЕ ОТВЕЧАЕТ НА ВОПРОС---")
            return "not useful"
        else:
            logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ НЕ ОСНОВАНА НА ДОКУМЕНТАХ, ПОВТОРИТЕ ПОПЫТКУ---")
        return "not supported"
    else:
        logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
        return "max retries"





workflow = StateGraph(GraphState)

# Определение узлов
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Построение графа
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Компиляция графа
graph = workflow.compile()
# graph_image = Image(graph.get_graph().draw_mermaid_png())
# display(graph_image)

# Сохраняем картинку в файл
graph_image = graph.get_graph().draw_mermaid_png()
with open("../graph_image.png", "wb") as png:
    png.write(graph_image)

# if __name__ == "__main__":
#     inputs = {"question": "Что такое нейрон?", "max_retries": 3}
#
#     for event in graph.stream(inputs, stream_mode="values"):
#         logger.debug(event)