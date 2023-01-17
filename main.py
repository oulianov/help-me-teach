import os
import wget
import PyPDF2
import re

from random import sample
from dotenv import load_dotenv
from langchain.text_splitter import SpacyTextSplitter
from langchain.agents import load_tools, Tool
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


load_dotenv()


# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)


google_search = GoogleSearchAPIWrapper()


def select_snippets_with_keywords(keyword: str, text: str, window_size: int = 500):
    # Window size is in characters
    all_keyword_indexes = [m.start() for m in re.finditer(f"({keyword}).*[\\.]", text)]
    output = []
    for index in all_keyword_indexes:
        left_window = max(0, index - window_size)
        right_window = min(len(text), index + window_size)
        snippet = text[left_window:right_window]
        output.append(snippet)

    return output


def get_keywords(query):
    template = """Select by decreasing order of importance the keywords in lowercase and separated by a comma from this text : {text}"""
    prompt = PromptTemplate(template=template, input_variables=["text"])
    get_keywords_chain = LLMChain(
        prompt=prompt, llm=OpenAI(temperature=0), verbose=True
    )

    return [k.strip() for k in get_keywords_chain.run(query).split(",")][:3]


def google_search_about_education(query: str) -> str:
    keywords = get_keywords(query)

    r = google_search._google_search_results(
        f"{query} filetype:pdf site:eduscol.education.fr"
    )
    if len(r) == 0:
        return "No link"
    else:
        all_docs = []
        embeddings = HuggingFaceEmbeddings()
        vectorstore = None
        text_splitter = SpacyTextSplitter.from_tiktoken_encoder(
            chunk_size=2000, chunk_overlap=0, pipeline="fr_core_news_sm"
        )

        all_relevant_snippets = []

        for result in r[:3]:
            pdf_url = result["link"]  # Take the URL of the first result

            # Download file
            filepath = "/home/haxxor/projects/help-me-teach/data/result.pdf"
            wget.download(url=pdf_url, out=filepath)
            document_text = ""
            with open(filepath, "rb") as pdf_file:
                # creating a pdf reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    # extracting text from page
                    document_text += page.extract_text()
            os.remove(filepath)

            relevant_snippets = []

            snippets_with_keyword = []
            for keyword in keywords:
                snippets_with_keyword = select_snippets_with_keywords(
                    keyword, document_text, window_size=500
                )
            snippets_with_keyword = text_splitter.split_text(
                "\n".join(snippets_with_keyword)
            )

            min_snippets = 1
            max_snippets = 3
            max_vector_store = 50

            if len(snippets_with_keyword) > max_snippets:
                if len(snippets_with_keyword) > max_vector_store:
                    snippets_with_keyword = sample(
                        snippets_with_keyword, max_vector_store
                    )
                vectorstore = FAISS.from_texts(snippets_with_keyword, embeddings)
                relevant_snippets = vectorstore.similarity_search(query, max_snippets)
            elif len(snippets_with_keyword) < min_snippets:
                all_snippets = text_splitter.split_text(document_text)
                random_snippets = sample(
                    all_snippets,
                    min(
                        max_vector_store - len(snippets_with_keyword), len(all_snippets)
                    ),
                )
                vectorstore = FAISS.from_texts(random_snippets, embeddings)
                relevant_snippets = text_splitter.create_documents(
                    snippets_with_keyword
                )
                relevant_snippets += vectorstore.similarity_search(
                    query,
                    min(
                        max_snippets - len(snippets_with_keyword), len(random_snippets)
                    ),
                )
            else:
                relevant_snippets = text_splitter.create_documents(
                    snippets_with_keyword
                )

            all_relevant_snippets += relevant_snippets

        # Select the most relevant snippets from the collection
        vectorstore = FAISS.from_documents(all_relevant_snippets, embeddings)
        docs = vectorstore.similarity_search(query, k=3)
        # Reply to the question
        chain = load_qa_chain(llm, chain_type="refine", verbose=True)
        summary = chain.run({"question": query, "input_documents": docs})
        return summary


google_search_education = Tool(
    "Google Search about education",
    google_search_about_education,
    "A wrapper around Google Search, that returns only extracts of pdf about Education from the French government. Useful for when you need to answer questions about teaching. Input should be a short question in French.",
)

tools = [google_search_education]

# TODO : Add a case "no data for such response"
# TODO : Add a tool reply using obs


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# agent.run(
#     "exemple d'activité à organiser pour favoriser apprentissage trandisciplinaire ?"
# )

agent.run("autonomie ce1")
