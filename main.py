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
    template = """Select by decreasing order of importance the keywords in \
    lowercase and separated by a comma from this text : {text}"""
    prompt = PromptTemplate(template=template, input_variables=["text"])
    get_keywords_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    keywords = [k.strip() for k in get_keywords_chain.run(query).split(",")][:3]
    print(f"Keywords extracted: {keywords}")
    return keywords


def select_relevant_snippets(
    document_text,
    keywords,
    query,
    min_snippets=1,
    max_snippets=3,
    max_vector_store=10,
):
    # Favour snippets with the keyword, otherwise build a document vector store
    # with random snippets
    snippets_with_keyword = []
    for keyword in keywords:
        snippets_with_keyword = select_snippets_with_keywords(
            keyword, document_text, window_size=500
        )
    snippets_with_keyword = text_splitter.split_text("\n".join(snippets_with_keyword))

    if len(snippets_with_keyword) > max_snippets:
        if len(snippets_with_keyword) > max_vector_store:
            snippets_with_keyword = sample(snippets_with_keyword, max_vector_store)
        vectorstore = FAISS.from_texts(snippets_with_keyword, embeddings)
        relevant_snippets = vectorstore.similarity_search(query, max_snippets)
    elif len(snippets_with_keyword) < min_snippets:
        all_snippets = text_splitter.split_text(document_text)
        random_snippets = sample(
            all_snippets,
            min(max_vector_store - len(snippets_with_keyword), len(all_snippets)),
        )
        vectorstore = FAISS.from_texts(random_snippets, embeddings)
        relevant_snippets = text_splitter.create_documents(snippets_with_keyword)
        relevant_snippets += vectorstore.similarity_search(
            query,
            min(max_snippets - len(snippets_with_keyword), len(random_snippets)),
        )
    else:
        relevant_snippets = text_splitter.create_documents(snippets_with_keyword)

    return relevant_snippets


def google_search_about_education(query: str) -> str:
    keywords = get_keywords(query)

    print(f"Searching google for : {query}")
    google_search = GoogleSearchAPIWrapper()
    r = google_search._google_search_results(
        f"{query} filetype:pdf site:eduscol.education.fr"
    )
    if len(r) == 0:
        return "No results found, try a search query with other keywords."
    else:
        all_relevant_snippets = []

        for result in r[:3]:
            pdf_url = result["link"]  # Take the URL of the first result

            # Download file
            print(f"\nDownloading: {pdf_url}\n")
            filepath = "/Users/Nicolas_Oulianov/help-me-teach/data/result.pdf"
            try:
                wget.download(url=pdf_url, out=filepath)
            except:
                continue
            document_text = ""
            with open(filepath, "rb") as pdf_file:
                # creating a pdf reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                print(f"\nReading pdf content: {len(pdf_reader.pages)} pages")
                for page in pdf_reader.pages:
                    # extracting text from page
                    document_text += page.extract_text()
            os.remove(filepath)
            print("Selecting relevant snippets")
            relevant_snippets = select_relevant_snippets(document_text, keywords, query)
            print(f"{len(relevant_snippets)} relevant snippets selected")
            all_relevant_snippets += relevant_snippets

        # Select the most relevant snippets from the collection
        vectorstore = FAISS.from_documents(all_relevant_snippets, embeddings)
        docs = vectorstore.similarity_search(query, k=3)
        # Reply to the question
        print(f"Generating a response using the 3 most relevant snippets.")
        chain = load_qa_chain(llm, chain_type="refine", verbose=True)
        summary = chain.run({"question": query, "input_documents": docs})
        return summary


# TODO : Add a case "no data for such response"


# Loading API keys
load_dotenv()
# Loading text splitter
text_splitter = SpacyTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=0, pipeline="fr_core_news_sm"
)
# Loading local embeddings model
embeddings = HuggingFaceEmbeddings()

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Then, load the tools the language model can use
google_search_education = Tool(
    "Google Search about education",
    google_search_about_education,
    "A wrapper around Google Search, that returns only extracts of pdf"
    + "about Education from the French government."
    + "Useful for when you need to answer questions about teaching."
    + "Input should be a short question in French.",
)

tools = [google_search_education]


# Finally, initialize an agent with the tools, the language model,
# and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# agent.run(
#     "exemple d'activité à organiser pour favoriser apprentissage trandisciplinaire ?"
# )

agent.run("quelle forme prend l'autonomie en ce1 ?")
