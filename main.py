import os
import wget
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import SpacyTextSplitter
from spacy.matcher import PhraseMatcher

import spacy
import re

load_dotenv()

from langchain.agents import load_tools, Tool
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from random import sample

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)


google_search = GoogleSearchAPIWrapper()


def google_search_about_education(query: str) -> str:
    query = query.lower()
    query = query.replace("bulletin officiel", "")
    r = google_search._google_search_results(
        f"{query} filetype:pdf site:eduscol.education.fr"
    )
    if len(r) == 0:
        return "No link"
    else:
        relevant_snippets = []

        for result in r:
            pdf_url = result["link"]  # Take the URL of the first result
            try:
                wget.download(url=pdf_url, out="result.pdf", bar=None)
                document_text = ""
                with open("result.pdf", "rb") as pdf_file:
                    # creating a pdf reader object
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        # extracting text from page
                        document_text += page.extract_text()

                os.remove("result.pdf")
            except:
                print(f"Error while downloading: {pdf_url}")
                continue

            clean_doc = document_text.replace("\n\n", "\n")
            tokenizer = spacy.load("fr_core_news_sm")

            keywords = query.split(" ")

            for keyword in keywords:
                all_keyword_indexes = [
                    m.start() for m in re.finditer(f"({keyword}).*[\\.]", clean_doc)
                ]
                if len(all_keyword_indexes) > 25:
                    all_keyword_indexes = sample(all_keyword_indexes, 25)

                len(all_keyword_indexes)
                phrase_matcher = PhraseMatcher(tokenizer.vocab, attr=None)
                patterns = [tokenizer(text) for text in [keyword]]
                phrase_matcher.add("keyword", None, *patterns)

                snippets = []
                for index in all_keyword_indexes:
                    window_size = 500
                    left_window = max(0, index - window_size)
                    right_window = min(len(clean_doc), index + window_size)
                    snippet = clean_doc[left_window:right_window]
                    tokenized_snippet = tokenizer(snippet)

                    for sent in tokenized_snippet.sents:
                        for match_id, start, end in phrase_matcher(
                            tokenizer(sent.text)
                        ):
                            if tokenizer.vocab.strings[match_id] in ["keyword"]:
                                snippets.append(sent.text.replace("\n", "").strip())

                relevant_snippets += snippets

            if len(relevant_snippets) > 50:
                relevant_snippets = sample(relevant_snippets, 50)
                relevant_snippets = [s[:1000] for s in relevant_snippets]
                break

        return " | ".join(relevant_snippets)[:4000]


google_search_education = Tool(
    "Google Search about education",
    google_search_about_education,
    "A wrapper around Google Search, that returns only extracts of pdf about Education from the French government. Useful for when you need to answer questions about teaching. Input should be only relevant keywords, in French, separated by spaces.",
)

tools = [google_search_education]


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run(
    "Ecris un plan de leçon pour des CM1 concernant la résolution de problème en mode transdisciplinaire"
)
