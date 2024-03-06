# Run this cell and paste the API key in the prompt
import os
import getpass

os.environ['GOOGLE_API_KEY'] = getpass.getpass('Gemini API Key:')

from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document

loader = WebBaseLoader("https://valorinveste.globo.com/ultimas-noticias/")
docs = loader.load()

print(docs)


from langchain_google_genai import ChatGoogleGenerativeAI

# If there is no env variable set for API key, you can pass the API key
# to the parameter `google_api_key` of the `ChatGoogleGenerativeAI` function:
# `google_api_key="key"`.

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.7, top_p=0.85)

# To extract data from WebBaseLoader
doc_prompt = PromptTemplate.from_template("{page_content}")

# To query Gemini
llm_prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)

# Create Stuff documents chain using LCEL.
# This is called a chain because you are chaining
# together different elements with the LLM.
# In the following example, to create stuff chain,
# you will combine content, prompt, LLM model and
# output parser together like a chain using LCEL.
#
# The chain implements the following pipeline:
# 1. Extract data from documents and save to variable `text`.
# 2. This `text` is then passed to the prompt and input variable
#    in prompt is populated.
# 3. The prompt is then passed to the LLM (Gemini).
# 4. Output from the LLM is passed through an output parser
#    to structure the model response.

stuff_chain = (
    # Extract data from the documents and add to the key `text`.
    {
        "text": lambda docs: "\n\n".join(
            format_document(doc, doc_prompt) for doc in docs
        )
    }
    | llm_prompt         # Prompt for Gemini
    | llm                # Gemini function
    | StrOutputParser()  # output parser
)

stuff_chain.invoke(docs)
