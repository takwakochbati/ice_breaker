from dotenv import load_dotenv
import os
from langchain.prompts.prompt import PromptTemplate

# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

# from langchain.chains.summarize.refine_prompts import prompt_template

def ice_breaker_with(name : str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)
    summary_template = """
       given the linkedin information {information} about a person, I want you to create:
       1. A short summary
       2. two interesting facts about them
       """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo")
    llm = ChatOllama(model="llama3.1")
    chain = summary_prompt_template | llm | StrOutputParser()
    #linkedin_data = scrape_linkedin_profile(
    #    linkedin_profile_url="https://linkedin.com/in/eden-marco/", mock=True
    #)
    res = chain.invoke(input={"information": linkedin_data})
    print(res)


if __name__ == "__main__":
    load_dotenv()
    print("Ice breaker Enter")
    ice_breaker_with(name="Eden Marco")


    '''
    print(os.environ["COOL_API_KEY"])

    summary_template = """
    given the linkedin information {information} about a person, I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo")
    llm = ChatOllama(model="llama3.1")
    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://linkedin.com/in/eden-marco/", mock=True
    )
    res = chain.invoke(input={"information": linkedin_data})
    print(res)
    '''
