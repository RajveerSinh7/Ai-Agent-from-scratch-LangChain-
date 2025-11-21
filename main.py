from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import Tool, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_types import AgentType
from tools import search_tool, wiki_tool

load_dotenv()

# specify all fields we want as output from LLM call
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]

# initialize the LLM
llm = ChatMistralAI(
    api_key="",
    model="mistral-large-latest"
)

# initialize parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         '''
         You are a research assistant that will help generate a research paper.
         Answer the user query and use necessary tools.
         Wrap the output in this format and provide no other text \n {format_instructions}
         '''
        ),
        ("human","{query}")
    ]
).partial(format_instructions=parser.get_format_instructions())

# define tools 
tools = [search_tool, wiki_tool]

# Using AgentExecutor directly since `initialize_agent` is removed
agent_executor = AgentExecutor.from_agent_and_tools(
    llm=llm,
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=prompt,
    verbose=True
)

# run a query
query = input("what can i help you research? ")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)
