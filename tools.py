from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_web",
    func=search.run,
    description="search the web for info",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=150)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)