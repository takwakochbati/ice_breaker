from langchain_community.tools.tavily_search import TavilySearchResults


def get_profile_url_Tavily(name: str):
    """search for twitter or linkedin profile pages"""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res
