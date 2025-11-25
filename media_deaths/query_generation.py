"""
This file generates the queries used in the media_deaths_analysis notebook. It includes the following functions:

function create_query_str(query_dict, proximity=1000)
    This function takes the query dictionary for each cause of death (from create_queries)
    and creates a query string, including both the keywords and combinations of keywords.

function create_queries_by_cause(dict_queries)
    This function takes the dictionary of all causes of death and all query terms
    and creates a string query (by calling create_query_str) for each cause of death.

function create_full_queries()
    This function creates the full queries for all causes of death.
    The output is a dictionary of the form {cause_of_death: query_string}

function create_single_keyword_queries()
    This function creates queries that take all articles with even
    a single mention into account for all causes of death.
    The output is a dictionary of the form {cause_of_death: single_keyword_query_string}

The full queries can also be found in the methodology document here:
    https://docs.owid.io/projects/etl/analyses/media_deaths/methodology/#queries-for-each-cause-of-death
"""

def create_full_queries(queries_dict):
    """Create full queries from a queries dictionary.

    Args:
        queries_dict: Dictionary of queries from config

    Returns:
        Dictionary of cause: query_string
    """
    queries = create_queries_by_cause(queries_dict)
    return queries


def create_single_keyword_queries(queries_dict):
    """Create single keyword queries from a queries dictionary.

    Args:
        queries_dict: Dictionary of queries from config

    Returns:
        Dictionary of cause: single_keyword_query_string
    """
    single_keyword_queries = {}
    for cause, query_dict in queries_dict.items():
        query_str = ""
        for term in query_dict["single_terms"][:-1]:
            query_str += f'"{term}" OR '
        query_str += f'"{query_dict["single_terms"][-1]}"'
        single_keyword_queries[cause] = query_str
    return single_keyword_queries


def create_query_str(query_dict, proximity=1000):
    """
    Create a query string from a dictionary of queries.
    If proximity is set, it will determine how far apart the keywords in the combinations can be.
    The default is 1000 words for an entire article.
    """
    comb_ls = query_dict["combinations"]
    query_str = ""
    for el in comb_ls[:-1]:
        query_str += f'"{el}"~{proximity} OR '
    query_str += f'"{comb_ls[-1]}"~{proximity}'

    query_str = f"({query_str}) AND ("

    for term in query_dict["single_terms"][:-1]:
        query_str += f'"{term}" OR '
    query_str += f'"{query_dict["single_terms"][-1]}")'

    if query_dict["exclude_terms"]:
        ex_terms = '" OR "'.join(query_dict["exclude_terms"])
        query_str += f' NOT ("{ex_terms}")'

    return query_str


def create_queries_by_cause(dict_queries):
    string_queries = {}
    for term, query_dict in dict_queries.items():
        string_queries[term] = create_query_str(query_dict)
    return string_queries
