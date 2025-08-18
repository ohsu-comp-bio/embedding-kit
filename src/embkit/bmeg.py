
"""
BMEG

Data retrieval from BMEG instance
"""

import gripql

def connect(url="https://bmeg.io/grip", graph="rc6", cred_file="./bmeg_credentials.json"):
    """
    Establish a connection to the BMeg server and return the specified graph.

    Parameters:
        url (str): The URL of the BMeg server. Defaults to "https://bmeg.io/grip".
        graph (str): The name of the graph to retrieve. Defaults to "rc6".
        cred_file (str): The path to the JSON file containing the credentials for the BMeg server. Defaults to "./bmeg_credentials.json".

    Returns:
        gripql.Graph: A Graph object from the specified BMeg server and graph.
    """
    conn = gripql.Connection(url=url, credential_file=cred_file)
    G = conn.graph(graph)
    return G

def get_gene_map(G):
    """
    Build mapping from Ensemb Gene IDs to Hugo Symbols
    """
    gm = dict( (i[0],i[1]) for i in G.query().V().hasLabel("Gene").render(["gene_id", "symbol"]).execute() )
    return gm
