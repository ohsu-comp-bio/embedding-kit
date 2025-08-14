
import pandas as pd

def extract_pathway_interactions(sif_path, relation='controls-expression-of'):
    pc = pd.read_csv(sif_path, sep="\t", 
                     header=None, names=["from", "relation", "to"])
    pc = pc[ ~(pc['from'].map(lambda x:"CHEBI" in x) | pc['to'].map(lambda x:"CHEBI" in x)) ]
    return pc[ pc["relation"] == relation ]
