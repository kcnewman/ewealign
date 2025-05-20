import numpy as np
import pandas as pd
from typing import List,Tuple,Dict
import re
import os

def load_corpus(file_path):
    df = pd.read_csv(file_path,names=['english','twi'])
    df['english'] = df['english'].str.strip().str.lower()
    df["english"] = df["english"].str.strip().str.lower()
    df = df.dropna()
    
    print(f"Loaded {len(df)} parallel sentences")
    return df

def clean_en(text):
    pass

def clean_ak(text):
    pass


    
