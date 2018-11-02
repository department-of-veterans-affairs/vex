"""
Helper functions for LexDiscover.ipynb
"""

import os, glob
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd

def read_text_from_file(path, lowercase=True):
    """
    Helper function to read in the corpus. path can be either
    a directory containing .txt files that will be read in
    or a path to a single .txt file
    """
    assert os.path.exists(path), "Please check your path."
    text = ''
    if os.path.isfile(path):
        with open(path) as f:
            if lowercase:
                text += f.read().lower()
            else:
                text += f.read()
        print("1 file, {} characters".format(len(text)))
        return text
    text_files = glob.glob(os.path.join(path, '*.txt'))
    for text_file in text_files:
        with open(text_file) as f:
            text += f.read().lower()
            text += ' . ' # Adding a sentence to make sure to split up sentences at the end of docs
    print("{} files, {} characters".format(len(text_files), len(text)))
    return text

def read_base_lex(path, sep='\n'):
    assert os.path.exists(path), "Please check your path."
    with open(path) as f:
        content = f.read()
    split = content.split(sep)[:-1] # Ignore the last line, which is empty
    print("{} terms in base lexicon".format(len(split)))
    return split

def read_base_codes(path, sep='\n'):
    assert os.path.exists(path), "Please check your path."
    with open(path) as f:
        content = f.read()
    split = content.split(sep)
    codes = []
    for s in split[:-1]:  # Ignore the last line, which is empty
        try:
            codes.append(int(s))
        except ValueError as e:
            print("Invalid SNOMED code: {} . Code will be ignored".format(s))
    print("{} codes in base codes".format(len(codes)))
    return codes

def read_SQL_server(database, query, col_name, lowercase=True):
    """
    Helper function to read in the corpus from SQL server in VINCI. Requires a database name
    and a query to execute.
    e.g. read_SQL_Server("vhacdwrb02", "Select * FROM TABLE")
    Returns a pandas dataframe.
    May want to restrict this to only input/output text fields.  This works for now though I suppose.
    
    Losing Column header, col_name is column index
    """
    import win32com.client
    conn = win32com.client.Dispatch('ADODB.Connection')
    DSN = 'provider=SQLOLEDB.1;Data Source={};Integrated Security = SSPI;'.format(database)
    cnxn = conn.Open(DSN)
    print('Connection Status : [%d]' % conn.State)
    recordset = conn.execute(query)
    result = recordset.GetRows()
    print('Total Rows: [%d]' % (len(result[0])))
    df = pd.DataFrame(data=list(result))
    data = df.transpose()
    print(data.head())
    text_series = data[col_name]
    if lowercase:
        return ' . '.join(text_series).lower()
    else:
        return ' . '.join(text_series)
