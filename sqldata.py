import sqlite3
import pandas as pd

conn = sqlite3.connect('data.db')
sql = ("SELECT * FROM History ")
q = pd.read_sql_query(sql, 
                  con=conn, 
                  parse_dates=["date"]) #, 
         #         index_col="date_norm")
q.to_csv("data.csv")