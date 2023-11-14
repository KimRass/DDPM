import duckdb
import pandas as pd
import numpy as np


db_path = "/Users/jongbeomkim/Downloads/flittopic_model/image.db"
conn = duckdb.connect(database=db_path, read_only=False)
temp = conn.table("image")
df = conn.execute("SELECT * FROM temp").fetch_df()
df["s3url"][5]