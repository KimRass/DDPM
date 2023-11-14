import duckdb
import pandas as pd
import numpy as np


db_path = "/Users/jongbeomkim/Downloads/flittopic_model/image.db"
conn = duckdb.connect(database=db_path, read_only=False)
conn.sql("SHOW TABLES").to_df()
conn.table("image")
conn.execute("SHOW TABLES").fetchdf()
conn.execute("SHOW TABLES")
conn.execute("SHOW TABLES").to_df()

conn.execute("SHOW TABLES").fetchdf()["name"].tolist()

df = pd.DataFrame({
   'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
   'C': np.random.randn(8),
   'D': np.random.randn(8)
})
result = duckdb.query("SELECT A, AVG(D) FROM df GROUP BY A").to_df()
result

