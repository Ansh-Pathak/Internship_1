import pandas as pd
import numpy as np
df =pd.read_csv("Uncleaned_DS_jobs.csv")
df=df.drop(columns=["index"])
df =df.drop_duplicates()
df["Company Name"] =df["Company Name"].str.split("\n").str[0]
df["Company Name"] =df["Company Name"].str.strip()
df["Rating"] =df["Rating"].replace(-1, np.nan)
df["Rating"] =df["Rating"].replace(-1.0, np.nan)

df["Founded"]= df["Founded"].replace(-1, np.nan)

df["Headquarters"]=df["Headquarters"].replace("-1", "Unknown")
df["Size"] =df["Size"].replace("-1", "Unknown")
df["Type of ownership"] =df["Type of ownership"].replace("-1", "Unknown")
df["Industry"] =df["Industry"].replace("-1", "Unknown")
df["Sector"] =df["Sector"].replace("-1", "Unknown")
df["Revenue"]= df["Revenue"].replace("-1", "Unknown")
df["Revenue"] =df["Revenue"].replace("Unknown / Non-Applicable", "Unknown")
df["Competitors"] =df["Competitors"].replace("-1", "Unknown")
df["Job Title"] =df["Job Title"].str.strip()
df["Location"] =df["Location"].str.strip()
df["Headquarters"] =df["Headquarters"].str.strip()
df["Salary Estimate"]= df["Salary Estimate"].str.replace("(Glassdoor est.)", "", regex=False)
df["Salary Estimate"] = df["Salary Estimate"].str.replace("(Employer est.)", "", regex=False)
df["Salary Estimate"] =df["Salary Estimate"].str.strip()

df["Job State"] = df["Location"].str.split(",").str[-1]
df["Job State"] = df["Job State"].str.strip()

df["Company Age"] = 2026 - df["Founded"]
df.loc[df["Company Age"] < 0, "Company Age"] = np.nan

df = df.reset_index(drop=True)

df.to_csv("Final.csv", index=False)
print(df)