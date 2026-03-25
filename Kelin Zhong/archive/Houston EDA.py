
# %%

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import sklearn 

# %%

Market_30yr = pd.read_csv("D:/Improvement Plan/Erdos Project/data/MORTGAGE30US.csv")
Treasure_10yr = pd.read_csv("D:/Improvement Plan/Erdos Project/data/DGS10.csv")
Housing_Activity = pd.read_csv("D:/Improvement Plan/Erdos Project/data/Housing Activity.csv")
WTI_record = pd.read_csv("D:/Improvement Plan/Erdos Project/data/DCOILWTICO.csv")
NewListing = pd.read_csv("D:/Improvement Plan/Erdos Project/data/NEWLISCOU26420.csv")

# %%
Housing_Activity.info()
Housing_Activity["Market Name"].unique()
Housing_Activity["Date"].agg(["min", "max"]) # 1990-01-01 to 2025-12-01 (monthly)

# %%
Market_30yr.info()
Market_30yr["observation_date"].agg(["min", "max"]) # 1971-04-02 to 2026-02-12 (daily)

# %%
Treasure_10yr.info()
Treasure_10yr["observation_date"].agg(["min", "max"]) # 1971-04-01 to 2026-02-13 (daily)

# %%
NewListing.info()
NewListing["observation_date"].agg(["min", "max"]) # 2016-07-01 to 2026-01-01 (monthly)

# %%
WTI_record.info()
WTI_record["observation_date"].agg(["min", "max"]) # 1986-01-02 to 2026-02-09 (daily)

# %%
### Convert to monthly
Housing_Activity["Date"] = pd.to_datetime(Housing_Activity["Date"])
Housing_Activity_monthly = (
    Housing_Activity.assign(date=pd.to_datetime(Housing_Activity["Date"]))
      .set_index("Date")
      .sort_index()
      .loc["2016-07-01":"2025-12-31"]
      .resample("M")
      .last()
      .assign(year_month=lambda x: x.index.to_period("M"))   # key like 2010-01
      .reset_index(drop=True)                                # optional: drop daily date index
)

Market_30yr["observation_date"] = pd.to_datetime(Market_30yr["observation_date"])
Market_30yr_monthly = (
    Market_30yr.assign(date=pd.to_datetime(Market_30yr["observation_date"]))
      .set_index("observation_date")
      .sort_index()
      .loc["2016-07-01":"2025-12-31"]
      .resample("M")
      .last()
      .assign(year_month=lambda x: x.index.to_period("M"))   # key like 2010-01
      .reset_index(drop=True)                                # optional: drop daily date index
)

Treasure_10yr["observation_date"] = pd.to_datetime(Treasure_10yr["observation_date"])
Treasure_10yr_monthly = (
    Treasure_10yr.assign(date=pd.to_datetime(Treasure_10yr["observation_date"]))
      .set_index("observation_date")
      .sort_index()
      .loc["2016-07-01":"2025-12-31"]
      .resample("M")
      .last()
      .assign(year_month=lambda x: x.index.to_period("M"))   # key like 2010-01
      .reset_index(drop=True)                                # optional: drop daily date index
)

NewListing["observation_date"] = pd.to_datetime(NewListing["observation_date"])
NewListing_monthly = (
    NewListing.assign(date=pd.to_datetime(NewListing["observation_date"]))
      .set_index("observation_date")
      .sort_index()
      .loc["2016-07-01":"2025-12-31"]
      .resample("M")
      .last()
      .assign(year_month=lambda x: x.index.to_period("M"))   # key like 2010-01
      .reset_index(drop=True)                                # optional: drop daily date index
)

WTI_record["observation_date"] = pd.to_datetime(WTI_record["observation_date"])
WTI_record_monthly = (
    WTI_record.assign(date=pd.to_datetime(WTI_record["observation_date"]))
      .set_index("observation_date")
      .sort_index()
      .loc["2016-07-01":"2025-12-31"]
      .resample("M")
      .last()
      .assign(year_month=lambda x: x.index.to_period("M"))   # key like 2010-01
      .reset_index(drop=True)                                # optional: drop daily date index
)


# %%
Housing_Activity_monthly.info()
Housing_Activity_monthly["year_month"].agg(["min", "max"])

# %%
Market_30yr_monthly.info()
Market_30yr_monthly["year_month"].agg(["min", "max"])

# %%
Treasure_10yr_monthly.info()
Treasure_10yr_monthly["year_month"].agg(["min", "max"])

# %%
NewListing_monthly.info()
NewListing_monthly["year_month"].agg(["min", "max"])

# %%
WTI_record_monthly.info()
WTI_record_monthly["year_month"].agg(["min", "max"])


# %%

EDA_data = (
    NewListing_monthly[["year_month","NEWLISCOU26420"]]
    .merge(Housing_Activity_monthly[["year_month","Months Inventory"]], on = "year_month", how = "left")
    .merge(Market_30yr_monthly[["year_month", "MORTGAGE30US"]], on = "year_month", how = "left")
    .merge(Treasure_10yr_monthly[["year_month", "DGS10"]], on = "year_month", how = "left")
    .merge(WTI_record_monthly[["year_month", "DCOILWTICO"]], on = "year_month", how = "left")
)
EDA_data["Spread"] = EDA_data["MORTGAGE30US"] - EDA_data["DGS10"]
EDA_data.info()

# %%

import matplotlib.pyplot as plt

# EDA_data["year_month"] = EDA_data["year_month"].dt.to_timestamp()
EDA_data = EDA_data.set_index("year_month")


# %%

### speard, 10-yr yield,  and new listings

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

fig, ax1 = plt.subplots(figsize=(12, 6))

# -------------------------
# LEFT AXIS
# -------------------------
EDA_data["DGS10"].plot(ax=ax1, label="DGS10", linewidth=2)
EDA_data["MORTGAGE30US"].plot(ax=ax1, label="Mortgage 30Y", linewidth=2)
EDA_data["Spread"].plot(ax=ax1, label="Spread", linewidth=2)
# EDA_data["Months Inventory"].plot(
#     ax=ax1, label="Months Inventory", linewidth=2, linestyle=":"
# )

ax1.set_ylabel("Rate / Months Inventory")
ax1.axhline(0, color="black", linewidth=1)
ax1.legend(loc="upper left")

# -------------------------
# RIGHT AXIS
# -------------------------
ax2 = ax1.twinx()

EDA_data["NEWLISCOU26420"].plot(
    ax=ax2,
    label="New Listings",
    color="red",
    linestyle="--",
    linewidth=2
)

ax2.set_ylabel("New Listings")
ax2.legend(loc="upper right")

# -------------------------
# TITLE & GRID
# -------------------------
ax1.set_title("Rates, Spread & Inventory vs New Listings")
ax1.grid(True)

plt.show()

# %%
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

fig, ax1 = plt.subplots(figsize=(12, 6))

# -------------------------
# LEFT AXIS → WTI
# -------------------------
EDA_data["DCOILWTICO"].plot(
    ax=ax1,
    color="black",
    linewidth=2,
    label="WTI Oil Price"
)

ax1.set_ylabel("WTI Price (USD)")
ax1.legend(loc="upper left")
ax1.grid(True)

# -------------------------
# RIGHT AXIS → New Listings
# -------------------------
ax2 = ax1.twinx()

EDA_data["NEWLISCOU26420"].plot(
    ax=ax2,
    color="red",
    linestyle="--",
    linewidth=2,
    label="New Listings"
)

ax2.set_ylabel("New Listings")
ax2.legend(loc="upper right")

# -------------------------
# TITLE
# -------------------------
ax1.set_title("WTI vs New Listings")

plt.show()

# %%
