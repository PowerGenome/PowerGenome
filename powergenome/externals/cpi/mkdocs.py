import cpi
import pandas as pd


pd.DataFrame([o.__dict__() for o in cpi.areas]).to_csv("./docs/areas.csv", index=False)
pd.DataFrame([o.__dict__() for o in cpi.items]).to_csv("./docs/items.csv", index=False)
pd.DataFrame([o.__dict__() for o in cpi.periods]).to_csv(
    "./docs/periods.csv", index=False
)
pd.DataFrame([o.__dict__() for o in cpi.periodicities]).to_csv(
    "./docs/periodicities.csv", index=False
)
