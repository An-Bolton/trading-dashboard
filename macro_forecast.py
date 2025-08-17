import pandas as pd
import numpy as np
import requests
import streamlit as st
import os
import sqlite3
from sklearn.linear_model import LinearRegression

def save_to_sqlite(df: pd.DataFrame, table_name: str):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/makrodata.db")
    df.to_sql(table_name, conn, if_exists="replace", index=True)
    conn.close()

def macro_compare_tab():
    st.header("📊 Sammenlign indikatorer (indeks = 100) og BNP-prognose")

    startår = st.slider("Velg startår", min_value=1970, max_value=2024, value=2000)

    indikatorer = {
        "BNP": {
            "url": "https://data.ssb.no/api/v0/no/table/09842",
            "query": lambda startår: {
                "query": [
                    {
                        "code": "ContentsCode",
                        "selection": {
                            "filter": "item",
                            "values": ["BNP"]
                        }
                    },
                    {
                        "code": "Tid",
                        "selection": {
                            "filter": "item",
                            "values": [str(y) for y in range(startår, 2025)]
                        }
                    }
                ],
                "response": {"format": "json-stat2"}
            }
        },
        "KPI": {
            "url": "https://data.ssb.no/api/v0/no/table/03013",
            "query": lambda startår: {
                "query": [
                    {
                        "code": "ContentsCode",
                        "selection": {
                            "filter": "item",
                            "values": ["KPI"]
                        }
                    },
                    {
                        "code": "Tid",
                        "selection": {
                            "filter": "item",
                            "values": [f"{y}M01" for y in range(startår, 2025)]
                        }
                    }
                ],
                "response": {"format": "json-stat2"}
            }
        },
        "Arbeidsledighet": {
            "url": "https://data.ssb.no/api/v0/no/table/05111",
            "query": lambda startår: {
                "query": [
                    {
                        "code": "ArbStyrkStatus",
                        "selection": {
                            "filter": "item",
                            "values": ["2"]
                        }
                    },
                    {
                        "code": "Kjonn",
                        "selection": {
                            "filter": "item",
                            "values": ["0"]
                        }
                    },
                    {
                        "code": "Alder",
                        "selection": {
                            "filter": "item",
                            "values": ["15-74"]
                        }
                    },
                    {
                        "code": "ContentsCode",
                        "selection": {
                            "filter": "item",
                            "values": ["Prosent"]
                        }
                    },
                    {
                        "code": "Tid",
                        "selection": {
                            "filter": "item",
                            "values": [str(y) for y in range(startår, 2025)]
                        }
                    }
                ],
                "response": {"format": "json-stat2"}
            }
        }
    }

    valg = st.multiselect("Velg indikatorer", list(indikatorer.keys()), default=["BNP", "KPI", "Arbeidsledighet"])

    df_samlet = pd.DataFrame()
    bnp_rå = None

    for navn in valg:
        try:
            r = requests.post(indikatorer[navn]["url"], json=indikatorer[navn]["query"](startår))
            if r.status_code != 200:
                st.warning(f"Feil for {navn} (status {r.status_code})")
                continue

            data = r.json()
            labels = list(data["dimension"]["Tid"]["category"]["label"].values())
            values = data["value"]

            st.write(f"{navn} labels: {labels[:5]} ... [{len(labels)} totalt]")
            st.write(f"{navn} values: {values[:5]} ... [{len(values)} totalt]")

            if len(labels) != len(values):
                st.error(f"Feil: labels og values har ulik lengde for {navn} (labels={len(labels)}, values={len(values)})")
                continue
            if not any(values):
                st.warning(f"Ingen {navn}-verdier tilgjengelig fra SSB – dobbeltsjekk API, tabell og koder!")
                continue

            df = pd.DataFrame({"År": labels, navn: values})
            første_label = labels[0]
            if "M" in første_label:
                df["År"] = pd.to_datetime(df["År"], format="%YM%m", errors="coerce")
                df["År"] = df["År"].dt.to_period("M").dt.to_timestamp()
            else:
                df["År"] = pd.to_datetime(df["År"], format="%Y", errors="coerce")

            df[navn] = pd.to_numeric(df[navn], errors="coerce")
            df = df.dropna(subset=["År", navn]).set_index("År")
            df = df.sort_index()
            df = df[df.index.year >= startår]
            if not df.empty:
                if navn == "BNP":
                    bnp_rå = df.copy()  # Lagre til regresjon
                df = df / df.iloc[0] * 100
                df = df.resample("Y").mean().dropna()

            if df_samlet.empty:
                df_samlet = df
            else:
                df_samlet = df_samlet.join(df, how="outer")

        except Exception as e:
            st.error(f"Feil ved henting av {navn}: {e}")

    df_samlet = df_samlet.dropna(how="all")

    # Tegn hoved-graf for sammenligning
    if not df_samlet.empty:
        df_samlet = df_samlet.round(1)
        st.line_chart(df_samlet)
        df_vis = df_samlet.tail(10).copy()
        df_vis.index = df_vis.index.strftime("%Y")
        st.dataframe(df_vis)

        csv = df_samlet.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("🗕️ Last ned CSV", csv, "indikatorsammenligning.csv", "text/csv")

        os.makedirs("data", exist_ok=True)
        df_samlet.to_csv("data/sammenligning_auto.csv")
        save_to_sqlite(df_samlet, "indikator_sammenligning")
        st.success("📁 Data lagret til CSV og SQLite (data/makrodata.db)")
    else:
        st.warning("Ingen data tilgjengelig for valgt år eller indikator. Prøv et tidligere år eller annen indikator.")

    # --- BNP-regresjon ---
    if bnp_rå is not None and not bnp_rå.empty:
        st.subheader("🔮 BNP-prognose (regresjon, nivå-data)")
        # Ta utgangspunkt i "rå" BNP-data for regresjon
        X = bnp_rå.index.year.values.reshape(-1, 1)
        y = bnp_rå["BNP"].values
        model = LinearRegression().fit(X, y)

        fremtid_år = np.arange(bnp_rå.index.year.max() + 1, bnp_rå.index.year.max() + 6)
        fremtid = pd.DataFrame({"År": fremtid_år})
        fremtid["BNP-prognose"] = model.predict(fremtid[["År"]])
        fremtid["År"] = pd.to_datetime(fremtid["År"], format="%Y")
        fremtid.set_index("År", inplace=True)

        vis = pd.concat([bnp_rå[["BNP"]], fremtid[["BNP-prognose"]]], axis=1)
        st.line_chart(vis)
        st.dataframe(fremtid)
        save_to_sqlite(vis, "bnp_forecast")

