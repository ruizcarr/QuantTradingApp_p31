import requests
import pandas as pd
import io
import matplotlib.pyplot as plt

def get_euribor_1y_daily():
    """
    Retrieves EURIBOR 1-year data from the ECB API, processes it,
    and returns a Pandas DataFrame with daily reindexing and forward fill.

    Returns:
        pandas.DataFrame: DataFrame with 'date' as index and 'Euribor' as values,
                          or None if an error occurs.
    """
    dataflow_id = "FM/M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA"
    api_url = f"https://data-api.ecb.europa.eu/service/data/{dataflow_id}?format=csvdata"

    try:
        response = requests.get(api_url)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "Euribor"}, inplace=True)

        # Convert 'Euribor' to float
        df["Euribor"] = pd.to_numeric(df["Euribor"], errors='coerce')/100  # coerce will turn non number values into NaN.

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        #df = df.reindex(daily_index, method="ffill")
        df = df.reindex(daily_index)
        df['Euribor'] = df['Euribor'].interpolate(method='linear')

        #Add Cash Values applying to starting 1.000 USD
        #df['cash_values']=1000*(1+df['Euribor']/255).cumprod()
        #df['dayly_Euribor']=df['cash_values'].pct_change() #*255

        df.to_csv("datasets/EURIBOR.csv")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        if response is not None:
            print(response.text)
        return None  # Return None to indicate an error
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return None # Return None to indicate an error

if __name__ == "__main__": #Example of how to use the function.
    euribor_df = get_euribor_1y_daily()
    if euribor_df is not None:
        print(euribor_df)
        euribor_df.plot()

    else:
        print("Failed to retrieve EURIBOR data.")

plt.show()