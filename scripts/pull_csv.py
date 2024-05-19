from influxdb_client import InfluxDBClient
import pathlib
import time
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv('secret.env')

my_token = os.getenv("INFLUXDB_TOKEN")
my_org = "IU"
client = InfluxDBClient(url="http://e-002.echo.ise.luddy.indiana.edu:8086/", token=my_token, org=my_org, debug=False)
  


def get_measurements(bucket: str, org: str) -> list:
    """
    This function fetches all the table names for given bucket in your organization.

    Args:
        bucket: The name of the bucket according to InfluxDB
        org: The name of the organization on InfluxDB

    Returns:
        the name of the tables in the bucket (manufacturing center)
    """
    query = f"""
    import \"influxdata/influxdb/schema\"
    schema.measurements(bucket: \"{bucket}\")
    """
    query_api = client.query_api()
    tables = query_api.query(query=query, org=org)    
    measurements = [row.values["_value"] for table in tables for row in table]
    return measurements


def fetch_influx(client_conn: InfluxDBClient, query: str) -> pd.DataFrame:
    """
    This function fetches a dataframe from InfluxDB for a given query.

    Args:
        client_conn: The InfluxDBClient object
        query: The query string for InfluxDB
    """
    df = client_conn.query_api().query_data_frame(org=my_org, query=query)
    return df


def df_read(bucket: str, feature: str, start=None, end=None, step="s") -> pd.DataFrame:
    """
    This creates flux query for given timestamp, fetches it from the server and then returns output as a dataframe.

    Args:
        bucket: The name of the bucket according to InfluxDB
        feature: The name of the feature to be queried
        start: The start time of the query
        end: The end time of the query
        step: The step size for the query (s,m,h,d)
    
    Returns:
        A pandas dataframe with the queried data
    """
    flux_query = f"""
    from(bucket: "{bucket}")
    |> range(start:{start}, stop: {end})
    |> filter(fn: (r) => r["_measurement"] == "{feature}")
    |> filter(fn: (r) => r["_field"] == "val")
    |> aggregateWindow(every: 1{step}, fn: mean, createEmpty: true) 
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    print(flux_query)
    df = fetch_influx(client, flux_query)
    return df


def main(bucket: str, output_folder: str, starts: list, ends: list, labels: list, granularity = "s"):
    """
    Creates a csv file for each true power variable in the bucket (short name) at the time ranges specified in lists.
    The lists starts, ends, and labels should be the same length, because this function will iterate over them together
    to create the csv files.

    Args:
        bucket: The name of the bucket according to InfluxDB
        output_folder: The name of the folder where the csv files will be stored (all files are stored under data/ 
            so no need to include that in the output_folder name)
        starts: A list of start times for the queries
        ends: A list of end times for the queries
        labels: A list of labels for the csv files
        granularity: The time granularity for the query (s,m,h,d)
    """
    bucket_dict = {
    'OptoMMP': {
        'bucket_name': 'energy_OptoMMP/Modules/Channels',
        'feature_roots': ['A_TruePower_W', 'B_TruePower_W', 'C_TruePower_W']
        },
    'amatrol': {
        'bucket_name': 'amatrol',
        'feature_roots': ['TruePowerWatts/Value']
        }
    }
    assert len(starts) == len(ends) == len(labels), "The lists starts, ends, and labels should be the same length."
    features = get_measurements(bucket=bucket_dict[bucket]['bucket_name'], org=my_org)
    tp_features = []
    for feature in features:
        for feature_root in bucket_dict[bucket]['feature_roots']:
            if feature_root in feature:
                tp_features.append(feature)

    for tp_feature in tp_features:
        for label, start, end in zip(labels, starts, ends):
            start_time = time.time()  
            df = df_read(bucket_dict[bucket]['bucket_name'], feature=tp_feature, start=start, end=end)
            if df.empty:
                continue
            p = pathlib.Path(f"data/{output_folder}/{bucket}-{label}")
            p.mkdir(parents=True, exist_ok=True)
            tp_feature_name = tp_feature.replace("/", "_")
            df.to_csv(p / f"{tp_feature_name}.csv", index=False, columns=["_time", "val"])
            print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    starts = [
            '2024-02-27T00:00:00Z',
        ]
    ends = [
            '2024-03-25T23:00:00Z',
        ]
    labels = [
            'Mar24',
        ]
    # output_folder = "min_avg"
    # main('amatrol', starts, ends, labels, granularity="m")