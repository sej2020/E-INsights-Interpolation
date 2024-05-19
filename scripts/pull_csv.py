from influxdb_client import InfluxDBClient
import pathlib
import time
from dotenv import load_dotenv
import os
load_dotenv('secret.env')


def get_measurements(bucket, org):
    """
    This function fetches all the tables for given bucket
    """
    
    query = f"""
    import \"influxdata/influxdb/schema\"

    schema.measurements(bucket: \"{bucket}\")
    """

    query_api = client.query_api()
    tables = query_api.query(query=query, org=org)

    
    measurements = [row.values["_value"] for table in tables for row in table]
    
    return measurements


def fetch_influx(client_conn, query):
    """
    This function fetches a dataframe for given flux query
    """

    df = client_conn.query_api().query_data_frame(org=my_org, query=query)
    print(df)
    return df


def df_read(bucket, feature, start=None, end=None):
    """
    This creates flux query for given timestamp and then returns output as a dataframe.
    If both delta and start and end are provided, then delta will be ignored. In the default
    setting, delta is set to 1 hour, and the step size is set to 1 minute.
    """
    flux_query = f"""
    from(bucket: "{bucket}")
    |> range(start:{start}, stop: {end})
    |> filter(fn: (r) => r["_measurement"] == "{feature}")
    |> filter(fn: (r) => r["_field"] == "val")
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: true) 
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    print(flux_query)
    df = fetch_influx(client, flux_query)
    return df


def main(bucket: str):
    """
    
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

    features = get_measurements(bucket=bucket_dict[bucket]['bucket_name'], org=my_org)
    tp_features = []
    for feature in features:
        for feature_root in bucket_dict[bucket]['feature_roots']:
            if feature_root in feature:
                tp_features.append(feature)

    tp_features
    for idx, tp in enumerate(tp_features):
        print(idx, tp)

    
    starts = [
        '2024-02-27T00:00:00Z',
    ]
    ends = [
        '2024-03-25T23:00:00Z',
    ]
    week_labels = [
        'Mar24',
    ]
    for tp_feature in tp_features:
        for label, start, end in zip(week_labels, starts, ends):
            start_time = time.time()  
            df = df_read(bucket_dict[bucket]['bucket_name'], feature=tp_feature, start=start, end=end)
            if df.empty:
                continue
            p = pathlib.Path(f"data/min_av/{bucket}-{label}")
            p.mkdir(parents=True, exist_ok=True)
            tp_feature_name = tp_feature.replace("/", "_")
            df.to_csv(p / f"{tp_feature_name}.csv", index=False, columns=["_time", "val"])
            print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    my_token = os.getenv("INFLUXDB_TOKEN")
    my_org = "IU"
    client = InfluxDBClient(url="http://e-002.echo.ise.luddy.indiana.edu:8086/", token=my_token, org=my_org, debug=False)

  
    # main('OptoMMP')
    main('amatrol')

# from(bucket: "amatrol")
#   |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
#   |> filter(fn: (r) => r["_measurement"] == "Power_Meters/HVAC/RoofTopUnit1_old/PhA-TruePowerWatts/Value")
#   |> filter(fn: (r) => r["_field"] == "val")
#   |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
#   |> yield(name: "mean")