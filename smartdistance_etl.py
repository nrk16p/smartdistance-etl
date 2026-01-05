# ============================================================
# IMPORT
# ============================================================
from pymongo import MongoClient, UpdateOne
import pandas as pd
import numpy as np
import math
import requests
from datetime import datetime, timedelta

import os
from datetime import datetime, timedelta, timezone

# ============================================================
# CONFIG (FROM ENV)
# ============================================================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("❌ MONGO_URI not set")

DB_ANALYTICS = os.getenv("DB_ANALYTICS", "analytics")
COL_SUMMARY  = os.getenv("COL_SUMMARY", "smartdistance")
COL_RAW      = os.getenv("COL_RAW", "raw_smartdistance")

START_DATE = os.getenv("START_DATE")
END_DATE   = os.getenv("END_DATE")

if not START_DATE or not END_DATE:
    raise RuntimeError("❌ START_DATE / END_DATE not set")

PLANT_RADIUS_M = int(os.getenv("PLANT_RADIUS_M", 200))
SITE_RADIUS_M  = int(os.getenv("SITE_RADIUS_M", 200))

OSRM_BASE = os.getenv("OSRM_BASE", "https://router.project-osrm.org")

LOGIC_VERSION = os.getenv("LOGIC_VERSION", "roundtrip_v1")
# ============================================================
# HELPER FUNCTIONS
# ============================================================
def daterange(start_date, end_date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_latlng(coord):
    try:
        lat, lng = coord.split(",")
        return float(lat.strip()), float(lng.strip())
    except:
        return None, None


def osrm_distance_km(lat1, lng1, lat2, lng2, timeout=10):
    url = (
        f"{OSRM_BASE}/route/v1/driving/"
        f"{lng1},{lat1};{lng2},{lat2}"
        "?overview=false"
    )
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()["routes"][0]["distance"] / 1000
    except:
        return None


def to_float_safe(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)
    except:
        return None


# ============================================================
# GEOJSON FEATURE (WITH TIMESTAMPS)
# ============================================================
def make_geojson_feature(linestring_coords, properties=None, timestamps=None):
    props = properties or {}
    if timestamps is not None:
        props["timestamps"] = timestamps

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": linestring_coords,
        },
        "properties": props,
    }

def utcnow():
    return datetime.now(timezone.utc)

# ============================================================
# PROCESS ONE TICKET (FIRST COMPLETE ROUNDTRIP ONLY)
# ============================================================
def process_one_ticket(rmc_row, client, osrm_cache):

    ticket_no = rmc_row.get("TicketNo")
    plate = rmc_row.get("TruckPlateNo")
    ticket_create_at = rmc_row.get("TicketCreateAt")

    if not ticket_no or not plate or not ticket_create_at:
        return None, None

    # ---------- time window ----------
    dt = pd.to_datetime(ticket_create_at, errors="coerce")
    if pd.isna(dt):
        return None, None

    dt = dt.tz_localize(None)
    date_str = dt.strftime("%d/%m/%Y")
    start_time = dt - timedelta(hours=1)
    end_time = dt + timedelta(hours=3)

    # ---------- pull logs ----------
    logs = list(
        client["terminus"]["driving_log"].find(
            {"วันที่": date_str, "ทะเบียนพาหนะ": plate},
            {"_id": 0}
        )
    )
    if not logs:
        return None, None

    df = pd.DataFrame(logs)

    # ---------- parse datetime ----------
    df["datetime"] = pd.to_datetime(
        df["วันที่"].astype(str) + " " + df["เวลา"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    df = df[(df["datetime"] >= start_time) & (df["datetime"] <= end_time)]
    if df.empty:
        return None, None

    df = (
        df[["ลำดับ", "datetime", "พิกัด", "ระยะทาง(กม.)"]]
        .sort_values(["datetime", "ลำดับ"])
        .reset_index(drop=True)
    )

    df[["lat", "lng"]] = df["พิกัด"].apply(lambda x: pd.Series(parse_latlng(x)))
    df = df.dropna(subset=["lat", "lng"])
    if df.empty:
        return None, None

    plant_lat = to_float_safe(rmc_row.get("Latitude"))
    plant_lng = to_float_safe(rmc_row.get("Longitude"))
    site_lat = to_float_safe(rmc_row.get("SiteLat"))
    site_lng = to_float_safe(rmc_row.get("SiteLng"))
    if None in (plant_lat, plant_lng, site_lat, site_lng):
        return None, None

    df["dist_plant"] = df.apply(
        lambda r: haversine_m(plant_lat, plant_lng, r["lat"], r["lng"]), axis=1
    )
    df["dist_site"] = df.apply(
        lambda r: haversine_m(site_lat, site_lng, r["lat"], r["lng"]), axis=1
    )

    df["state"] = np.where(
        df["dist_plant"] <= PLANT_RADIUS_M, "PLANT",
        np.where(df["dist_site"] <= SITE_RADIUS_M, "SITE", "ON_ROUTE")
    )
    df["prev_state"] = df["state"].shift(1)

    def detect_direction(r):
        if r["prev_state"] == "PLANT" and r["state"] == "ON_ROUTE":
            return "PLANT_TO_SITE"
        if r["prev_state"] == "SITE" and r["state"] == "ON_ROUTE":
            return "SITE_TO_PLANT"
        return None

    df["direction"] = df.apply(detect_direction, axis=1)

    # ---------- find first roundtrip ----------
    loop_started = False
    site_reached = False
    start_index = None
    end_index = None

    for i, r in df.iterrows():
        if not loop_started and r["prev_state"] == "PLANT" and r["state"] == "ON_ROUTE":
            loop_started = True
            start_index = max(i - 1, 0)
            continue

        if loop_started and not site_reached and r["prev_state"] == "ON_ROUTE" and r["state"] == "SITE":
            site_reached = True
            continue

        if loop_started and site_reached and r["prev_state"] == "ON_ROUTE" and r["state"] == "PLANT":
            end_index = i
            break

    if start_index is None or end_index is None:
        return None, None

    df_trip = df.iloc[start_index:end_index + 1].copy()
    df_trip["direction"] = df_trip["direction"].ffill()

    # ---------- summary ----------
    df_trip["ระยะทาง(กม.)"] = pd.to_numeric(
        df_trip["ระยะทาง(กม.)"], errors="coerce"
    ).fillna(0)

    summary = (
        df_trip.dropna(subset=["direction"])
        .groupby("direction")
        .agg(
            gps_distance_km=("ระยะทาง(กม.)", "sum"),
            trip_points=("direction", "count"),
        )
        .reset_index()
    )

    result = {}
    for _, r in summary.iterrows():
        if r["direction"] == "PLANT_TO_SITE":
            result["gps_distance_km_p2s"] = float(r["gps_distance_km"])
            result["trip_points_p2s"] = int(r["trip_points"])
            result["rmc_distance_km_p2s"] = to_float_safe(rmc_row.get("PlantToSiteDistance"))
        elif r["direction"] == "SITE_TO_PLANT":
            result["gps_distance_km_s2p"] = float(r["gps_distance_km"])
            result["trip_points_s2p"] = int(r["trip_points"])
            result["rmc_distance_km_s2p"] = to_float_safe(rmc_row.get("SiteToPlantDistance"))

    if not result:
        return None, None

    key = (plant_lat, plant_lng, site_lat, site_lng)
    if key not in osrm_cache:
        osrm_cache[key] = {
            "p2s": osrm_distance_km(plant_lat, plant_lng, site_lat, site_lng),
            "s2p": osrm_distance_km(site_lat, site_lng, plant_lat, plant_lng),
        }

    result["osrm_distance_km_p2s"] = osrm_cache[key]["p2s"]
    result["osrm_distance_km_s2p"] = osrm_cache[key]["s2p"]

    coords = df_trip[["lng", "lat", "datetime"]].values.tolist()
    cleaned, timestamps, prev = [], [], None
    for lng, lat, dtt in coords:
        pt = [float(lng), float(lat)]
        ts = pd.to_datetime(dtt, errors="coerce")
        if prev is None or pt != prev:
            cleaned.append(pt)
            timestamps.append(ts.isoformat() if not pd.isna(ts) else None)
            prev = pt

    raw_doc = {
        "TicketNo": ticket_no,
        "TruckPlateNo": plate,
        "TicketCreateAt": ticket_create_at,
        "PlantCode": rmc_row.get("PlantCode"),
        "SiteCode": rmc_row.get("SiteCode"),
        "plant_lat": plant_lat,
        "plant_lng": plant_lng,
        "site_lat": site_lat,
        "site_lng": site_lng,
        "loop_start_at": df_trip.iloc[0]["datetime"],
        "loop_end_at": df_trip.iloc[-1]["datetime"],
        "point_count": len(cleaned),
        "geojson": make_geojson_feature(
            cleaned,
            {"TicketNo": ticket_no, "plate": plate, "logic_version": "roundtrip_v1"},
            timestamps,
        ),
        "updated_at": datetime.utcnow(),
        "logic_version": "roundtrip_v1",
    }

    result.update({
        "TicketNo": ticket_no,
        "TruckPlateNo": plate,
        "TruckNo": rmc_row.get("TruckNo"),
        "TicketCreateAt": ticket_create_at,
        "PlantCode": rmc_row.get("PlantCode"),
        "SiteCode": rmc_row.get("SiteCode"),
        "loop_completed": True,
        "loop_start_at": df_trip.iloc[0]["datetime"],
        "loop_end_at": df_trip.iloc[-1]["datetime"],
        "logic_version": "roundtrip_v1",
    })

    return pd.DataFrame([result]), raw_doc


# ============================================================
# MAIN (DAY BY DAY)
# ============================================================
def main():

    client = MongoClient(MONGO_URI)

    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    plants = pd.DataFrame(list(client["atms"]["plants"].find({}, {"_id": 0})))

    for run_date in daterange(start, end):

        FILTER_DATE = run_date.strftime("%Y-%m-%d")
        print(f"\n📅 RUN DATE: {FILTER_DATE}")

        job_start = datetime.utcnow()
        processed = summary_written = raw_written = 0
        status = "success"
        error_msg = None

        try:
            rmc = pd.DataFrame(
                list(
                    client["terminus"]["rmcconcretetrip"].find(
                        {"TicketCreateAt": {"$regex": FILTER_DATE}},
                        {"_id": 0}
                    )
                )
            )
            if rmc.empty:
                print("⚠️ no data")
                continue

            rmc = rmc.merge(plants, left_on="PlantCode", right_on="plant_code", how="left")

            rmc["TruckPlateNo"] = (
                rmc["TruckPlateNo"].astype(str)
                .str.replace("สบ.", "", regex=False)
                .str.replace("สบ", "", regex=False)
                .str.replace(" ", "", regex=False)
            )

            smart_col = client[DB_ANALYTICS][COL_SUMMARY]
            raw_col = client[DB_ANALYTICS][COL_RAW]

            osrm_cache = {}
            ops_s, ops_r = [], []

            for _, row in rmc.iterrows():
                processed += 1
                summary_df, raw_doc = process_one_ticket(row, client, osrm_cache)
                if summary_df is None:
                    continue

                ops_s.append(UpdateOne(
                    {"TicketNo": summary_df.iloc[0]["TicketNo"]},
                    {"$set": summary_df.iloc[0].to_dict()},
                    upsert=True
                ))
                ops_r.append(UpdateOne(
                    {"TicketNo": raw_doc["TicketNo"]},
                    {"$set": raw_doc},
                    upsert=True
                ))

            if ops_s:
                res = smart_col.bulk_write(ops_s, ordered=False)
                summary_written = res.upserted_count + res.modified_count
            if ops_r:
                res = raw_col.bulk_write(ops_r, ordered=False)
                raw_written = res.upserted_count + res.modified_count

        except Exception as e:
            status = "failed"
            error_msg = str(e)

        finally:
            client[DB_ANALYTICS]["etl_jobs"].update_one(
                {"_id": f"smartdistance_{FILTER_DATE}"},
                {
                    "$set": {
                        "job": "smartdistance",
                        "run_date": FILTER_DATE,
                        "status": status,
                        "processed": processed,
                        "summary_written": summary_written,
                        "raw_written": raw_written,
                        "error": error_msg,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    "$setOnInsert": {
                        "created_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True
            )
            
            print(f"✅ DONE {FILTER_DATE} processed={processed}")


if __name__ == "__main__":
    main()
