import os
import sys
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

from src.exception.exception import CVDException
from src.logger.logging import logging

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class CVDExtractData:

    def csv_to_json_converter(self, file_path: str):
        try:
            data = pd.read_csv(file_path, compression="zip")
            records = data.to_dict(orient="records")
            return records
        except Exception as e:
            raise CVDException(e)

    def insert_data_mongodb(self, records: list, database: str, collection: str):
        try:
            mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=ca
            )

            db = mongo_client[database]
            col = db[collection]

            col.insert_many(records)
            return len(records)

        except Exception as e:
            raise CVDException(e)


if __name__ == "__main__":
    FILE_PATH = "data/cardio_train.csv.zip"
    DATABASE = "CVD"
    COLLECTION = "CVD_Data"

    cvd = CVDExtractData()
    records = cvd.csv_to_json_converter(FILE_PATH)
    no_of_records = cvd.insert_data_mongodb(records, DATABASE, COLLECTION)

    print(f"Inserted {no_of_records} records into MongoDB")
