import logging
import TollData as td
import numpy as np
import pandas as pd
import os

TRANSMITTAL_HEADERS = [('', 3), ('TOTAL', 8), ('TYPE1', 8), ('TYPE2', 8), ('TYPE93', 8),
                       ('TYPE99', 8), ('TYPE94', 1), ('TYPE96', 1), ('TYPE97', 6),
                       ('TYPE92', 8), ('TYPE90', 8), ('TYPE95', 8), ('TYPE98', 8)]
TRANSMITTAL_TABLE_TRX_TYPE = ['TYPE1_Total.1', 'TYPE2_Total.2', 'TYPE98_Total.9',
                              'TYPE95_Total.8', 'TYPE90_Total.7', 'TYPE99_Total.4']
TRANSMITTAL_FILE_IDENTIFIER = 'Transmittal'
TRIP_FILE_IDENTIFIER = 'TripTxnDetail'
DIRECTIONS = ['NB', 'SB', 'EB', 'WB']
PASS_TRANSACTION_TYPES = ['HOV', 'AVI']
TRANSACTION_FILE_IDENTIFIER = 'TrxnDetail'
TOLLING_START_HOUR = 5
TOLLING_END_HOUR = 19
OCR_THRESHOLD = 90

"""
TODO
    Process OCR files for OCR metric
    Add in error checking to trip summary, weekend, weekday, and add exemption dates
    Error checking by hour and by daily count
"""


def expand_transmittal_headers() -> list:
    out = []
    spacer = '_'
    for item in TRANSMITTAL_HEADERS:
        for i in range(item[1]):
            if item[0] == '':
                out.append(item[0])
            else:
                out.append(item[0] + spacer)
    return out


def combine_headers(list1: list, list2: list) -> list:
    if len(list1) != len(list2):
        raise ValueError(f'List lengths {len(list1)}:{len(list2)} do not match')
    out = []
    for i in range(len(list1)):
        out.append(list1[i] + list2[i])
    return out


def process_transmittal_file(filename):
    df = pd.read_csv(filename, skiprows=6)
    df = df.dropna(subset=['Plaza'])
    new_headers = combine_headers(expand_transmittal_headers(), df.columns)
    df.columns = new_headers
    return df


def process_transmittal_files():
    transmittal_files = [i for i in os.listdir(os.getcwd()) if TRANSMITTAL_FILE_IDENTIFIER in i]
    df_transmittal = pd.concat([process_transmittal_file(i) for i in transmittal_files])
    df_transmittal['DATETIME'] = pd.to_datetime(df_transmittal['Trx Date'])
    df_transmittal['DATE'] = df_transmittal['DATETIME'].dt.date

    transmittal_tables = []
    for i in TRANSMITTAL_TABLE_TRX_TYPE:
        table = df_transmittal.pivot_table(values=i, index='DATE', aggfunc=np.count_nonzero)
        transmittal_tables.append(table)
    tables = pd.concat(transmittal_tables, axis=1)
    tables.to_csv('temp_transmittal.csv')


def process_trip_file(filename: str) -> pd.DataFrame:
    df = td.TripFile(filename).get_df()
    df = df.drop_duplicates(subset=['Trip ID'])
    return df


def get_cardinal_direction(value: str) -> str:
    for direction in DIRECTIONS:
        if direction in value:
            return direction
    return 'NA'


def process_trip_files():
    trip_files = [i for i in os.listdir(os.getcwd()) if TRIP_FILE_IDENTIFIER in i]
    df_trips = pd.concat([process_trip_file(i) for i in trip_files])
    df_trips['DATETIME'] = pd.to_datetime(df_trips['Entry Time'])
    df_trips['DATE'] = df_trips['DATETIME'].dt.date
    df_trips['HOUR'] = df_trips['DATETIME'].dt.hour
    df_trips['DIRECTION'] = df_trips['Plaza'].apply(get_cardinal_direction)

    # Trip counts by hour, day, and trip ID
    trip_table_day_hour = df_trips.pivot_table(values='Trip ID', index=['DATE', 'HOUR'],
                                               columns='TripDefID', aggfunc=np.count_nonzero)
    # Trip count by day and trip ID
    trip_table_day = df_trips.pivot_table(values='Trip ID', index='DATE',
                                          columns='TripDefID', aggfunc=np.count_nonzero)

    # Daily revenue
    trip_table_revenue = df_trips.pivot_table(values='Fare', index='DATE', columns='DIRECTION',
                                              aggfunc=np.sum)

    # Highest fare
    trip_table_highest_fare = df_trips.pivot_table(values='Fare', index='DATE', columns='DIRECTION',
                                                   aggfunc=np.max)

    # Agency Summary
    trip_table_agency = df_trips.pivot_table(values='Trip ID', index='DATE', columns='AG',
                                             aggfunc=np.count_nonzero)

    # Trip count by day
    trip_table_day = df_trips.pivot_table(values='Trip ID', index='DATE', aggfunc=np.count_nonzero)

    # Remove trips from memory
    df_trips = []


def ocr_passing(values: list) -> float:
    not_passing = 0
    for value in values:
        if value < OCR_THRESHOLD:
            not_passing += 1
    return 1 - (not_passing / len(values))


def get_pass_percentage(values: list):
    total_count = len(values)
    pass_count = 0
    for value in values:
        if value in PASS_TRANSACTION_TYPES:
            pass_count += 1
    return pass_count / total_count


def process_transaction_file(filename: str) -> pd.DataFrame:
    df = td.TransactionFile(filename).get_df()
    df = df.dropna(subset=['Lane'])
    return df


def trip_id_percentage(values: list):
    total_count = len(values)
    trip_count = 0

    for value in values:
        if (isinstance(value, int) or isinstance(value, float)) and value > 0:
            trip_count += 1

    return trip_count / total_count


def process_transaction_files():
    transaction_files = [i for i in os.listdir(os.getcwd()) if TRANSACTION_FILE_IDENTIFIER in i]
    df_transaction = pd.concat([process_transaction_file(i) for i in transaction_files])
    df_transaction['DATETIME'] = pd.to_datetime(df_transaction['Trx DateTime'])
    df_transaction['DATE'] = df_transaction['DATETIME'].dt.date
    df_transaction['HOUR'] = df_transaction['DATETIME'].dt.hour

    # Tag penetration statistics
    transaction_table_payment = df_transaction.pivot_table(values='Trx Typ', index='HOUR',
                                                           columns='DATE', aggfunc=get_pass_percentage)

    # Count transactions to trips
    df_transaction_tolling_hour = df_transaction[(df_transaction['HOUR'] > TOLLING_START_HOUR) &
                                                 (df_transaction['HOUR'] < TOLLING_END_HOUR)]

    # Merge trip files
    trip_files = [i for i in os.listdir(os.getcwd()) if TRIP_FILE_IDENTIFIER in i]
    df_trip = pd.concat([td.TripFile(i).get_df() for i in trip_files])
    df_trip['Trx ID'] = pd.to_numeric(df_trip['Trx ID'])
    df_transaction_tolling_hour['Trx ID'] = pd.to_numeric(df_transaction_tolling_hour['Trx ID'])
    df_transaction_tolling_hour = df_transaction_tolling_hour.merge(df_trip, on='Trx ID')

    transaction_table_trip_building = df_transaction_tolling_hour.pivot_table(values='Trip ID',
                                                                              index='DATE',
                                                                              aggfunc=trip_id_percentage)


def main():
    process_transmittal_files()
    process_trip_files()
    process_transaction_files()


if __name__ == '__main__':
    main()
