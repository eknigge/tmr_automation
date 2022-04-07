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
TRANSMITTAL_TABLE_CHECKS = [TRANSMITTAL_TABLE_TRX_TYPE[0], TRANSMITTAL_TABLE_TRX_TYPE[1],
                            TRANSMITTAL_TABLE_TRX_TYPE[4]]
TRANSMITTAL_FILE_IDENTIFIER = 'Transmittal'
TRANSMITTAL_HEADER_ROW = 6
TRIP_FILE_IDENTIFIER = 'TripTxnDetail'
TRIP_ZERO_THRESHOLD_HOURS = 2
DIRECTIONS = ['NB', 'SB', 'EB', 'WB']
PASS_TRANSACTION_TYPES = ['HOV', 'AVI']
TRANSACTION_FILE_IDENTIFIER = 'TrxnDetail'
TOLLING_START_HOUR = 5
TOLLING_END_HOUR = 19
OCR_THRESHOLD = 90
OCR_FILE_IDENTIFIER = 'OCR'
OUTPUT_REPORTS = {}


def expand_transmittal_headers() -> list:
    """
    Expand header information, if blank, do not add spacer
    :return: list of expanded headers, includes spacers
    """
    logging.info('Expand Transmittal Headers')
    out = []
    spacer = '_'
    for item in TRANSMITTAL_HEADERS:
        for i in range(item[1]):
            if item[0] == '':
                out.append(item[0])
            else:
                out.append(item[0] + spacer)
    return out


def combine_headers(left_list: list, right_list: list) -> list:
    """
    Combine headers from the left and right lists
    :return: list of combined values
    """
    logging.info('Combine headers')
    if len(left_list) != len(right_list):
        error_message = f'List lengths {len(left_list)}:{len(right_list)} do not match'
        logging.error(error_message)
        raise ValueError(error_message)
    out = []
    for i in range(len(left_list)):
        out.append(left_list[i] + right_list[i])
    return out


def process_transmittal_file(filename: str) -> pd.DataFrame:
    """
    Process a single transmittal file
    :param filename: filename of transmittal file
    :return: Pandas DataFrame
    """
    logging.info(f'Process transmitatl file:  {filename}')
    df = pd.read_csv(filename, skiprows=TRANSMITTAL_HEADER_ROW)
    df = df.dropna(subset=['Plaza'])
    new_headers = combine_headers(expand_transmittal_headers(), df.columns)
    df.columns = new_headers
    return df


def process_transmittal_files():
    """
    Process transmittal files and summarize sent trips into an output table
    """
    logging.info('Start processing all transmittal files')
    transmittal_files = [i for i in os.listdir(os.getcwd()) if TRANSMITTAL_FILE_IDENTIFIER in i]
    df_transmittal = pd.concat([process_transmittal_file(i) for i in transmittal_files])
    logging.debug('All transmittal files combined')
    df_transmittal['DATETIME'] = pd.to_datetime(df_transmittal['Trx Date'])
    df_transmittal['DATE'] = df_transmittal['DATETIME'].dt.date
    logging.debug('Date time information added to dataframe')

    transmittal_tables = []
    for i in TRANSMITTAL_TABLE_TRX_TYPE:
        logging.debug(f'Processing transaction type: {i}')
        table = df_transmittal.pivot_table(values=i, index='DATE', aggfunc=np.count_nonzero)
        transmittal_tables.append(table)
    transmittal_summary_table = pd.concat(transmittal_tables, axis=1)
    OUTPUT_REPORTS.update({'transmittal_summary': transmittal_summary_table})

    # Run transmittal table checks
    find_zeros_dataframe(transmittal_summary_table, TRANSMITTAL_TABLE_CHECKS)

    logging.info('End processing all transmittal files')


def find_single_zero_list(values: list) -> list:
    """
    Find and return index where a value of 0 is found
    :param values: list of integer or float values
    :return: list
    """
    out = []
    for i in range(len(values)):
        if values[i] == 0 or values[i] == 0.0:
            out.append(i)
    return out


def found_multiple_zeros(values: list) -> bool:
    """
    Determine whether there are multiple zeros in a list, utilizes the global
    TRIP_ZERO_THRESHOLD_HOURS variable to set allowed number of consecutive zeros
    :param values: list of values
    :return: bool
    """
    out = False
    n = TRIP_ZERO_THRESHOLD_HOURS
    for i in range(len(values)):
        consecutive_zeros = []
        for j in range(1, n + 1):
            if values[i - j] == 0 or values[i - j] == 0.0:
                consecutive_zeros.append(1)
        if sum(consecutive_zeros) == TRIP_ZERO_THRESHOLD_HOURS:
            return True


def find_zeros_dataframe(df: pd.DataFrame, columns: list, number_zeros=1):
    logging.info('Checking for zeros in columns')
    multiple_zeros = False
    index_values = []

    for column in columns:
        if number_zeros == 1:
            index_values = find_single_zero_list(df[column].tolist())
        else:
            multiple_zeros = found_multiple_zeros(df[column].tolist())

        # For multi-zero search
        if multiple_zeros:
            logging.error(f'Multiple zero values found for {column}')
        # For single search search
        if len(index_values) > 0:
            for value in index_values:
                logging.error(f'Zeros found for {column}'
                              f' and on date {df[column].iloc[value]}')


def process_trip_file(filename: str) -> pd.DataFrame:
    """
    Process a single trip file
    :param filename: filename of trip file
    :return: Pandas DataFrame
    """
    logging.info(f'Processing Trip File: {filename}')
    df = td.TripFile(filename).get_df()
    df = df.drop_duplicates(subset=['Trip ID'])
    return df


def get_cardinal_direction(value: str) -> str:
    """
    Determine cardinal direction, if not application return NA
    :param value: value to be search for direction information
    :return: str, cardinal direction
    """
    for direction in DIRECTIONS:
        if direction in value:
            return direction
    return 'NA'


def create_trip_summary_tables(dataframe: pd.DataFrame):
    logging.info('Begin create table summaries')
    df_trips = dataframe

    # Trip counts by hour, day, and trip ID
    trip_table_day_hour = df_trips.pivot_table(values='Trip ID', index=['DATE', 'HOUR'],
                                               columns='TripDefID', aggfunc=np.count_nonzero)
    OUTPUT_REPORTS.update({'trip_table_hour': trip_table_day_hour})
    logging.debug('Table for trip ID by day and hour created')

    # Trip count by day and trip ID
    trip_table_day = df_trips.pivot_table(values='Trip ID', index='DATE',
                                          columns='TripDefID', aggfunc=np.count_nonzero)
    OUTPUT_REPORTS.update({'trip_table_day': trip_table_day})
    logging.debug('Table for trip ID by day created')

    # Daily revenue
    trip_table_revenue = df_trips.pivot_table(values='Fare', index='DATE', columns='DIRECTION',
                                              aggfunc=np.sum)
    OUTPUT_REPORTS.update({'daily_revenue': trip_table_revenue})
    logging.debug('Table for daily revenue created')

    # Highest fare
    trip_table_highest_fare = df_trips.pivot_table(values='Fare', index='DATE', columns='DIRECTION',
                                                   aggfunc=np.max)
    OUTPUT_REPORTS.update({'highest_fare': trip_table_highest_fare})
    logging.debug('Table for highest fare created')

    # Agency Summary
    trip_table_agency = df_trips.pivot_table(values='Trip ID', index='DATE', columns='AG',
                                             aggfunc=np.count_nonzero)
    OUTPUT_REPORTS.update({'agency_summary': trip_table_agency})
    logging.debug('Table for agency summary created')

    logging.info('End create table summaries')


def process_trip_files():
    """
    Process all trip files and output summary tables
    """
    logging.info('Start process all trip files')
    # Import and combine trip files
    logging.debug('Open and combine all trip files')
    trip_files = [i for i in os.listdir(os.getcwd()) if TRIP_FILE_IDENTIFIER in i]
    df_trips = pd.concat([process_trip_file(i) for i in trip_files])

    # Add datetime information to trip files
    logging.debug('Add datetime information to trip dataframe')
    df_trips['DATETIME'] = pd.to_datetime(df_trips['Entry Time'])
    df_trips['DATE'] = df_trips['DATETIME'].dt.date
    df_trips['HOUR'] = df_trips['DATETIME'].dt.hour
    df_trips['DIRECTION'] = df_trips['Plaza'].apply(get_cardinal_direction)

    # Create summary tables
    logging.debug('Create summary tables')
    create_trip_summary_tables(df_trips)

    logging.info('End process all trip files')


def ocr_passing(values: list) -> float:
    """
    Function to determine whether OCR value series passes the OCR threshold
    :param values: list or list-like
    :return: float, percentage passing OCR threshold
    """
    not_passing = 0
    for value in values:
        try:
            value = int(value.replace('%', ''))
        except ValueError:
            value = 0
        if value < OCR_THRESHOLD:
            not_passing += 1
    output = 1 - (not_passing / len(values))
    logging.debug(f'OCR passing result: {output}')
    return output


def get_pass_percentage(values: list):
    """
   Function to evaluate the number pass-based transactions versus the
   number of transactions based on other status such as image.
    :param values: list or list-like
    :return: float, percentage that are pass-based
    """
    total_count = len(values)
    pass_count = 0
    for value in values:
        if value in PASS_TRANSACTION_TYPES:
            pass_count += 1
    output = pass_count / total_count
    logging.debug(f'Pass percentage result: {output}')
    return output


def process_transaction_file(filename: str) -> pd.DataFrame:
    """
    Function to import transaction files. Utilizes Toll Data library and removes
    metadata rows
    :param filename: filename to import
    :return: Pandas DataFrame
    """
    logging.info(f'Processing Transaction File: {filename}')
    df = td.TransactionFile(filename).get_df()
    df = df.dropna(subset=['Lane'])
    return df


def trip_id_percentage(values: list) -> float:
    """
    Function to determine the percentage of values that have a valid trips ID.
    Excludes values that are not an int or a float
    :param values: list or list-like
    :return: float, percentage of valid trips IDs
    """
    total_count = len(values)
    trip_count = 0

    for value in values:
        if (isinstance(value, int) or isinstance(value, float)) and value > 0:
            trip_count += 1

    output = trip_count / total_count
    logging.debug(f'Transactios to Trips Metric: {output}')
    return output


def process_transaction_files():
    """
    Process transactions files. This function is used for determining tag penetration
    statistics since the transaction file contains day 24/7, and for counting the number of
    transaction that become trips.
    """
    logging.info('Start processing all transaction files')
    transaction_files = [i for i in os.listdir(os.getcwd()) if TRANSACTION_FILE_IDENTIFIER in i]
    df_transaction = pd.concat([process_transaction_file(i) for i in transaction_files])
    df_transaction['DIRECTION'] = df_transaction['CSC Lane'].apply(get_cardinal_direction)

    # Add datetime information
    logging.debug('Add datetime information to transaction file dataframe')
    df_transaction['DATETIME'] = pd.to_datetime(df_transaction['Trx DateTime'])
    df_transaction['DATE'] = df_transaction['DATETIME'].dt.date
    df_transaction['HOUR'] = df_transaction['DATETIME'].dt.hour

    logging.info('Calculate transaction file metrics')

    # Tag penetration statistics
    transaction_table_penetration = df_transaction.pivot_table(values='Trx Typ', index='HOUR',
                                                               columns=['DATE', 'DIRECTION'],
                                                               aggfunc=get_pass_percentage)
    OUTPUT_REPORTS.update({'tag_penetration': transaction_table_penetration})

    # Calculate transactions to trips metric
    transactions_to_trip_count(df_transaction)

    logging.info('End processing all transaction files')


def transactions_to_trip_count(dataframe: pd.DataFrame):
    """
    Process input dataframe and open trip files to determine the transactions to trips metric.
    :param dataframe: Pandas DataFrame
    """
    logging.info('Begin process to count transactions to trips')
    df_transaction = dataframe

    # Calculate transactions to trips
    logging.debug(f'Shape of dataframe before removal of non-tolling hour '
                  f'transactions: {df_transaction.shape}')
    df_transaction_tolling_hour = df_transaction[(df_transaction['HOUR'] > TOLLING_START_HOUR) &
                                                 (df_transaction['HOUR'] < TOLLING_END_HOUR)]
    logging.debug(f'Shape of dataframe after removal: {df_transaction_tolling_hour.shape}')

    logging.debug(f'Open trip files and combine')
    trip_files = [i for i in os.listdir(os.getcwd()) if TRIP_FILE_IDENTIFIER in i]
    df_trip = pd.concat([td.TripFile(i).get_df() for i in trip_files])

    # Convert transaction ID fields to numeric
    logging.debug('Convert transaction ID fields to numeric')
    df_trip['Trx ID'] = pd.to_numeric(df_trip['Trx ID'])
    df_transaction_tolling_hour['Trx ID'] = pd.to_numeric(df_transaction_tolling_hour['Trx ID'])

    logging.debug('Merge transaction and trip dataframes')
    df_transaction_tolling_hour = df_transaction_tolling_hour.merge(df_trip, on='Trx ID')

    logging.debug('Output summary table of combined dataframe')
    transaction_table_trip_building = df_transaction_tolling_hour.pivot_table(values='Trip ID',
                                                                              index='DATE',
                                                                              aggfunc=trip_id_percentage)
    OUTPUT_REPORTS.update({'trip_building_metric': transaction_table_trip_building})


def get_ocr_confidence(filename: str) -> dict:
    """
    Open and process file to determine OCR confidence. Confidence values set globally.
    :param filename: filename of OCR file
    :return: dict {direction-year-month-date : float (percent ocr passing metric)}
    """
    df = pd.read_csv(filename, skiprows=4)
    df = df[df['CSC'] == 'Y']
    direction = get_cardinal_direction(df['Lane'].iloc[0])
    date = pd.to_datetime(df['Trx DateTime'].iloc[0])
    file_date = f'{direction}-{date.year}-{date.month}-{date.day}'

    output = {file_date: ocr_passing(df['OCR Cnfd'])}
    logging.debug(f'OCR Confidence Result: {output}')
    return output


def process_ocr_files():
    """
    Wrapper function to process all OCR files and output OCR metric
    """
    logging.info('Start Processing all OCR files')
    ocr_files = [i for i in os.listdir(os.getcwd()) if OCR_FILE_IDENTIFIER in i]
    df_ocr_confidence = {}
    for file in ocr_files:
        logging.debug(f'Adding OCR results from {file}')
        df_ocr_confidence.update(get_ocr_confidence(file))
    df = pd.Series(df_ocr_confidence)

    logging.info('End Processing all OCR files')


def configure_logging():
    """
    Function to enable logging for script
    """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', filename='tmr_automation.log',
                        level=logging.DEBUG)


def export_summary_tables():
    """
    Export all output reports
    """
    logging.info('Start report Export')
    with pd.ExcelWriter('reports_combined.xlsx') as writer:
        for key in OUTPUT_REPORTS:
            OUTPUT_REPORTS[key].to_excel(writer, sheet_name=key)
    logging.info('End report Export')


def main():
    stars = 40 * '*'
    print(f'{stars}\nEnable Logging\n{stars}')
    configure_logging()
    print(f'{stars}\nProcess Transmittal Files\n{stars}')
    process_transmittal_files()
    print(f'{stars}\nProcess Trip Files\n{stars}')
    process_trip_files()
    print(f'{stars}\nProcess Transaction files\n{stars}')
    process_transaction_files()
    print(f'{stars}\nProcess OCR files\n{stars}')
    process_ocr_files()
    print(f'{stars}\nExport Summary Tables\n{stars}')
    export_summary_tables()


if __name__ == '__main__':
    main()
