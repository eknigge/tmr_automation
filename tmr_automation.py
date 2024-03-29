import logging
import TollData as td
import numpy as np
import pandas as pd
import os

headers = [('', 3), ('TOTAL', 8), ('TYPE1', 8), ('TYPE2', 8), ('TYPE93', 8),
           ('TYPE99', 8), ('TYPE94', 1), ('TYPE96', 1), ('TYPE97', 6),
           ('TYPE92', 8), ('TYPE90', 8), ('TYPE95', 8), ('TYPE98', 8)]
TRANSMITTAL_TABLE_TRX_TYPE = [('TYPE1_Accept.1', 'TYPE1_Prev Accept.1'),
                              ('TYPE2_Accept.2', 'TYPE2_Prev Accept.2'),
                              ('TYPE90_Accept.7', 'TYPE90_Prev Accept.7'),
                              ('TYPE95_Accept.8', 'TYPE95_Prev Accept.8'),
                              ('TYPE98_Accept.9', 'TYPE98_Prev Accept.9'),
                              ('TYPE99_Accept.4', 'TYPE99_Prev Accept.4'),
                              ('TYPE97_Accept.5', 'TYPE97_Prev Accept.5'),
                              ('TYPE96_Unsent', 0),
                              ('TYPE92_Accept.6', 'TYPE92_Prev Accept.6')]

TRANSMITTAL_TABLE_HEADERS = ['TYPE1', 'TYPE2', 'TYPE90', 'TYPE95', 'TYPE98', 'TYPE99',
                             'TYPE97', 'TYPE96', 'TYPE92']
TRANSMITTAL_TABLE_CHECKS = [TRANSMITTAL_TABLE_HEADERS[0], TRANSMITTAL_TABLE_HEADERS[1],
                            TRANSMITTAL_TABLE_HEADERS[2]]
TRANSMITTAL_FILE_IDENTIFIER = 'Transmittal'
TRANSMITTAL_HEADER_ROW = 6
TRIP_FILE_IDENTIFIER = 'TripTxnDetail'
TRIP_FILE_FILTERS = [0, 71, 73, 96, 98]
TRIP_ZERO_THRESHOLD_HOURS = 2
DIRECTIONS = ['NB', 'SB', 'EB', 'WB']
PASS_TRANSACTION_TYPES = ['HOV', 'AVI']
TRANSACTION_FILE_IDENTIFIER = 'TrxnDetail'
TOLLING_START_HOUR = 5
TOLLING_END_HOUR = 19
OCR_THRESHOLD = 90
OCR_FILE_IDENTIFIER = 'OCR'
OUTPUT_REPORTS = {}
INVALID_TRXN_TYPES = ['BUF', 'UNK', 'SPU']


def expand_transmittal_headers(headers: list, spacer='_') -> list:
    """
    Expand header information ([header title], [quantity]), if blank, do not add spacer. E.g.
    (HEADER1, 2) will become ['HEADER1', 'HEADER1'].
    :return: list of expanded headers, including spacers
    """
    logging.info('Expand Transmittal Headers')
    out = []
    for item in headers:
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
    logging.info(f'Process transmittal file:  {filename}')
    df = pd.read_csv(filename, skiprows=TRANSMITTAL_HEADER_ROW)
    df = df.dropna(subset=['Plaza'])
    new_headers = combine_headers(expand_transmittal_headers(headers), df.columns)
    df.columns = new_headers
    return df


def get_transmittal_column_name(value: str) -> str:
    """
    Return transmittal column name without formatting
    :param value: str
    :return: transaction type in TRANSMITTAL_TABLE_HEADERS
    """
    for name in TRANSMITTAL_TABLE_HEADERS:
        if name in value:
            return name
    return 'UNKNOWN'


def process_transmittal_files():
    """
    Process transmittal files and summarize sent trips into an output table
    """
    logging.info('Start processing all transmittal files')
    transmittal_files = [i for i in os.listdir(os.getcwd()) if TRANSMITTAL_FILE_IDENTIFIER in i]
    df_transmittal = pd.concat([process_transmittal_file(i) for i in transmittal_files])
    df_transmittal.to_csv('temp.csv')
    logging.debug('All transmittal files combined')
    df_transmittal['DATETIME'] = pd.to_datetime(df_transmittal['Trx Date'])
    df_transmittal['DATE'] = df_transmittal['DATETIME'].dt.date
    logging.debug('Date time information added to dataframe')

    transmittal_tables = []
    for i in range(len(TRANSMITTAL_TABLE_TRX_TYPE)):
        accept = TRANSMITTAL_TABLE_TRX_TYPE[i][0]
        previous_accept = TRANSMITTAL_TABLE_TRX_TYPE[i][1]
        logging.debug(f'Processing transaction type: {accept}, {previous_accept}')

        # Convert to numeric
        df_transmittal[accept] = df_transmittal[accept].apply(numeric_conversion)
        try:
            df_transmittal[previous_accept] = df_transmittal[previous_accept].apply(numeric_conversion)
        except KeyError:
            df_transmittal[previous_accept] = 0

        # Summarize and combine
        table_accept = df_transmittal.pivot_table(values=accept, index='DATE', aggfunc=np.sum)
        table_prev_accept = df_transmittal.pivot_table(values=previous_accept, index='DATE', aggfunc=np.sum)
        table_out = table_accept[accept] + table_prev_accept[previous_accept]
        table_out.name = get_transmittal_column_name(accept)

        transmittal_tables.append(table_out)
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
        if values[i] == 0 or values[i] == 0.0 or np.isnan(values[i]):
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
            if values[i - j] == 0 or values[i - j] == 0.0 or np.isnan(values[i - j]):
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
                              f' and on date {df.index.tolist()[value]}')


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


def trip_filter_type_accept(value: str):
    """
    Return whether valid trip transaction type
    :param value: string
    :return: bool
    """
    return value in TRIP_FILE_FILTERS


def create_trip_summary_tables(dataframe: pd.DataFrame):
    """
    Wrapper function to create the trip summary tables
    :param dataframe: Pandas DataFrame
    """
    logging.info('Begin create table summaries')
    df_trips = dataframe

    # Filter trips by valid transaction types
    df_trips['REMOVE_BY_FILTER'] = df_trips['Filtertype'].apply(trip_filter_type_accept)
    df_trips = df_trips[df_trips['REMOVE_BY_FILTER'] == True]

    # Trip counts by hour, day, and trip ID
    trip_table_day_hour = df_trips.pivot_table(values='Trip ID', index=['DATE', 'HOUR'],
                                               columns='TripDefID', aggfunc=np.count_nonzero)
    find_zeros_dataframe(trip_table_day_hour, trip_table_day_hour.columns, TRIP_ZERO_THRESHOLD_HOURS)
    OUTPUT_REPORTS.update({'trip_table_hour': trip_table_day_hour})
    logging.debug('Table for trip ID by day and hour created')

    # Trip count by day and trip ID
    trip_table_day = df_trips.pivot_table(values='Trip ID', index='DATE',
                                          columns='TripDefID', aggfunc=np.count_nonzero)
    OUTPUT_REPORTS.update({'trip_table_day': trip_table_day})
    logging.debug('Table for trip ID by day created')

    # Payment type by hour
    trip_table_type = df_trips.pivot_table(values='Trip ID', index=['DATE', 'HOUR'],
                                           columns='Pmnt Type', aggfunc=np.count_nonzero)
    OUTPUT_REPORTS.update({'trip_pmnt_type': trip_table_type})

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
    df_agency = df_trips[df_trips['St'].apply(is_valid_or_moto_tag) == True]
    trip_table_agency = df_agency.pivot_table(values='Trip ID', index='DATE', columns='AG',
                                              aggfunc=np.count_nonzero)
    OUTPUT_REPORTS.update({'agency_summary': trip_table_agency})
    logging.debug('Table for agency summary created')
    df_agency = None
    logging.debug('Table for agency summary, data removed')

    logging.info('End create table summaries')


def is_valid_or_moto_tag(value: str) -> bool:
    return 'v' in str(value).lower() or 'm' in str(value).lower()


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


def validate_trip_input(value):
    try:
        new_value = int(value)
        return 1
    except ValueError:
        return 0


def is_invalid_transaction_type(value: str):
    """
    Determine whether input transaction is invalid for counting an 
    data statistics
    """
    return str(value.upper()) in INVALID_TRXN_TYPES


def process_transaction_files():
    """
    Process transactions files. This function is used for determining tag penetration
    statistics since the transaction file contains day 24/7, counting the number of
    transaction that become trips, and for axle statistics.
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

    # remove invalid transaction types
    df_transaction['IS_INVALID'] = df_transaction['Trx Typ'].apply(is_invalid_transaction_type)
    df_transaction = df_transaction[df_transaction['IS_INVALID'] != True]

    # Tag penetration statistics
    transaction_table_penetration = df_transaction.pivot_table(values='Trx Typ', index='HOUR',
                                                               columns=['DATE', 'DIRECTION'],
                                                               aggfunc=get_pass_percentage)
    OUTPUT_REPORTS.update({'tag_penetration': transaction_table_penetration})

    # Generate axle report and total tables
    table_names = ['IMG', 'AVI']
    transaction_table_axle = create_axle_total_tables(df_transaction, table_names)

    # Rename columns
    column_names = list(transaction_table_axle.columns)
    n = len(column_names)
    for i in range(n - len(table_names), n):
        column_names[i] = table_names.pop(0)
    transaction_table_axle.columns = column_names

    transaction_table_axle.to_csv('temp.csv')
    OUTPUT_REPORTS.update({'axle_report': transaction_table_axle})

    # Calculate transactions to trips metric
    transactions_to_trip_count(df_transaction)

    logging.info('End processing all transaction files')


def create_axle_total_tables(df_transaction: pd.DataFrame, table_names) -> pd.DataFrame:
    """
    Create axle tables and return combined dataframe
    :param df_transaction: Pandas DataFrame
    :param table_names: list of str values
    :return: Pandas DataFrame
    """
    axle_report_tables = []
    df_transaction['AXLE'] = df_transaction['Fwd'].apply(axle_count_validation)
    df_transaction['TYPE'] = df_transaction['Trx Typ'].apply(transaction_type_axle_report)
    transaction_table_axle = df_transaction.pivot_table(values='Trx ID', index=['DATE', 'HOUR'],
                                                        columns=['DIRECTION', 'TYPE', 'AXLE'],
                                                        aggfunc=np.count_nonzero)

    # Create total tables
    df_transaction_img = df_transaction[df_transaction['TYPE'] == 'IMG']
    transaction_table_img = df_transaction_img.pivot_table(values='Trx ID', index=['DATE', 'HOUR'],
                                                           aggfunc=np.count_nonzero)
    axle_report_tables.append(transaction_table_img)

    df_transaction_avi = df_transaction[df_transaction['TYPE'] == 'AVI']
    transaction_table_avi = df_transaction_avi.pivot_table(values='Trx ID', index=['DATE', 'HOUR'],
                                                           aggfunc=np.count_nonzero)
    axle_report_tables.append(transaction_table_avi)

    directions = df_transaction['DIRECTION'].drop_duplicates().tolist()
    for direction in directions:
        table_names.append(direction)
        df_transaction_dir = df_transaction[df_transaction['DIRECTION'] == direction]
        transaction_table_direction = df_transaction_dir.pivot_table(values='Trx ID',
                                                                     index=['DATE', 'HOUR'],
                                                                     aggfunc=np.count_nonzero)
        axle_report_tables.append(transaction_table_direction)

    # Combine all tables
    for report in axle_report_tables:
        transaction_table_axle = pd.concat([transaction_table_axle, report], axis='columns')

    return transaction_table_axle


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

    count_dict = {}
    n = df_transaction_tolling_hour.shape[0]
    for i in range(n):
        date = df_transaction_tolling_hour['DATE'].iloc[i]
        is_valid_trip = validate_trip_input(df_transaction_tolling_hour['Trip'].iloc[i])
        if date in count_dict:
            count_dict[date][1] += is_valid_trip
        else:
            count_dict[date] = [n, 0]

    for key in count_dict:
        total = count_dict[key][0]
        trips = count_dict[key][1]
        count_dict[key].append(trips / total)

    output = pd.DataFrame(count_dict).T
    output.columns = ['Total_Transactions', 'Total_Trips', 'Percent_Built']

    OUTPUT_REPORTS.update({'trip_building_metric': output})


def get_ocr_confidence(filename: str) -> dict:
    """
    Open and process file to determine OCR confidence. Confidence values set globally.
    :param filename: filename of OCR file
    :return: dict {direction-year-month-date : float (percent ocr passing metric)}
    """
    df = pd.read_csv(filename, skiprows=4)
    df = df[df['CSC'] == 'Y']
    if df.shape[0] == 0:
        return
    direction = get_cardinal_direction(df['Lane'].iloc[0])
    date = pd.to_datetime(df['Trx DateTime'].iloc[0])
    file_date = f'{date.year}-{date.month}-{date.day}'
    ocr_result = ocr_passing(df['OCR Cnfd'])

    output = [file_date, direction, ocr_result]
    logging.debug(f'OCR Confidence Result: {output}')
    return output


def process_ocr_files():
    """
    Wrapper function to process all OCR files and output OCR metric
    """
    logging.info('Start Processing all OCR files')
    ocr_files = [i for i in os.listdir(os.getcwd()) if OCR_FILE_IDENTIFIER in i]
    if len(ocr_files) > 0:
        df_ocr_confidence = []
        for file in ocr_files:
            logging.debug(f'Adding OCR results from {file}')
            ocr_result = get_ocr_confidence(file)
            if ocr_result is not None:
                df_ocr_confidence.append(ocr_result)
        column_names = ['DATE', 'DIRECTION', 'METRIC']
        df = pd.DataFrame(df_ocr_confidence, columns=column_names)
        table = df.pivot_table(values='METRIC', index='DATE',
                               columns='DIRECTION', aggfunc=np.sum)
        OUTPUT_REPORTS.update({'ocr_audit': table})

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


def axle_count_validation(value):
    """
    Function to limit the axle count values from 2 (min) to 6 (max)
    :param value: numerical
    :return: int, range 2-6
    """
    if value < 2:
        return 2
    elif value > 6:
        return 6
    else:
        return value


def transaction_type_axle_report(value: str) -> str:
    """
    Function to set transactions to either IMG or AVI
    :param value: string, transaction type
    :return: string, IMG or AVI
    """
    if 'IMG' in value:
        return value
    else:
        return 'AVI'


def export_error_log():
    """
    Read script log and create separate error log file.
    """
    output = []
    with open('tmr_automation.log', 'r') as f:
        for line in f:
            line_data = f.readline().strip()
            if 'ERROR' in line_data:
                output.append(line_data)

    with open('tmr_errors.log', 'w') as f:
        for line in output:
            f.write(line + '\n')


def numeric_conversion(value: str):
    value = str(value).replace(',', '')
    return int(value)


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
    export_error_log()


if __name__ == '__main__':
    main()
