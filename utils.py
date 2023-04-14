import datetime
import pytz
def ast_to_est(ast):
    # Where ast takes the form: 2022-01-01 05:58:52 Arabian Standard Time
    date, time = ast.split(' ')[:2]

    year, month, day = list(map(int, date.split('-')))
    hour, min, sec = list(map(int, time.split(':')))
    
    parsed_ast = datetime.datetime(year, month, day, hour, min, sec, tzinfo=pytz.timezone('Asia/Riyadh'))

    return parsed_ast.astimezone(pytz.timezone('America/New_York'))

