import os

# Paths
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
IMAGE_PATH = os.path.join(BASE_PATH, 'images')
RESULTS_PATH = os.path.join(BASE_PATH, 'results')
MODELS_PATH = os.path.join(BASE_PATH, 'best_models')
MODEL_INFO = os.path.join(MODELS_PATH, 'best_models_info.pkl')
TUNING_PATH = os.path.join(BASE_PATH, 'Tuning')

# Variable Names
inflation = 'CPIAUCSL'
treasury = 'GS5'
unemployment = 'UNRATE'
wage = 'RPI'
TARGET_VARIABLES = [inflation, treasury, unemployment, wage]
VARIABLES_MAP = {inflation: 'Inflation', treasury: 'Treasury',
                 wage: 'Wage', unemployment: 'Unemployment'}
PERIODS_MAP = {'During COVID': 2019, 'After GFC': 2017,
               'During GFC': 2010, 'Before GFC': 2006}
REVERSE_PERIOD_MAP = {PERIODS_MAP[k]: k for k in PERIODS_MAP}
OUTPUT_STEPS = [1, 3, 6, 12, 24]
NUMBER_OF_PCAS = [0, 5, 20, 60, 120]
LOOK_BACK_YEARS = [2, 1, 0.25]

# Model Parameters
FREQUENCY = 'md'
TEST_YEARS = 2
VAL_YEARS = 4
DATA_PATH = f'Data/fred{FREQUENCY}.csv'
TRANSFORM_PATH = f'Data/fred{FREQUENCY}_description.csv'
NA_CUTOFF = 34
SKIP_PERIODS = 1
EPOCHS = 25
TEST_STEPS = TEST_YEARS*12
VAL_STEPS = VAL_YEARS*12
BATCH_SIZE = int(TEST_STEPS/2)
REMOVE_OUTLIER = 0.05
START_YEAR = 1960
ALPHAS = [0.01, 0.05, 0.1, 0.4]


# Plotting Parameters
COLOURS = ['paleturquoise', 'cyan', 'teal', 'darkslategray']
SMALLEST_SIZE = 6
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
BIGGEST_SIZE = 18
FIG_SIZE = (20, 6)

# Result Outputs
RESULTS_COLUMNS = [
    ['During COVID', 'During COVID', 'During COVID', 'During COVID', 'During COVID', 'During COVID', 'During COVID', 'During COVID',
     'After GFC', 'After GFC', 'After GFC', 'After GFC', 'After GFC', 'After GFC', 'After GFC', 'After GFC',
     'During GFC', 'During GFC', 'During GFC', 'During GFC', 'During GFC', 'During GFC', 'During GFC', 'During GFC',
     'Before GFC', 'Before GFC', 'Before GFC', 'Before GFC', 'Before GFC', 'Before GFC', 'Before GFC', 'Before GFC'],
    [inflation, inflation, treasury, treasury, unemployment, unemployment, wage, wage,
     inflation, inflation, treasury, treasury, unemployment, unemployment, wage, wage,
     inflation, inflation, treasury, treasury, unemployment, unemployment, wage, wage,
     inflation, inflation, treasury, treasury, unemployment, unemployment, wage, wage],
    ['Val', 'Test', 'Val', 'Test', 'Val', 'Test', 'Val', 'Test',
     'Val', 'Test', 'Val', 'Test', 'Val', 'Test', 'Val', 'Test',
     'Val', 'Test', 'Val', 'Test', 'Val', 'Test', 'Val', 'Test',
     'Val', 'Test', 'Val', 'Test', 'Val', 'Test', 'Val', 'Test']
]

AGGREGATE_COLUMNS = [
    [inflation, inflation, treasury, treasury,
        unemployment, unemployment, wage, wage],
    ['Val', 'Test', 'Val', 'Test', 'Val', 'Test', 'Val', 'Test']
]

PARAM_COLUMNS = [
    [inflation, inflation, inflation, inflation, treasury, treasury, treasury, treasury,
        unemployment, unemployment, unemployment, unemployment, wage, wage, wage, wage],
    ['During COVID', 'After GFC', 'During GFC', 'Before GFC', 'During COVID', 'After GFC', 'During GFC', 'Before GFC',
        'During COVID', 'After GFC', 'During GFC', 'Before GFC', 'During COVID', 'After GFC', 'During GFC', 'Before GFC']
]
