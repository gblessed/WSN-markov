import math

import numpy as np

# Describe the scenarios that will be simulated
# scenarios should be described in the following format:
# scenario_name = (routing_topology, sleep_scheduling, aggregation_model)
# where routing_topology may be:
#   'DC'   : Direct Communication
#   'MTE'  : Minimum Transmission Energy
#   'LEACH': LEACH
#   'FCM  ': Fuzzy C-Means
# and sleep_scheduling may be:
#   None                 : No sleep scheduling (mind that None is not a string)
#   'Pso'                : Particle Swarm Optimization
#   'ModifiedPso'        : Modified PSO
#   'GeneticAlgorithm'   : Genetic Algorithm
# and aggregation_model may be:
#   'zero'  : Zero cost
#   'total' : 100% cost
#   'linear': TODO spec
#   'log'   : log cost
# the 4th argument is the nickname of that plot, if not specified (None),
# then the name is: routing_topology + sleep_scheduling

# for convenience, the scenarios list also accepts commands that are
# executed in run.py

scenario0 = ('DC',    None,              'zero',  None)
scenario1 = ('LEACH', None,              'total',  None)
scenario2 = ('MTE',   None,              'total', None)
scenario3 = ('FCM',   None,              'total',  None)
scenario4 = ('FCM',  'ModifiedPso',      'zero',  'FCMMPSO')
scenario5 = ('FCM',  'Pso',              'zero',  None)
scenario6 = ('FCM',  'Ecca',             'zero',  'ECCA')
scenario7 = ('FCM',  'GeneticAlgorithm', 'zero',  None)
scenario31 = ('FCM',   None,              'zero',  'BS at (125,125)')
scenario32 = ('FCM',   None,              'zero',  'BS at (65,65)')
scenario33 = ('FCM',   None,              'zero',  'BS at (0,0)')
scenario34 = ('FCM',   None,              'zero',  'BS at (-65,-65)')

# scenario15 = ('LEACH', 'Pso', 'zero',  None)
# Modified Leach Algorithm with Controller
# Paper Implementation is based on this scenario
scenario14 = ('MLC', None, 'total', None)

# list with all scenarios to simulate

# example of configuration to get first part of results
# scenarios = [
#              "cf.FITNESS_ALPHA=0.5",
#              "cf.FITNESS_BETA=0.5",
# scenario3,
#              "plot_clusters(network)",
#              scenario0,
# scenario1,
# scenario2,
# scenario5,
#              scenario4,
#              "plot_time_of_death(network)",
#              "plot_traces(traces)",
#              "network.get_BS().pos_y=-75.0",
#              scenario3,
#              scenario0,
#              scenario1,
#              scenario2,
#              scenario5,
#              scenario4,
#              "save2csv(traces)",
#            ]

dry_run_scenarios = [
    "cf.FITNESS_ALPHA=0.7",
    "cf.FITNESS_BETA=0.3",
    scenario0,
    scenario1,
    scenario2,
    scenario3,
    scenario4,
    scenario6,
    scenario7,
    scenario31,
    scenario32,
    scenario33,
    scenario14,
    "save_timer_logs2csv(timer_logs)",
    "save2csv_raw(traces)",
    "plot_traces(traces)",
    "save2csv(traces)",
]

scenarios = [
    "cf.FITNESS_ALPHA=0.7",
    "cf.FITNESS_BETA=0.3",
      #            scenario0,
                #  scenario1,
    #              scenario2,
                #  scenario3,
    #              scenario4,
#                   scenario5,
    #              "cf.FITNESS_ALPHA=0.34",
    #              "cf.FITNESS_BETA=0.33",
    #              "cf.FITNESS_GAMMA=0.33",
    #             scenario6,
    #              scenario6,
    #              'cf.BS_POS_X=65.0',
    #              'cf.BS_POS_Y=65.0',
    #              scenario32,
    #              'cf.BS_POS_X=0.0',
    #              'cf.BS_POS_Y=0.0',
    #              scenario33,
    #              'cf.BS_POS_X=-65.0',
    #              'cf.BS_POS_Y=-65.0',
    #              scenario34,
    #scenario15,
    scenario14,
    "save_timer_logs2csv(timer_logs)",
    "save2csv_raw(traces)",
    "plot_traces(traces)",
    "save2csv(traces)",
]

# scenarios = [
#              "cf.FITNESS_ALPHA=0.5",
#              "cf.FITNESS_BETA=0.5",
#              scenario4,
#              scenario5, #              scenario6,
#              "cf.FITNESS_ALPHA=0.75",
#              "cf.FITNESS_BETA=0.25",
#              scenario4,
#              scenario5,
#              scenario6,
#              "cf.FITNESS_ALPHA=0.25",
#              "cf.FITNESS_BETA=0.75",
#              scenario4,
#              scenario5,
#              scenario6,
#              "cf.FITNESS_ALPHA=1.0",
#              "cf.FITNESS_BETA=0.0",
#              scenario4,
#              scenario5,
#              scenario6,
#              "cf.FITNESS_ALPHA=0.0",
#              "cf.FITNESS_BETA=1.0",
#              scenario4,
#              scenario5,
#              scenario6,
#              "save2csv(traces)",
#            ]

# tracer options
TRACE_ENERGY = 1
TRACE_ALIVE_NODES = 1
TRACE_COVERAGE = 1
TRACE_LEARNING_CURVE = 1
TRANS_RATE = 4150       # IN bps
# Runtime configuration
MAX_ROUNDS = 21
# number of transmissions of sensed information to cluster heads or to
# base station (per round)
MAX_TX_PER_ROUND = 1

NOTIFY_POSITION = 1

# time step for checking a node's stage in s

TIME_STEP = 1.5e-9 # set to 10ms

MARKOV_PREDICTION_INTERVAL = 4

ITTR_COUNT = 0 # 
markov_path = ''

DISTANCE_THRESH = 10
# Network configurations:
# number of nodes
NB_NODES = 100
SUBCONT0 = -3
SUBCONT1 = -4
# Set to True if number of clusters should be the same as number of
# controllers
NB_CLUSTERS_CONTROLLERS = False
# Number of clusters
NB_CLUSTERS = 0
# node sensor range
COVERAGE_RADIUS = 15  # meters
# node transmission range
TX_RANGE = 30  # meters
BSID = -1
# All ids for controllers
CONTROLLER_IDS = [BSID, ]
# The number of Controllers
NB_CONTROLLERS = 2
TIME_SLOT = 0.10
# area definition
AREA_WIDTH = 200.0
AREA_LENGTH = 200.0
# base station position
BS_POS_X = 50.0
BS_POS_Y = 50.0
SUB_CON0_POS_X= 50.0
SUB_CON0_POS_Y=100.0
SUB_CON1_POS_X=150.0
SUB_CON1_POS_Y=100.0

# packet configs
MSG_LENGTH = 4000  # bits
HEADER_LENGTH = 150  # bits
# initial energy at every node's battery
INITIAL_ENERGY = 2  # Joules

# In Heterogeanous Node Mode, different nodes have different Energy
HETEROGEANOUS = False
# When True, Nodes next hop is determine by the entry in the flow table
USE_FLOW_TABLE = False

# sensing energy = 3v * 2.5mA * 2uS
SENSING_ENERGY = 1.5771e-6 #Joules

# energy dissipated processing -> sensing
E_PROCESSING_SENSING = 14.9e-6 #Joules

# energy dissipated processing -> transmitting
E_PROCESSING_TRANSMITTING = 6.3e-6 #Joules

# energy dissipated idle -> transmitting
E_IDLE_TRANSMITTING = 5.86e-6 #Joules

# energy dissipated idle -> aggregating
E_IDLE_AGGREGATING = 0.0064e-6 #Joules

# ENERGY DISSIPATED WHEN IDLE
E_IDLE = 3.27e-10 # Joules

# ENERGY DISSIPATED WHEN SLEEPING
E_SLEEP = 0
# Energy Configurations
# energy dissipated at the transceiver electronic (/bit)
E_ELEC = 50e-9  # Joules
# energy dissipated at the data aggregation (/bit)
E_DA = 5e-9  # Joules
# energy dissipated at the power amplifier (supposing a multi-path
# fading channel) (/bin/m^4)
E_MP = 0.0013e-12  # Joules
# energy dissipated at the power amplifier (supposing a line-of-sight
# free-space channel (/bin/m^2)
E_FS = 10e-12  # Joules
THRESHOLD_DIST = math.sqrt(E_FS/E_MP)  # meters
# energy dissipated when generation transition matrices
E_TS = 5e-9  # Joules
#sub-controller region
CLUSTER2_BASE_DIST= 0.5 * (AREA_LENGTH)

# Routing configurations:
CLUSTER_BASE_DIST = 0.5 * AREA_LENGTH
# Maximum Number of Iterations to get cluster centroids
MAX_NUM_ITER = 100

# FCM fuzzyness coeficient
FUZZY_M = 2
FUZZY_M2 = 3
# Sleep Scheduling configurations:
NB_INDIVIDUALS = 10
MAX_ITERATIONS = 50
# ALPHA and BETA are the fitness function' weights
# where ALPHA optimizes energy lifetime, BETA the coverage
FITNESS_ALPHA = 0.34
FITNESS_BETA = 0.33
FITNESS_GAMMA = 0.33
WMAX = 0.6
WMIN = 0.1



# Other configurations:
# grid precision (the bigger the faster the simulation)
GRID_PRECISION = 1  # in meters
# useful constants (for readability)
INFINITY = float('inf')
MINUS_INFINITY = float('-inf')

RESULTS_PATH = './results/'
