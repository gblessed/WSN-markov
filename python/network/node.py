from matplotlib.pyplot import xcorr
import numpy as np
import config as cf
from collections import OrderedDict
from python.network.energy_source import *
from python.utils.utils import *
from python.utils import markov_model
from python.utils.timer import Timer, Stage
import time
import timer
from timer4 import SignalManager
from apscheduler.schedulers.background import BackgroundScheduler

class Controller(object):
    def __init__(self,id,x,y,parent=None):
        self.pos_x = x
        self.pos_y = y
        self.energy_source = PluggedIn(self)

        self.id = id
        self.network_handler = parent
        self.alive = 1
        self.flow_table = None
        self.amount_transmitted = 0
        self.amount_received = 0
        self.is_sleeping = 0
        self.controller_membership = cf.BSID
        self.membership = -999

    def is_head(self):
        return 0
    
    @classmethod
    def new_controller(cls, id, parent):
        """ Return a controller Node """
        node = cls(id, parent)
        node.energy_source = PluggedIn(cls)
        return node

    #Transmission for controller    
    def transmit(self, msg_length=None, destination=None, msg_type = 0):
        logging.debug("node %d transmitting." % (self.id))
        if not msg_length:
            msg_length = self.tx_queue_size
        msg_length += cf.HEADER_LENGTH

        # If flow table is defined use destination set on flow table
        if self.flow_table:
            destination = self.flow_table[self.id]["destination"]
            distance = self.distance_to_endpoint
        elif not destination:
            destination = self.network_handler[self.next_hop]
            distance = self.distance_to_endpoint
        else:
            distance = calculate_distance(self, destination)

        destination.receive(msg_length,msg_type)
        # after the message is sent, queue is emptied
        self.tx_queue_size = 0
        self.amount_transmitted += msg_length


    def receive(self, msg_length,msg_type = 0):
        logging.debug("node %d receiving." % (self.id))
        self._aggregate(msg_length - cf.HEADER_LENGTH)
        self.amount_received += msg_length

    def _aggregate(self, msg_length):
        logging.debug("node %d aggregating." % (self.id))
        # number of bits to be sent increase while forwarding messages
        aggregation_cost = self.aggregation_function(msg_length)
        self.tx_queue_size += aggregation_cost        
       


class Node(object):
    def __init__(self, id, parent=None):
        self.pos_x = np.random.uniform(0, cf.AREA_WIDTH)
        self.pos_y = np.random.uniform(0, cf.AREA_LENGTH)

        #identify the IDs for sub-controllers and controller and give them infinite energy
        if id == cf.BSID:
            self.energy_source = PluggedIn(self)
        else:
            self.energy_source = Battery(self)

        self.id = id
        self.network_handler = parent
        # A list of all timer events measured and stored
        self.timer_logs = OrderedDict()

        # Variables used for predictions
        # A list of all transitions from state to state
        self.transitions = []
        # A list of predicted transitions from state to state
        self.predicted_transitions = []
        # A list of all energy usage per round in line with transitions
        self.energy_map = []
        # Predicted energy consumption for all the states transition predicted
        self.predicted_total_energy_consumed = 0
        self.predicted_remain_energy_list = []
        # the node transition Matrix generated
        transition_matrix = None
        self.scheduler = None
        self.cluster_head_times = []
        self.node_stage = Stage.SLEEPING
        self.sched = BackgroundScheduler()
        self.sched.add_job(self.log_stage, 'interval', seconds=cf.TIME_STEP)
        self.sched.start()
        self.log_stage()
        self.reactivate()
        self.markov_energy = 0



    def __repr__(self):
        if self.is_controller:
            return "<Controller %s>" % (self.id)
        if self.id == cf.BSID:
            return "<BS>"
        return "<Node %s energy %s>" % (self.id, float(self.energy_source.energy))

    def __str__(self):
        if self.is_controller:
            return "Controller %s at (%s, %s)" % (self.id, self.pos_x, self.pos_y)
        if self.id == cf.BSID:
            return "BS at (%s, %s)" % (self.pos_x, self.pos_y)
        return "Node %s at (%s, %s) with energy %s" % (self.id, self.pos_x, self.pos_y, self.energy_source.energy)

    @property
    def is_controller(self):
        # A controller has infinite energy but it's not the base station
        return isinstance(self.energy_source, PluggedIn) and self.id != cf.BSID
    '''
    @classmethod
    def new_controller(cls, id, parent):
        """ Return a controller Node """
        node = cls(id, parent)
        node.energy_source = PluggedIn(cls)
        return node
    '''

    def reactivate(self):
        """Reactivate nodes for next simulation."""
        self.alive = 1
        self.tx_queue_size = 0
        self._next_hop = cf.BSID#change to reflect subcontrollers
        self.distance_to_endpoint = 0
        self.amount_sensed = 0
        self.amount_transmitted = 0
        self.amount_received = 0
        self.membership = cf.BSID
        # Controllers are attached to certain cluster(s) in this case
        # Controller Nodes which controls flow table creation
        # and cluster heads
        # aggregation function determines the cost of forwarding messages
        # (in number of bits)
        self.aggregation_function = lambda x: 0
        self.time_of_death = cf.INFINITY
        self._is_sleeping = 0
        self.sleep_prob = 0.0
        # for coverage purposes
        self.neighbors = []
        self.nb_neighbors = -1
        self.exclusive_radius = 0
        # Flow Tables control communication when enabled
        self.flow_table = None
        self.packet_received =0

    @property
    def next_hop(self):
        # If Flow Table routing, next hop is obtained from flow table
        if  self.flow_table and self.id != cf.BSID:
            return self.flow_table[self.id]["destination"]
        return self._next_hop

    @next_hop.setter
    def next_hop(self, value):
        self._next_hop = value
        distance = calculate_distance(self, self.network_handler[value])
        self.distance_to_endpoint = distance

    @property
    def is_sleeping(self):
        if self.is_head():
            self._is_sleeping = 0
        return self._is_sleeping

    @is_sleeping.setter
    def is_sleeping(self, value):
        """Cluster heads cannot be put to sleep."""
        self._is_sleeping = value if not self.is_head() else 0

    def _record_time(Stage=None):
        """ A decorator that measures the time of execution for the method """
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if not Stage:
                    raise "Stage must be defined to measure time"
                timer = Timer(self.id, Stage)
                self.transitions.append(Stage)
                
                x = func(self, *args, **kwargs)
                timer.stop()
                # print('elapsed time-> ',timer.elapsed_time)
                self.timer_logs[timer.uuid] = timer

            return wrapper
        return decorator

   # @_record_time(Stage=Stage.SLEEPING)
    def set_sleeping_stage(self):
        """ Set node as sleeping for this Stage"""
        self.energy_source.consume(cf.E_SLEEP)
        self.node_stage = Stage.SLEEPING
        # time.sleep(1)
        pass

    #@_record_time(Stage=Stage.IDLE)
    def set_idle_stage(self):
        """ Set node as idle for this Stage"""
        
        self.energy_source.consume(cf.E_IDLE)
        self.node_stage = Stage.IDLE
        pass


    def _only_active_nodes(func):
        """This is a decorator. It wraps all energy consuming methods to
        ensure that only active nodes execute this method. Also it automa-
        tically calls the battery.
        """

        def wrapper(self, *args, **kwargs):
            if self.alive and not self.is_sleeping:
                func(self, *args, **kwargs)
                return 1
            else:
                return 0
        return wrapper

    def is_head(self):
        # a head node's destination is set to either a controller  or a Base Station
        #if self.next_hop == cf.BSID and self.id != cf.BSID and self.alive:
        #    return 1
        if self.next_hop == cf.SUBCONT0:
            return 1
        elif self.next_hop == cf.SUBCONT1:
            return 1
        else:
            return 0

    def is_ordinary(self):
        return 1 if self.next_hop != cf.BSID and self.id != cf.BSID else 0

    @_only_active_nodes
    #@_record_time(Stage=Stage.AGGREGATING)
    def _aggregate(self, msg_length):
        if self.node_stage == Stage.IDLE:
            self.energy_source.consume(cf.E_IDLE_AGGREGATING )
        self.node_stage = Stage.AGGREGATING
        logging.debug("node %d aggregating." % (self.id))
        # number of bits to be sent increase while forwarding messages
        aggregation_cost = self.aggregation_function(msg_length)
        self.tx_queue_size += aggregation_cost

        # energy model for aggregation
        energy = cf.E_DA * aggregation_cost
        # energy = 2.2e-6 * 3 * 2.6e-6 * msg_length
        self.energy_source.consume(energy)

    @_only_active_nodes
    #@_record_time(Stage=Stage.TRANSMITTING)
    def transmit(self, msg_length=None, destination=None, msg_type = 0):
        if self.node_stage == Stage.AGGREGATING:
            self.energy_source.consume(cf.E_PROCESSING_TRANSMITTING)
        elif self.node_stage == Stage.IDLE:
            self.energy_source.consume(cf.E_IDLE_TRANSMITTING)
        self.node_stage = Stage.TRANSMITTING
        logging.debug("node %d transmitting." % (self.id))
        if not msg_length:
            msg_length = self.tx_queue_size
        msg_length += cf.HEADER_LENGTH
        msg_length = cf.MSG_LENGTH+cf.HEADER_LENGTH

        # If flow table is defined use destination set on flow table
        if self.flow_table:
            destination = self.flow_table[self.id]["destination"]
            distance = self.distance_to_endpoint
        elif not destination:
            destination = self.network_handler[self.next_hop]
            distance = self.distance_to_endpoint
        else:
            distance = calculate_distance(self, destination)

        # transmitter energy model
        energy = cf.E_ELEC
        if distance > cf.THRESHOLD_DIST:
            energy += cf.E_MP * (distance**4)
        else:
            energy += cf.E_FS * (distance**2)
        energy *= msg_length

        time_taken = 0
        while msg_length != 0:
            if msg_length <= cf.TRANS_RATE:
                if distance < cf.DISTANCE_THRESH:
                    destination.receive(msg_length, msg_type)
                msg_length -= msg_length
                if msg_length == 0:
                    self.amount_transmitted += 1
            else:
                destination.receive(cf.TRANS_RATE, msg_type)
                msg_length -= cf.TRANS_RATE
                if msg_length == 0:
                        self.amount_transmitted += 1
            time_taken += 1
        #if msg_length == 0:
            #self.amount_transmitted += 1
        #print('transmit time', time_taken)
        # after the message is sent, queue is emptied
        if msg_type == 1:
            self.markov_energy += energy
        self.tx_queue_size = 0
        #print(str(self.id)+' transmitted ->'+str(self.amount_transmitted)+' to '+str(destination.id))
        self.energy_source.consume(energy)
        # node enters idle state then sleeps
        self.set_idle_stage()
        self.set_sleeping_stage()
        self.set_idle_stage()
        # self.sense()


    @_only_active_nodes
    #@_record_time(Stage=Stage.RECEIVING)
    def receive(self, msg_length, msg_type = 0):
        self.node_stage = Stage.RECEIVING
        logging.debug("node %d receiving." % (self.id))
        if msg_type == 0:
            if self.is_head():
                self._aggregate(msg_length - cf.HEADER_LENGTH)
        # print(str(self.id)+'  received '+str(self.amount_received))
        self.amount_received += msg_length
        if (self.amount_received == cf.MSG_LENGTH+cf.HEADER_LENGTH): 
            self.packet_received += 1
        # energy model for receiver
        energy = cf.E_ELEC * msg_length
        self.energy_source.consume(energy)

    @_only_active_nodes
   # @_record_time(Stage=Stage.SENSING)
    def sense(self):
        if self.node_stage == Stage.AGGREGATING:
            self.energy_source.consume(cf.E_PROCESSING_SENSING)
        self.node_stage = Stage.SENSING
        self.tx_queue_size = cf.MSG_LENGTH
        self.amount_sensed += cf.MSG_LENGTH
        self.energy_source.consume(cf.SENSING_ENERGY)
        # node enters idle state then sleeps
        self.set_idle_stage()
        self.set_sleeping_stage()
    

    def battery_depletion(self):
        self.alive = 0
        self.sleep_prob = 0.0
        self.time_of_death = self.network_handler.round
        self.network_handler.deaths_this_round +=1

    def generate_transition_matrix(self):
        self.transition_matrix = markov_model.generate_transition_matrix(self.transitions)
        
        # print(self.transitions)
        # print("Transition Matrix")
        # print(self.transition_matrix)
        # energy model for transition matrix generation
        self.energy_source.consume(cf.E_TS)
        self.transmit(destination = self.network_handler[self.next_hop], msg_type = 1)
        self.markov_energy += cf.E_TS
        
        return self.transition_matrix


    def log_stage(self):
        print('logged Stage -> ', self.node_stage)
        if not self.node_stage:
            raise "Stage must be defined to measure time"
        timer = Timer(self.id, self.node_stage)
        self.transitions.append(self.node_stage)
        # x = func(self, *args, **kwargs)
        timer.stop()
        # print('elapsed time-> ',timer.elapsed_time)
        self.timer_logs[timer.uuid] = timer
        # print('TIMER    ',timer.uuid, timer.Stage.name)
        # QTimer.singleShot(cf.TIME_SLOT, self.log_stage)
