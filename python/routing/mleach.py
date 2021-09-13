import logging
import sys
import numpy as np
import skfuzzy
from scipy.cluster.vq import kmeans, vq, whiten
from skfuzzy import membership
import config as cf
from python.network.flow_table import FlowTable
from python.network.network import Network
from python.network.node import *
from python.routing.mte import *
from python.routing.routing_protocol import *
#from python.sleep_scheduling.sleep_scheduler import *
from python.utils.markov_model import MarkovChain
from python.network.sleepscheduler import SleepScheduler
import csv
from python.utils.utils import  get_result_path
from datetime import datetime

"""Every node communicate its position to the base station. Then the  Controllers/ BS uses Fuzzy C-Means
Clustering to define clusters and broadcast this information to the network. Finally, a round is executed.
"""

class MLC(RoutingProtocol):
    """ Modified Leach with Controller Nodes """
    itrr  = 0
    this_remaining_energies_result_path = ''
    # Controller network sizes
    width = cf.AREA_WIDTH
    length = cf.AREA_LENGTH/cf.NB_CONTROLLERS

    networks = [] # store the controller networks
    no_of_times_am_head ={}
    for i in range(0, cf.NB_NODES):
        no_of_times_am_head['id '+str(i)] = 0
    # store the original number of nodes in each cluster. This would be used to determine
    # when the total number of remaining alive nodes in the cluster is less than a certain
    # percentage of the original
    clusters = {}

    def ch_selection_energy_function(self, node, subc):
        """ An energy function to compute the optimal energy of node to be Cluster head
        taking into account the distance of node from base station and energy level. The
        higher the value, the more likely the node is best fit to be cluster head
        """
        distance_from_subc = calculate_distance(node, subc)
        # The distance cost is the difference between the farthest possible distance a node can
        # be from the base station and the distance of the current node from the base station
        edge_node = Node(0)
        edge_node.pos_x = self.width/2
        edge_node.pos_y = 0

        max_possible_distance = calculate_distance(edge_node, subc)
        return (max_possible_distance - distance_from_subc) * node.energy_source.energy

    def node_in_controller_region(self, node, controller):
        """ Returns the controller in the region of the node """
        # If a node is within the controller region
        return (controller.pos_x - MLC.length/2) <= node.pos_x and \
            (MLC.length/2 + controller.pos_x) > node.pos_x

    def pre_communication(self, network):
        """This method is called before round 0."""
        # Setup Controllers on network
        super().pre_communication(network)

        # Nodes closest to centroids i.e
        centroids = []

        
        for controller in network.controller_list:
            #controller.pos_x = MLC.length/2 + (MLC.length*i)
            #controller.pos_y = MLC.width/2
            
            nodes = [node for node in network.get_sensor_nodes() if \
                        self.node_in_controller_region(node, controller)]
            #for node in nodes:
             #   node.membership = controller.id            
            controller_network = Network(sensor_nodes= nodes)
            #controller_network.append(controller)
            MLC.networks.append(controller_network)
        
        nodeidlist = []
        for node in MLC.networks[0]:
            nodeidlist.append(node.id)
            with open('controllernodes1.txt', 'w') as f:
                f.write(str(nodeidlist))

        nodeidlist = []
        for node in MLC.networks[1]:
            nodeidlist.append(node.id)
            with open('controllernodes2.txt', 'w') as f:
                f.write(str(nodeidlist))

        # Initialize Energy Maps for each node
        for node in network.get_sensor_nodes():
            if isinstance(node,Controller) == False:
                node.energy_map.append(node.energy_source.energy)

    def setup_phase(self, network, round_nb):
        """ Network Setup Phase. The network setup for all the controller networks
        are executed
        """
        for i, ntwk in enumerate(MLC.networks):
            self._setup_phase_network(ntwk, round_nb, i)
            # # Controllers send data to Base Station after each round
            controller = ntwk[-1]
            controller.transmit(destination=network.get_BS())

        # Update Energy Map at the end of each round
        for node in network.get_sensor_nodes():
            if isinstance(node,Controller) == False:
                node.energy_map.append(node.energy_source.energy)

        
        round_energies = []

        if round_nb == 0:
            #Perform prediction operations
            this_remaining_energies_result_dir = os.path.join(get_result_path(), 'markov_predictions')
            if not os.path.exists(this_remaining_energies_result_dir):
                os.makedirs(this_remaining_energies_result_dir)
            this_remaining_energies_result_path = os.path.join(this_remaining_energies_result_dir, 'MLC_' + datetime.today().strftime('%H-%M-%S') + '_markov_predictions.csv')
            cf.markov_path = this_remaining_energies_result_path
            round_energies_result_csv =  open(this_remaining_energies_result_path, mode='w')
            round_energies_result_csv_writer = csv.writer(round_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = [f'node_{i}' for i in range(cf.NB_NODES)]
            round_energies_result_csv_writer.writerow(header)
            round_energies_result_csv.close()
        
        if round_nb > 0:
            if round_nb % cf.MARKOV_PREDICTION_INTERVAL == 0:              
                print('kaiii')
                for node in network.get_sensor_nodes():
                  if isinstance(node,Controller) == False:
                    transition_matrix = node.generate_transition_matrix()  
                    markov_model = MarkovChain()
                    predictions = markov_model.predict(node.transitions[-1], transition_matrix, no_predictions=cf.MARKOV_PREDICTION_INTERVAL)
                    node.predicted_energy_consumed = predictions
                    print('predicted energy consumed->',node.predicted_energy_consumed)
                    node.transitions = []
                    node.predicted_remain_energy_list = []
                    initial_remaining = node.energy_source.energy
                    predicted_remaining_energy = np.zeros(cf.MARKOV_PREDICTION_INTERVAL)
                    predicted_remaining_energy[0] =  initial_remaining - node.predicted_energy_consumed[0]
                    predicted_remaining_energy[0] = 0 if predicted_remaining_energy[0] < 0 else predicted_remaining_energy[0]
                    # predicted_remaining_energy[0] = initial_remaining - node.predicted_energy_consumed[cf.ITTR_COUNT]
                    for i in range(1,cf.MARKOV_PREDICTION_INTERVAL):
                        predicted_remaining_energy[i] = predicted_remaining_energy[i-1] -node.predicted_energy_consumed[i]
                        if predicted_remaining_energy[i] < 0:
                            predicted_remaining_energy[i] = 0
                        initial_remaining -= node.predicted_energy_consumed[i]
                    node.predicted_remain_energy_list = predicted_remaining_energy
                    # if predicted_remaining_energy < 0:
                    #     predicted_remaining_energy = 0
                    print('predicted remaining consumed->',predicted_remaining_energy)
                    round_energies.append(predicted_remaining_energy)
                cf.ITTR_COUNT += 1
                round_energies_result_csv =  open(cf.markov_path, mode='a')
                round_energies_result_csv_writer = csv.writer(round_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  
                round_energies_result_csv_writer.writerow(round_energies)
                round_energies_result_csv.close()
                if cf.ITTR_COUNT>=2:
                    cf.ITTR_COUNT=0

        
        # if round_nb > 0 and round_nb % MARKOV_PREDICTION_INTERVAL == 0:
            
        #     for node in network.get_sensor_nodes():
        #        if isinstance(node,Controller) == False:
        #            predicted_remaining_energy = node.energy_source.energy-node.predicted_energy_consumed[cf.ITTR_COUNT]
        #            if predicted_remaining_energy < 0:
        #                predicted_remaining_energy = 0
        #            print('predicted remaining consumed->',predicted_remaining_energy)
        #            round_energies.append(predicted_remaining_energy)
        #     cf.ITTR_COUNT += 1
        #     round_energies_result_csv =  open(cf.markov_path, mode='a')
        #     round_energies_result_csv_writer = csv.writer(round_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  
        #     round_energies_result_csv_writer.writerow(round_energies)
        #     round_energies_result_csv.close()
            

    def _setup_phase_network(self, network, round_nb, controller):
        """The base station/controller uses Fuzzy C-Means to clusterize the network. The
        optimal number of clusters is calculated.
        The Cluster heads are determine by the base station using the `Algorithm` based on
        Paper X for each cluster (only in the initial round).
        Then each cluster head chooses a new cluster head for the next round.
        """

        # Add Controllers to the Nodes if enabled
        logging.debug('MLC: %s -> setup phase', network.id)
        if round_nb == 0:
            # Base station selects clusters and Cluster heads in the first round
            self.clusterize_network(network)
        else:
            # In subsequent rounds, new cluster heads are chosen from cluster util the total
            # number of nodes in the clusters are less than 1/3rd of the original nodes
            self.head_rotation(network, round_nb)


    def head_rotation(self, network, round_nb):
        """ Current cluster heads choose next cluster head based on `Algorithm` """

        logging.debug('MLC: Cluster Head Rotation')
        # current cluster heads choose next cluster head with the most
        # residual energy and nearest to the base station/<cluster centroid>
        #number of clusters is a function of alive nodes 
        #change to reflect sub-controllers
        # print('mlc network ', network)
        nb_clusters = calculate_opt_nb_clusters(len(network.get_alive_nodes()))
        for cluster_id in range(0, nb_clusters):
            # cluster = Network(network.get_nodes_by_membership(cluster_id))
            cluster = network.get_nodes_by_membership(cluster_id)
            # When the total number of nodes in a cluster falls below 2/3rd of the original
            # node numbers, reconstitute the clusters
            # print('the clusters are....', cluster)
            # print(len(cluster))
            # print(self.clusters[cluster_id])

            if len(cluster) < self.clusters[cluster_id]*0.67:
                return self.clusterize_network(network)

            # check if there is someone alive in this cluster
            if len(cluster) == 0:
                continue

            # someone is alive, find node with highest energy in the cluster
            # to be the next cluster head
            
            
            # get the current head
            # current_head =None
            # for node in cluster:
            #     if node.next_hop == cf.SUBCONT0:
            #         self.current_head = node
            #         self.no_of_times_am_head['id '+str(self.current_head.id)]+= 1
            #     if node.next_hop == cf.SUBCONT1:
            #         self.no_of_times_am_head['id '+str(self.current_head.id)]+= 1
            #         self.current_head = node
            # cluster_for_CH = cluster
            
            # for all_nodes in self.no_of_times_am_head.keys():
            #     if (self.no_of_times_am_head[all_nodes]) > 0.1* cf.MAX_ROUNDS:
            #         for node in cluster_for_CH:
            #             if node.id == int(all_nodes[-1]):
            #                 cluster_for_CH.remove(node)
            #                 self.no_of_times_am_head[all_nodes] = 0

            # get the node that has been cluster head for 2 consecutive rounds

            for node in cluster:
                if node.next_hop == cf.SUBCONT0:
                    self.current_head = node
                    node.cluster_head_times.append(True) # append True to the list of the node has been a cluster head
                if node.next_hop == cf.SUBCONT1:
                    self.current_head = node
                    # self.no_of_times_am_head['id '+str(self.current_head.id)]+= 1
                    node.cluster_head_times.append(True)  # append True to the list of the node has been a cluster head
                else:
                    node.cluster_head_times = []
            cluster_for_CH = cluster
            
            
            for node in cluster:
                if len(node.cluster_head_times)==1:
                        cluster_for_CH.remove(node)
                        node.cluster_head_times = []
                        
            


            # if self.current_head.energy_source.energy >= self.calculate_average_energy(cluster):
            if round_nb %3 != 0:
                continue
                

            highest_energy = cf.MINUS_INFINITY
            next_head = None
            for node in cluster_for_CH:
                node_energy = self.ch_selection_energy_function(node, network[-1])
                if node_energy > highest_energy:
                    highest_energy = node_energy
                    next_head = node

            if len(cluster) > 0:
                for node in cluster:
                    node.next_hop = next_head.id    
                    #print('next heads ', next_head.id)
                next_head.next_hop = network[-1].id

            if len(network.get_alive_nodes()[:-1]) == 1:
                next_head = network.get_alive_nodes()[0]
                next_head.next_hop = network[-1].id



            #print('next hops for heads ', next_head.next_hop)

    def clusterize_network(self, network):
        """ Create clusters from the network using FCM """

        logging.debug("MLC:: Reconstituting Cluster heads")
        sensor_nodes = network.get_alive_nodes()

        #calculate the average distance to the BS
        def transform(node): return calculate_distance(node, network[-1])
        distances_to_SUBC = [transform(node) for node in sensor_nodes]
        avg_distance_to_BS = np.average(distances_to_SUBC)
        #nb_clusters = calculate_nb_clusters(avg_distance_to_BS)
        #nb_clusters = int (0.1* (len(sensor_nodes)))
        nb_clusters = calculate_opt_nb_clusters(len(sensor_nodes))
        #print('len of clusters', nb_clusters)

        cf.NB_CLUSTERS = nb_clusters
        # cf.NB_CLUSTERS = calculate_opt_nb_clusters(len(sensor_nodes,))

        data = [[node.pos_x, node.pos_y]
                for node in network.get_alive_nodes()]
        if not data:
            return
        # Use Fuzzy C-Means Clustering to Determine Clusters
        # format data to shape expected by skfuzzy API
        data = np.array(data).transpose()
        centroids, membership = skfuzzy.cluster.cmeans(data, nb_clusters,
                                                       cf.FUZZY_M, error=0.005,
                                                       maxiter=1000,
                                                       init=None)[0:2]
        # print('---', centroids, membership, '------')
        membership = np.argmax(membership, axis=0)
        print('length of centroid',  len(centroids))

        heads = []
        # also annotates centroids to network
        network.centroids = []
        for cluster_id, centroid in enumerate(centroids):
            tmp_centroid = Node(0)
            tmp_centroid.pos_x = centroid[0]
            tmp_centroid.pos_y = centroid[1]

            network.centroids.append(tmp_centroid)
            self.clusters[cluster_id] = 0

            nearest_node = None
            max_energy_function= cf.MINUS_INFINITY
            for node in network.get_alive_nodes():
                if node in heads:
                    continue
                energy_function = self.ch_selection_energy_function(node, network[-1])
                if energy_function > max_energy_function:
                    nearest_node = node
                    max_energy_function = energy_function
            if nearest_node:
                # Nearest Node is made Cluster Head and CH send data to Cluster Controllers
                nearest_node.next_hop = network[-1].id
                nearest_node.membership = cluster_id
                heads.append(nearest_node)

       # print('heads are ', heads)
        # assign ordinary network to cluster heads using fcm
        for i, node in enumerate(network.get_alive_nodes()):
            if node in heads:
                continue
            if isinstance(node,Controller):
                continue
            if (node.id == cf.BSID):
                continue
            cluster_id = membership[i]
            node.membership = cluster_id
            head = [x for x in heads if x.membership == cluster_id][0]
           # print('head id', head.id)
            node.next_hop = head.id
            self.clusters[cluster_id] += 0

        # cluster_ids = set([x.next_hop for x in network.get_alive_nodes()])
        # print(cluster_ids)

        # for i in cluster_ids:
        #     cluster_nodes = []
        #     for node in network.get_alive_nodes():
        #         if node.next_hop == i:
        #             cluster_nodes.append(node)
        #     scheduler = SleepScheduler(cluster_nodes = cluster_nodes)
    

    def plot_network(self, round_nb):
        """ Plots the network topology """
        for i, ntwk in enumerate(MLC.networks):
            plot_clusters(ntwk, save=True, filename=f"{get_result_path()}controller-{i}--round-{round_nb}-network-structure.png")

    def export_energy_map(self, round_nb):
        """ Plots the network topology """
        for i, ntwk in enumerate(MLC.networks):
            plot_clusters(ntwk, save=True, filename=f"{get_result_path()}controller-{i}--round-{round_nb}-network-structure.png")

    def calculate_average_energy(self, nodes):
        x = np.array([node.energy_source.energy for node in nodes])
        return x.mean()

