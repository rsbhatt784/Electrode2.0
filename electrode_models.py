"""
Electrode2
"""

import abc
from typing import Callable, Dict, Hashable

import networkx as nx # version == 2.4

import matplotlib.pyplot as plt # version == 3.3.2

import random

import numpy as np # version == 1.20.1

import pandas as pd # version == 1.2.1

import statistics

import copy

from tqdm import tqdm

__version__ = "0.2.0"
        
#===============================================================================================================================

class Neuron(abc.ABC):
    """
    A base class for Neuron objects.
    """
    # Because of inheritance, all neuron models must feature this function:
    
    def get_membrane_potential(self):
                ...
            
class HHNeuron(Neuron):

    def __init__(self, membrane_potential: float = -69.5, offset: float=0.00, prob_m: float=0.05, prob_n = 0.6, prob_h = 0.32,
                 capacitance: float= 1.0, g_Na: float = 120.0, g_K: float = 36.0, g_L: float = 0.3, E_Na: float = 45.5,
                E_K:float=-75.5, E_l: float = -59.1, input_current: float=7, time_step: float=0.01,threshold=20):
        
        # Initial conditions
        self._fire_callbacks=[]
        
        self.offset = offset # unique deviation from initial membrane potential, mV
        
        self.initial_membrane_potential = membrane_potential + self.offset # in mV
        
        self._membrane_potential_mV = self.initial_membrane_potential # mV
        
        # there should always be baseline current activity!!!!!!!!
        
        self._input_current = input_current # in microAmps/cm^2
        
        self.prob_m = prob_m # dimensionless probability of ion channel activation. Range: (0,1)
        
        self.prob_n = prob_n # dimensionless probability of ion channel opening. Range: (0,1)
        
        self.prob_h = prob_h # dimensionless probability of ion channel inactivation. Range: (0,1)
        
        self.membrane_capacitance = capacitance # in microFarad/cm^2
        
        self.Na_channel_cond = g_Na # conductance in Siemens/ square centimeters
        
        self.K_channel_cond = g_K # conductance in Siemens/ square centimeters
        
        self.Leak_channel_cond = g_L # conductance in Siemens/ square centimeters
        
        self.Na_chem_potential = E_Na # (Nernst) chemical potential of sodium in mV
        
        self.K_chem_potential = E_K # (Nernst) chemical potential of potassium in mV
        
        self.Leak_chem_potential = E_l # (Nernst) chemical potential of ions in leak channel in mV
        
        self.age = 0 # time in ms
        
        self.dt = time_step # how often (in ms) we record measurements. 
        
        self.duration = 500 # ms, simulation duration. This is a placeholder. Will change in the Experiment class
        
        self.threshold = threshold # in mV, above this point, the neuron will fire.
        
        # initial matrices for current and all values. These will have to go into the step function
        
        self.V=np.zeros(int(self.duration/self.dt)) # voltage

        self.m=np.zeros(int(self.duration/self.dt)) # gating variable, inactivation
        
        self.n=np.zeros(int(self.duration/self.dt)) # gating variable, activation
        
        self.h=np.zeros(int(self.duration/self.dt)) # gating variable, inactivation 
        
        #self.I= np.ones(int(self.duration/self.dt))*7 # channel current
        
        self.I_Leak=np.zeros(int(self.duration/self.dt)) # leak channel current
        
        self.I_Na=np.zeros(int(self.duration/self.dt)) # sodium channel current

        self.I_K=np.zeros(int(self.duration/self.dt)) # potassium channel current 
        
        self.conductance_K=np.zeros(int(self.duration/self.dt)) # conductance of potassium
        
        self.conductance_Na=np.zeros(int(self.duration/self.dt)) # conductance of sodium
        
        # initialize the values of these matrices
        
        #self.I[0]= self._input_current
        
        self.V[0]= self.initial_membrane_potential
        
        self.m[0]= self.alpha_m(self.V[0])/(self.beta_m(self.V[0])+self.alpha_m(self.V[0]))
        
        self.n[0]= self.alpha_n(self.V[0])/(self.beta_n(self.V[0])+self.alpha_n(self.V[0]))
        
        self.h[0]= self.alpha_h(self.V[0])/(self.beta_h(self.V[0])+self.alpha_h(self.V[0]))
        
        self.i = 1 # This counter should never change. It is used in the step function.
        
        self.t = np.arange(0,self.duration,self.dt)
        
        # model name
        
        self.model_name = 'HHNeuron'
        
        # count the spikes. All the time points where a spike was fired.
        self.spike_count=[]
        
    def alpha_m(self,V):

        return (2.5-0.1*(V-self.initial_membrane_potential))/(np.exp(2.5-0.1*(V-self.initial_membrane_potential))-1)

    def beta_m(self,V):

        return 4*np.exp((self.initial_membrane_potential-V)/18)

    def alpha_n(self,V):

        return (0.1-0.01*(V-self.initial_membrane_potential))/(np.exp(1-0.1*(V-self.initial_membrane_potential))-1)

    def beta_n(self,V):

        return 0.125*np.exp((self.initial_membrane_potential-V)/80)

    def alpha_h(self,V):

        return 0.07*np.exp((self.initial_membrane_potential-V)/20)

    def beta_h(self,V):

        return 1/(1+np.exp(3-0.1*(V-self.initial_membrane_potential)))

    def I_sodium(self, V, m, h):

        return self.Na_channel_cond*(m**3)*h*(V-self.Na_chem_potential)
        
    def I_potassium(self,V,n):

        return self.K_channel_cond*(n**4)*(V-self.K_chem_potential)
        
    def I_leak(self, V):

        return self.Leak_channel_cond*(V-self.Leak_chem_potential)
            
    def I_injected(self):

        return self._input_current

    def step(self):

        # These are the HH equations approximated by Euler's method.
        self.alpham=self.alpha_m(self.V[self.i-1])
        self.alphan=self.alpha_n(self.V[self.i-1])
        self.alphah=self.alpha_h(self.V[self.i-1])
        self.betam=self.beta_m(self.V[self.i-1])
        self.betan=self.beta_n(self.V[self.i-1])
        self.betah=self.beta_h(self.V[self.i-1])
        
        # Monitor how conductance changes in the leak channels.
        self.conductance_K[self.i-1] = self.K_channel_cond*(self.n[self.i-1]**4)
        self.conductance_Na[self.i-1]=self.Na_channel_cond*(self.m[self.i-1]**3)*self.h[self.i-1]
        
        # Compute the updated current through the channels
        self.I_Na[self.i-1] = self.conductance_Na[self.i-1]*(self.V[self.i-1]-self.Na_chem_potential)
        self.I_K[self.i-1] = self.conductance_K[self.i-1]*(self.V[self.i-1]-self.K_chem_potential)
        self.I_Leak[self.i-1] = self.Leak_channel_cond*(self.V[self.i-1]-self.Leak_chem_potential)
        
        self.Input = self.I_injected() - (self.I_Na[self.i-1] + self.I_K[self.i-1] + self.I_Leak[self.i-1])
        #print(self.I_injected())
        
        # Calculating the new V, m, n, and h variables 
        self.V[self.i] = self.V[self.i-1] + self.Input* self.dt*(1/self.membrane_capacitance)
        
        self.m[self.i] = self.m[self.i-1] + (self.alpham *(1-self.m[self.i-1]) - self.betam * self.m[self.i-1])*self.dt
        
        self.n[self.i] = self.n[self.i-1] + (self.alphan *(1-self.n[self.i-1]) - self.betan * self.n[self.i-1])*self.dt
        
        self.h[self.i] = self.h[self.i-1] + (self.alphah *(1-self.h[self.i-1]) - self.betah * self.h[self.i-1])*self.dt
        
        self._membrane_potential_mV = float(self.V[self.i])
        
        if self.V[self.i-1]<self.threshold<self.V[self.i]:
            
            self._fire()
            #print("We've fired an AP!!")
            self.spike_count.append(self.age)
            
        # advance the age of the neuron
        
        self.age+=self.dt
        
        # advance the index of the V,m,n,h matrices
        self.i+=1
        
        # there should always be a nonzero baseline current activity!!!!!!!! Or spikes will cease.
        
        self._input_current = 7
        
        return None
    
    def plot(self):
    
        HH_V_plot = plt.figure(figsize=(6,6))
        plt.plot(self.t,self.V,color='purple')
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (mV)')
        
        HH_n_plot = plt.figure(figsize=(6,6))
        plt.plot(self.t,self.n,color='maroon')
        plt.xlabel('time (ms)')
        plt.ylabel('Gating variable: n')
        
        HH_H_plot = plt.figure(figsize=(6,6))
        plt.plot(self.t,self.h,color='turquoise')
        plt.xlabel('time (ms)')
        plt.ylabel('Gating variable: h')
        
        HH_m_plot = plt.figure(figsize=(6,6))
        plt.plot(self.t,self.m,color='green')
        plt.xlabel('time (ms)')
        plt.ylabel('Gating variable: m')
        
        #print(self.i)
        
        return None
    
    def voltages(self):
        
        return self.V

    def _fire(self):

        for cb in self._fire_callbacks:
            cb(self)

    def add_fire_hook(self, cb: Callable):

        self._fire_callbacks.append(cb)
            
    def get_membrane_potential(self):

        return self._membrane_potential_mV

    def add_current(self, target_micro_amp):

        self._input_current += target_micro_amp

    def set_current(self, target_micro_amp):

        self._input_current = target_micro_amp

    def set_membrane_potential(self, target_mV):

        self._membrane_potential = target_mV
        
        
class LIAFNeuron(Neuron):
    """
    A leaky integrate-and-fire neuron.
    https://files.meetup.com/469457/spiking-neurons.pdf
    https://ocw.mit.edu/resources/res-9-003-brains-minds-and-machines-summer-course-summer-2015/tutorials/tutorial-2.-matlab-programming/MITRES_9_003SUM15_fire.pdf
    https://goldmanlab.faculty.ucdavis.edu/wp-content/uploads/sites/263/2016/07/IntegrateFire.pdf
    """

    def __init__(self, time_step: float = 0.01, initial_membrane_potential: float=-69.5, membrane_resistance: float=1e-1, 
                 capacitance: float = 100, input_current: float=7, refractory_period: float=4, 
                 spike_mV: float=10, spike_threshold: float=-20.0, time_constant: float=0.01, 
                 offset: float=0.0):
        
        # fire_callbacks will be a list of lambda functions that are executed only when 
        # a neuron has fired. See the add_synapse function in the Brain class for more details.
        self._fire_callbacks=[]
        
        # dt, the smaller the smoother the curve.
        self._time_step_size_ms = time_step # ms
        
        self.dt = self._time_step_size_ms # ms
  
        self._age = 0 # ms
    
        self.offset = offset # unique deviation from initial membrane potential # mV
    
        # Initial Membrane potential for all neurons
        self._membrane_potential_mV = initial_membrane_potential + self.offset # mV
        
        self._reset_membrane_potential_mV = initial_membrane_potential  #mV

        self._resistance_ohm = membrane_resistance # megaohms-cm^2
        
        self.membrane_capacitance = capacitance # in nanoFarad/cm^2

        # input_current is a part of the function dV (change in membrane potential) that gets added to the
        # membrane potential after every time step, dt. 
        
        self._input_current = input_current # microamps/cm^2
        
        # The amount of time it takes for the cell to recover after firing
        self._refractory_ms = refractory_period # ms
        
        # This represents how old the neuron will be (its self._age value/ the point in time from 
        # the beginning of the simulation) when the refactory period will be over.
        self._refractory_until_ms = self._refractory_ms # ms

        # Voltage required for an action potential to fire.
        self._spike_threshold_mV = spike_threshold # in mV
        
        # Amount of time it takes to charge the neuron (capacitor): tau = RC
        
        self._time_constant_ms = self._resistance_ohm*self.membrane_capacitance # s

        # voltage of action potential
        self._spike_mV = spike_mV # V
        
        # model name
        self.model_name = 'LIAFNeuron'
        
        # count the spikes. All the time points where a spike was fired.
        self.spike_count=[]
        
        self.all_membrane_potentials=[]
        

    def _fire(self):
        '''
        Execute all of the operations/functions specified in the list self._fire_callbacks! This function is called
        when the neuron fires an AP.
        
        Parameters
        ----------
        None
        '''
            
        #cb is a callable function
        #Execute all of the cbs in the list _fire_callbacks!
        #This will pump the specified amount of current from the pre-synaptic neuron to the post-synaptic neuron.
        for cb in self._fire_callbacks:
            cb(self)

    def add_fire_hook(self, cb: Callable):
        '''
        Utilized in the Brain Class, this function allows every neuron to have a specified response to an action potential.
        The response comes in the form of adding current to its postsynaptic neurons. For further reference, see the 
        add_synapse function in the Brain Class.
        
        Parameters
        ----------
        cb is a lamda function. It tells the neuron what to do when an action potential is fired.
        '''
        
        #Look at the add_synapse function. Callbacks can be lambda functions. They are executed 
        #synchronously (immediately) or asynchronously (at a later time) with the function that calls them.
        #This function just adds the callable functions to the list so that they are all executed come time.
        
        self._fire_callbacks.append(cb)

    def step(self):
        '''
        Advance the LIAF simulation by taking a time step, dt. All pertinent variables (i.e age, membrane potential, current,
        etc.) will be updated. Note that this time step function is unqiue to the LIAF model.
        
        Parameters
        ----------
        None
        '''
        
        #Note, LIF model says that dv/dt = (-v + IR)/tau
        #Therefore dv, the change in membrane potential is given by dv=[(-v + IR)/tau]*dt
        
        # Prior to an action potential: 
            
        # if the neuron has passed its refractory period, do the following: 
        if self._refractory_ms < self._refractory_until_ms:
            # V = V*tau/dt + reset
            self._membrane_potential_mV = (
                (
                    (self._membrane_potential_mV * (self._time_constant_ms/self._time_step_size_ms)) + self._reset_membrane_potential_mV
                    + (self._input_current * self._resistance_ohm*1000)
                )
                / (self._time_constant_ms/self._time_step_size_ms + 1)
            )
            if self._membrane_potential_mV >= self._spike_threshold_mV:
                
                ## How come _spike_mV is not added to the membrane potential?
                self._membrane_potential_mV = self._spike_mV
                self._refractory_until_ms = 0
                self._fire()
                self.spike_count.append(self._age)
                
        # If the neuron is still in its refractory period, then no change in voltage will occur. Fix voltage at 0,
        # or the initial voltage.
        else:
            self._membrane_potential_mV = self._reset_membrane_potential_mV
            # If an action potential fires, compute the following:
            self._refractory_until_ms = self._refractory_until_ms + self._time_step_size_ms
        # Since input amp in integrated, set to 0
        self._input_current = 1 # in microAmps/cm^2
        # Increase the age of the neuron as the time elapses.
        self._age += self._time_step_size_ms
        
        self.all_membrane_potentials.append(self._membrane_potential_mV)  #in mV

    def get_membrane_potential(self):
        '''
        Obtain the membrane potential of the neuron at any given time.
        
        Parameters
        ----------
        None
        '''
        return self._membrane_potential_mV

    def add_current(self, target_micro_amp):
        '''
        Parameters
        ----------
        target_micro_amp: The amount of current that a presynaptic neuron pushes onto its post_sysnatpic partner.
        '''
        # Should this be reset to zero after the neuron has fired?
        self._input_current += target_micro_amp

    def set_current(self, target_micro_amp):
        '''
        Assign a certain value neuron's current input.
        
        Parameters
        ----------
        target_micro_amp: The value of current given to a neuron
        '''
        self._input_current = target_micro_amp

    def set_membrane_potential(self, target_mV):
        '''
        Pin the neuron's membrane potential to a certain value.
        
        Parameters
        ----------
        target_mV: The voltage that the neuron's membrane potential will be pinned to.
        '''
        self._membrane_potential_mV = target_mV
        
        
class Synapse:
    """
    An abstract Synapse object that manages connectivity of Neurons.
    """ 

    def __init__(self,u,v):
        '''
        Instantiating the presynaptic, u, and postsynaptic, v, neuron.
        
        Parameters
        ----------
        u: presynaptic neuron
        v: postsynaptic neuron
        '''
        self.u = u
        self.v = v


class BasicCurrentSynapse(Synapse):
    
    def __init__(self, u, v, input_current: float = 0.1, edge_weight: float = 0.0, incorporate_weights: bool = False):
        '''
        Instantiate a standard synapse between two neurons.
        
        Parameters
        ----------
        u: name of the presynaptic neuron.
        
        v: name of the postsynaptic neuron.
        
        edge_weight: value of the weight associated with the synapse. If not specified, the synapse will be assigned a default edge_weight of 1.0. 
        
        input_current: the amount of current transmitted across the synapse (in microAmps/cm^2).
        
        incorporate_weights: Determines whether or not current will be scaled by its weight. 
        
        '''
        
        # Include the variables of the Synapse.__init__ function.
        Synapse.__init__(self, u, v)
        
        # Input_current specifies how much current a pre-synaptic neuron
        # will transmit to its post-synaptic partner(s).
        self._current = input_current #uA/ cm^2
        
        # Represents the physical number, and type, of syanpses that exist b/w two neurons 
        # in the context of the C. elegans connectome. User may define it differently. 
        self._edge_weight = edge_weight
        
        # Whether user wants to apply an edge_weight to the synapse or leave it at 0. 
        self.incorporate_weights = incorporate_weights
            
        if self.incorporate_weights:
            self._current = self._current * self._edge_weight

    def check_u(self):
        '''
        Look at the presynaptic neuron
        
        Parameters
        ----------
        None
        '''
        print(self.u)
    
    def check_v(self):
        '''
        View the postsynaptic neuron
        
        Parameters
        ----------
        None
        '''
        print(self.v) 

    def get_output_current(self):
        '''
        Returns the amount of current that a presynaptic neuron will send to its postsynaptic partner.
        
        Parameters
        ----------
        None
        '''
        return float(self._current)


class Electrode:
    """
    An electrode can read and write to a cell. 
    Target neuron refers to the neuron with the electrode placed on it.
    By adding an electrode to a specific target neuron, the user can record and track the activity
    of the neuron throughout an experiment. 
    Adding an electrode is equivalent to Brian2's usage of the record parameter in the State Monitor.
    """
            
    def __init__(self, target: Neuron) -> None:
        self._target = target

    def measure(self) -> float:
        """
        Get membrane potential of the target neuron.
        """
        return self._target.get_membrane_potential()

    def pin(self, target_mV: float) -> None:
        """
        Pin to a target mV.
        This function stimulates neurons.
        
        """
        self._target.set_membrane_potential(target_mV)

    def inject(self, target_I: float) -> None:
        """
        This function stimulates neurons.

        """
        self._target.add_current(target_I)
        
        
class Brain():
    """
    A base class for emulating a "brain" structure using an underlying NetworkX Digraph. 
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Create a new Brain object.
        
        This will create an empty graph unless
        a nx graph is passed in as a kwarg.
        
        net = Network()
        
        or 
        
        net = Network(graph = nx.read_graphml('c_elegans_control.graphml'))
        
        Parameters
        ----------
        **kwargs : 
            graph:  networkx graph
            
            model:  default neuron model
            
            input_current: For a pre-existing graph, specify what current to propagate onto postsynaptic neurons
            
            incorporate_weights: If True, the synapse input_currents will be scaled by their (normalized) weights. 
            
            whole_brain: If True, all of the neurons in a brain will follow the same neuron model. If false, 
            
            initialv_offset: the maximum value of voltage that a neuron's initial_membrane potential can deviate from the base of -69.5 mV.  
            
            neuron_dict: a dictionary with keys as neurons and values as the neuron model.
            
            random_model: True, if user wants random assignment of the neuron_model.
            
            _print_message: trivial parameter implemented to regulate print messages.
            
        """
                
        # Dictionary with keys as neuron names and values as their object instance. This houses all of the neurons in the
        # final connectome.
        self.neurons = {}
        
        # Dictionary with keys as neuron names and values as the name of their neuron model (i.e Hodgkin Huxley or
        # Leaky Integrate and Fire)
        self.neuron_models={}
        
        # Dictionary with the synapse name (pre, post) as keys and the synapse instance as a value.
        # Additionally, can be used to check edge_weight. 
        self.synapses = {}
        
        #Dictionary containing the synapses and their associated weights.
        self.edge_weights = {}
        
        # Dictionary of all the lambda _fire_callbacks functions.
        # Keys are the synapses (pre,post) and values are the lambda functions.
        self.synapse_callbacks_functions = {}
        
        # List of tuples, (x,y) where x is the neuron name and y is the neuron's electrode instance.
        self.Electrodes=[]

        """
        Dictionaries that will be used for the ablation and undo_ablation functions. 
        """
        
        # Dictionary for storing ablated neurons for a neuron recovery system. 
        # Neuron name as the key and neuron object instance as the value.
        self.neuron_recovery_dictionary = {}
        
        # Dictionary for storing ablated or removed synapses. 
        # Synapse name (pre, post) as the key and synapse object instance as the value.
        self.synapse_storage={}
        
        # Dictionary for storing the relationship between ablated neurons and their synapses
        # which are subsequently ablated. 
        # Ablated neuron name as the key and a list of its ablated synaptic partners as value. 
        self.connections = {}        
        
        
        """ 
        Initialization of Brain object. 
        """ 
        
        #Keyword arguments/parameters of the Brain.
        self.kwargs = kwargs

        # This parameter will be recruited in the Experiment class. 
        # When we instantiate a perturbed brain, the Experiment class will need to know whether 
        # we ever called the add/remove neuron or synapse functions. 
        # By default, it's False.
        self.modified=False 

        if 'initialv_offset' not in kwargs:
            kwargs['initialv_offset']=1 # in mV
        
        if 'graph' not in kwargs.keys():
            print('Creating new nx.DiGraph(). Add neurons and synapses to build the brain.')
            self._graph = nx.DiGraph()
            
        else: 

            if 'whole_brain' not in kwargs.keys():
                print("ERROR: Must specify value for whole_brain parameter. No Brain nx.DiGraph initialized.")
                return
            else:
                # Here, we are attempting to create a homogeneous brain.
                if kwargs['whole_brain']: 
                    if 'model' not in kwargs.keys():
                        print("ERROR: Must specify model-type for whole brain. No Brain nx.DiGraph initialized.")
                        return
                    else: 
                        self._graph = nx.DiGraph(kwargs['graph'])
                        # Values is a dictionary of dictionaries. 
                        # Each dictionary holds a set of attributes for each neuron. 
                        values={}
                        for node in self._graph.nodes():
                            if kwargs['model']=='LIAFNeuron': 
                                self.add_neuron(node, neuron = 'LIAFNeuron', add_to_connectome=False)
                            elif kwargs['model']=='HHNeuron':
                                self.add_neuron(node, neuron = 'HHNeuron', add_to_connectome=False)
                            else: 
                                print("Unknown model-type. Neuron ", node, " was not added.")
                                continue 
                       
                        for neuron in self.neurons:
                            values[neuron] = {'neuron': self.neurons[neuron]} 
                        nx.set_node_attributes(self._graph, values)
                        
                else: 
                    values={}
                    
                    # Randomly assign model type for each neuron.
                    if 'random_model' in kwargs.keys():
                        self._graph = nx.DiGraph(kwargs['graph'])
                        for node in self._graph.nodes():
                            self.add_neuron(node, random_model=True, add_to_connectome=False, initialv_offset=kwargs['initialv_offset'])
                    
                    # Specify model type for each neuron.  
                    elif 'neuron_dict' in kwargs.keys():
                        self._graph = nx.DiGraph(kwargs['graph'])
                        for neuron in kwargs['neuron_dict']: 
                            
                            # Checks if neuron is a node that already exists in self._graph before 
                            # allowing user to specify its model-type via the dictionary (in any order). 
                            if neuron in self._graph.nodes():
                                if kwargs['neuron_dict'][neuron]=='LIAFNeuron':
                                    self.add_neuron(neuron, neuron = 'LIAFNeuron', initialv_offset=kwargs['initialv_offset'], add_to_connectome=False)
                                elif kwargs['neuron_dict'][neuron]=='HHNeuron':
                                    self.add_neuron(neuron, neuron = 'HHNeuron', initialv_offset=kwargs['initialv_offset'], add_to_connectome=False)
                                else: 
                                    print("Unknown model-type. Neuron " + neuron + " was not added.")
                                    continue 
                            else: 
                                print("Neuron, " + neuron + " in dictionary is not in Graph. Must add neuron separately.")
                                continue
                              
                        for neuron in self.neurons:
                            values[neuron] = {'neuron': self.neurons[neuron]} 
                        nx.set_node_attributes(self._graph, values)
                            
                    else:
                        print("Must specify dictionary of neurons and their model type. No Brain nx.DiGraph initialized.")
                        return
       
          
            # Checking if the graph has edge_weights as an attribute before adding them to a dictionary. 
            # Temporary dictionary for storing edge weights.
            for synapse in list(self._graph.edges.data()):
                pre,post,dictionary = synapse
                if 'weight' in dictionary:
                    self.edge_weights[(pre, post)] = float(dictionary['weight'])

                else: 
                    self.edge_weights[(pre, post)] = 0.0

        
            # Creating synapses for all of the edges specified in the given digraph.  
            # NOTE: add_to_connectome = False here since these synpases already exist as edges 
            # of the graph inputted. 
            for i,j in self._graph.edges:
                if 'input_current' in kwargs:
                    if 'incorporate_weights' in kwargs:
                        self.add_synapse(i, j, input_current = kwargs['input_current'], edge_weight = self.edge_weights[(i, j)], incorporate_weights = kwargs['incorporate_weights'], add_to_connectome = False)
                    else:
                        self.add_synapse(i, j, input_current = kwargs['input_current'], add_to_connectome = False)      
                else:
                    if 'incorporate_weights' in kwargs:
                        self.add_synapse(i, j, incorporate_weights = kwargs['incorporate_weights'], edge_weight = self.edge_weights[(i, j)], add_to_connectome = False)
                    else:
                        self.add_synapse(i, j, add_to_connectome = False)
            
            
            # Normalizing edge_weights to be a value between 0 and 1.
            max_weight = max(self.edge_weights.values())
            min_weight = min(self.edge_weights.values())
            

            for synapse, weight_value in self.edge_weights.items():
                # normalized = (weight_value-min_weight)/(max_weight-min_weight)
                normalized = weight_value/max_weight
                self.update_edge_weight(synapse=synapse, new_weight=normalized)

            #for synapse, weight_value in self.edge_weights.items():
                #normalized = (weight_value-min_weight)/(max_weight-min_weight)
                #self.update_edge_weight(synapse=synapse, new_weight=normalized)
       
    
        # Make a copy of the original items.                
        self.original_graph=self._graph.copy()

        
    def add_neuron(self,name: Hashable, neuron: Neuron='LIAFNeuron', random_model: bool = False, initialv_offset: float = 1, add_to_connectome: bool = True) -> Hashable:
        """
        Add a neuron to the network.
        
        Parameters
        ----------
        name: name is the label you set for the neuron. It can be a string or integer name (Brian2 prefers integer)
        
        neuron: explicitly assign a neuron model to the neuron. i.e., LIAFNeuron() or Brian2Neuron()
        
        random_model: True, if user wants random assignment of the neuron_model.
        """
        
        if random_model is True:
            possible_models = ['LIAFNeuron','HHNeuron','LIAFNeuron','HHNeuron',
                           'LIAFNeuron','HHNeuron','LIAFNeuron','HHNeuron','LIAFNeuron']
            random.shuffle(possible_models)
            neuron = possible_models[random.randint(0, len(possible_models)-1)]
            
        
        if name in self.neurons.keys():
            print('{} neuron has already been added.'.format(name))
            return 

        if neuron=='LIAFNeuron':
            self.neuron_models[name] = 'Leaky Integrate and Fire' 
            self.neurons[name] = LIAFNeuron(offset=initialv_offset*float(np.random.rand(1)))
            if add_to_connectome: 
                self._graph.add_node(name, neuron=self.neurons[name])
            self.modified=True
                
        elif neuron=='HHNeuron':
            self.neuron_models[name]='Hodgkin Huxley'
            self.neurons[name] = HHNeuron(offset=initialv_offset*float(np.random.rand(1)))
            if add_to_connectome: 
                self._graph.add_node(name, neuron=self.neurons[name])
            self.modified=True
            
        else: 
            print('Need to specify HHNeuron or LIAFNeuron. Currently do not offer other model-types.') 
            return 
        

    def add_synapse(self, u: Hashable, v: Hashable, input_current: float = 0.1, incorporate_weights: bool = False, edge_weight: float = 1.0, add_to_connectome: bool = True) -> None:
        '''
        Add a synapse between two neurons in the connectome.
        
        Parameters
        ----------
        u = presynaptic neuron name
        
        v = postsynaptic neuron name
        
        input_current = the amount of current (in microAmps) transmitted across the synapse
        
        incorporate_weights = If True, the current going through a synapse will be scaled by the strength of the synaptic (edge) weight. Otherwise, synapses will all be treated with the same strength.
        
        edge_weight = the weight given to a synapse based on an existing graph-specified attribute or given by the user. 
        
        add_to_connectome = This parameter ensures that the synapse gets incorporated into the final connectome. DO NOT change it to False. The only time it should be False is in the __init__ function of the brain, if the brain is instantiated with a ready-made connectome that already features these synapses.
        '''
        
        # First check if the neurons the user wants to connect exist.
        if u and v not in self.neurons.keys():
            print('Hi. Either neuron {} or {} does not exist in the network. Please add them before attempting to connect them.'.format(u,v))
            return
        
        if (u,v) in self.synapses.keys():
            print('Synapse between neurons {} and {} already exist'.format(u,v))
            return
        
        synapse = BasicCurrentSynapse(u, v, input_current, edge_weight, incorporate_weights)
        self.synapses[(u,v)] = synapse
        if add_to_connectome: 
            self._graph.add_edge(u, v, synapse=synapse) 
        self.modified = True
                                  
        # Create the function that will be executed when an AP is fired.
        fire_callback_func = lambda u: self.get_neuron_instance(v).add_current(synapse.get_output_current())
        self.get_neuron_instance(u).add_fire_hook(fire_callback_func)
        self.synapse_callbacks_functions[(u,v)]=fire_callback_func
    
    
    def view_nodes(self, all_data=False):
        
        '''
        See which nodes are present in the graph at any time.
        
        Nodes are given as the integer names from Brain.mapping dictionary.
        
        Parameters
        ----------
        all_data = Boolean. If all_data = True, function will also supply a dictionary of the node's attributes.
        '''
        
        if all_data is False:
            return self._graph.nodes(data=False)
        else:
            return self._graph.nodes(data=True)
        
        
    def view_node_attributes(self, neuron: Hashable):
        '''
        View the attributes for a given neuron.
        
        Parameters
        ----------
        neuron: neuron name
        '''
        
        if neuron in self.neurons.keys():
            return self._graph.nodes(data=True)[neuron]
        else: 
            print('{} does not exist in the connectome.'.format(neuron))
            return
           
    def modify_node_attributes(self, **kwargs) -> None:
        '''
        Add, Delete, or Change an Attribute for a Given Neuron.
        
        Parameters
        ----------
        **kwargs : 
            neuron:  neuron name
            action:  whether user wants to change, add, delete the attribute
            attribute: attribute name
            value: value of the attribute which user wants to change or add
        '''
        
        if 'neuron' not in kwargs.keys():
            print('Must enter neuron name')
            return 
        
        if kwargs['neuron'] not in self.neurons.keys():
            print('{} does not exist in the connectome.'.format(neuron))
            return
        else:
            if 'action' not in kwargs.keys():
                print('Must enter action argument.')
                return 
            if kwargs['action'].casefold() == 'add' or kwargs['action'].casefold() == 'change':
                if 'attribute' or 'value' not in kwargs.keys():
                    print('Must enter attribute and the value to be assigned to that attribute.')
                    return
                self._graph.nodes[kwargs['neuron']][kwargs['attribute']] = kwargs['value']
            if kwargs['action'].casefold() == 'delete':
                del self._graph.nodes[kwargs['neuron']][kwargs['attribute']] 
        
    
    def view_edges(self, all_data=False):
        
        '''
        Retrieve the edges (pre, post) in the connectome.
        
        Parameters
        ----------
        all_data = Boolean. If True, edges and edge weights will be displayed. Otherwise, false, and only edges will be shown.
        ''' 

        if all_data is False:
            return self._graph.edges(data=False)
        else:
            return self._graph.edges(data=True)          

        
    def view_neurons(self) -> Dict[Hashable, Neuron]:
        
        '''
        Returns a dictionary with keys as the neuron name and values as the neuron object instance.
        Parameters
        ----------
        None.
        '''
        return self.neurons
                
  
    def view_synapses(self) -> Dict[Hashable, Synapse]:
        
        '''
        Returns a dictionary with keys as the synapse name and values as the synapse object instance.
        
        Parameters
        ----------
        None.
        
        '''
        return self.synapses
         

    def get_neuron_instance(self, neuron: Hashable) -> Neuron:
        
        '''
        Function will return the neuron object instance associated with the supplied neuron.
        
        Parameters
        ----------
        neuron = the instantiated or user-given name of the neuron.
        '''
        if neuron in self.neurons.keys():
            return self.neurons[neuron]
        else: 
            print('{} does not exist in the connectome.'.format(neuron))
            return 
    
    
    def get_synapse_instance(self, synapse: tuple) -> Synapse:
        
        '''
        Return a dictionary with keys as the synapse name and values as the synapse object instance.
        
        Parameters
        ----------
        synapse = the pair of neurons that make up a given synapse. 
       
        '''
        if synapse in self.synapses.keys():
            return self.synapses[synapse]
        else: 
            print('{} does not exist in the connectome.'.format(synapse))
            return 
        
     
    def get_neuron_models_dict(self) -> Dict[Hashable, Neuron]:
        
        '''
        Creates a dictionary with the key being the user given neuron name and the value being 
        the name of the neuron model it follows, i.e LIAF or HH.
        
        Parameters
        ----------
        None
        '''
        return self.neuron_models
    
   
    def get_neuron_model(self, name: Hashable) -> Neuron:
        
        '''
        Function will return the neuron object instance associated with the supplied neuron.
        
        Parameters
        ----------
        name = the user-given name of the neuron, not the neuron Model.
        '''
        
        return self.neuron_models[name]


    def has_synapses(self, neuron: Hashable,_print_message: bool=True):  
        
        '''
        Returns all of the synapses a specific neuron has.
        
        Parameters
        ----------
        neuron = name of the neuron whose synapses will be returned. 
        
        _print_message = True, if you would like the following print statements to be given. This is given because the function will be called in other areas of the code where these print statements are not necessarily needed.
        '''
        if neuron not in self.neurons.keys():
            print('{} does not exist in the connectome.'.format(neuron))
            return
        
        total_synapses=[]
        for synapse in self.synapses.keys():
            u,v = synapse
            if u == neuron or v == neuron:
                total_synapses.append(synapse) 
        
        if len(total_synapses)==0:
            print("There are no synapses registered with this neuron.")
            return 
        else: 
            return total_synapses        
        
    
    def view_edge_weight(self, synapse: tuple):
        
        '''
        Return the edge weight for a given synapse.
        
        Parameters
        ----------
        synapse = tuple (pre,post) that represents the synapse.
        
        '''
        if synapse in self.synapses.keys():
            return self.synapses[synapse]._edge_weight
        else: 
            print('The synapse {} does not exist in the connectome.'.format(synapse))
            return 
        

    def update_edge_weight(self, synapse: tuple, new_weight):
        
        '''
        Update the edge weight for a given synapse.
        
        Parameters
        ----------
        synapse = tuple (pre,post) that represents the synapse.
        
        new_weight = value of the new synaptic weight.
        
        '''

        if synapse in self.synapses.keys(): 
            self.synapses[synapse]._edge_weight = float(new_weight)  
            self.edge_weights[synapse] = float(new_weight)
            return
        else: 
            print('The synapse {} does not exist in the connectome.'.format(synapse))
            return 
        

    '''
     Functions to be called in the Experiment Class that are related to ablations, undo_ablations, and applying electrodes. 
    '''
    '======================================================================================================================' 
    
    def ablate_neurons_by_non_numerical_attribute(self,**kwargs)-> None:
        
        '''
        Ablate synapses that have attributes with non-numerical values. e.g., group, type etc.
        
        Parameters
        ----------
        kwargs:
        attribute_value: specify what non-numerical attribute value you would like to ablate. e.g., 'inter'
        selected_attribute: Attribute to ablate by e.g., 'type'
        
        '''
       
        if 'selected_attribute' not in kwargs.keys():
            print('Please enter attribute name.')
            return
        if 'attribute_value' not in kwargs.keys():
            print('Please enter attribute value to ablate neurons by.')
            return
    
        selected_attribute=kwargs['selected_attribute']          
        attribute_value=kwargs['attribute_value']

        nodes_to_ablate=[]
        for outer_dictionary in self._graph.nodes.data():
            neuron,attribute_dictionary = outer_dictionary
            for attribute in attribute_dictionary.keys():
                if attribute == selected_attribute:
                    if attribute_dictionary[attribute] == attribute_value:
                        nodes_to_ablate.append(neuron)
        
        if len(nodes_to_ablate)==0:
            print('No nodes were ablated. Check if selected attribute or value exists.') 
            return 
                  
        self.remove_neuron(nodes_to_ablate)
        print('You have ablated:{}'.format(nodes_to_ablate))                    
         
    
    def ablate_neuron_by_numerical_attribute(self,**kwargs):
        
        '''
        Ablate neuron based on an attribute that has a numerical value. e.g., SomaPosition, outD etc.
        
        Parameters
        ----------
        kwargs:
            attribute: Attribute to ablate by e.g., 'AYNbr'
            
            threshold: If attribute has a numerical value, what threshold value is the cut-off for ablations? e.g., 43
            
            level: Ablate attributes that are Greater than, Less than, or Equal to specified threshold value.
            
        '''
        if 'attribute' not in kwargs.keys():
            print('Please enter attribute name.')
            return
        if 'threshold' not in kwargs.keys():
            print('Please enter attribute threshold to ablate neurons by.')
            return
        if 'level' not in kwargs.keys():
            print('Please enter the level of the attribute value/threshold you want to ablate the neurons by.') 
            return 
        
        attribute=kwargs['attribute']          
        threshold=kwargs['threshold']
        level=kwargs['level']

        nodes_to_ablate=[]
        for outer_dictionary in self._graph.nodes.data():
            neuron,attribute_dictionary = outer_dictionary
            for attr in attribute_dictionary.keys():
                if attr == attribute:
                    if level == 'Equal' and attribute_dictionary[attr] == threshold:
                        nodes_to_ablate.append(neuron)
                    if level == 'Greater than' and attribute_dictionary[attr] > threshold:
                        nodes_to_ablate.append(neuron)
                    if level == 'Less than' and attribute_dictionary[attr] < threshold:
                        nodes_to_ablate.append(neuron)                    
    
        if len(nodes_to_ablate)==0:
            print('No nodes were ablated. Check if selected attribute or value exists.') 
            return 
                  
        self.remove_neuron(nodes_to_ablate)
        print('You have ablated:{}'.format(nodes_to_ablate)) 
    
    
    def remove_neuron(self, neuron_list:list):
        
        '''
        Remove a neuron from the network
        
        Parameters:
        ----------
        neuron_list: A list of neurons that will be deleted.
        '''
        self.modified=True
        
        # Most importantly, we will delete the instance of the neuron object altogether. This will ensure that
        # the neuron never receives or transmits any input. 
        
        for neuron in neuron_list:
            if neuron not in self.neurons.keys():
                print('The neuron {} is already absent from the connectome.'.format(neuron))
                return 

            for name, instance in self.neurons.copy().items():
                if neuron==name:
                    saved_neuron = instance
                    self.neuron_recovery_dictionary[neuron] = saved_neuron
                    # All fire_hooks on the neuron should be removed once we delete the neuron object instance. 
                    # That way, the neuron can neither give nor receive signal.      
                    # deleting the neuron_instance will not affect the saved_neuron copy we had made.
                    self.neurons.pop(neuron, None)
                    self._graph.remove_nodes_from([neuron])


            ''' For every synapse that a neuron is a part of, we must ablate that synapse. ''' 

            # Here we will store all of the synapses that will be ablated due to the neuron's deletion.
            connections_jeopardized =[]
            # Here we will store the name of the synaptic partners that will no longer form a synapse 
            # with the ablated neuron.
            synaptic_partners_jeopardized = []

            # Check to see which synapses involve the neuron we wish to delete. 
            # The neuron can either be pre or postsynaptic.
            for synapse, synapse_instance in self.synapses.copy().items():
                if synapse_instance.u==neuron or synapse_instance.v==neuron:
                    # The following code ensures that we are correctly storing the ablated synapses in
                    # the connections list.
                    connections_jeopardized.append(synapse_instance)

                    # Add the jeopardized synaptic partners to the synaptic_partners_jeopardized_list
                    if neuron == synapse_instance.u:
                        synaptic_partners_jeopardized.append(synapse_instance.v)
                        # When we deleted this neuron instance, its callbacks function also deleted, as we'd like.
                        # Let's remove the synapse's record from the dictionary self.synapse_callbacks_functions.

                    elif neuron == synapse_instance.v:
                        synaptic_partners_jeopardized.append(synapse_instance.u)
                        # Here, the callbacks function needs to be deleted from the presynaptic neuron, u.
                        # Since our ablated neuron is postsynaptic in this case, we have to execute the following code

                        # Access the callbacks function associated with this synapse
                        # call_func = self.synapse_callbacks_functions[(synapse_instance.u,synapse_instance.v)]

                        # Drop the callback function from the presynaptic neuron's ._fire_callbacks list. In
                        # this case, that neuron is not the postsynaptic neuron we ablated.
                        # self.get_neuron_model(synapse_instance.u)._fire_callbacks.remove(call_func)

                    # Officially remove the synapses from the brain.
                    self.remove_synapse([(synapse_instance.u,synapse_instance.v)],_print_message=False)

            # store the synapse_instances that were deleted in the self.connections dictionary. We will use this for the 
            # neuron recovery function.
            self.connections[neuron] = connections_jeopardized
            # print(connections_jeopardized) 

            print('As a reminder, ablating {} will inadvertently disrupt all of its synapses. \n\nConsequently, the synapses between {} and the following list of neurons: \n\n{} \n\n are ablated too.'.format(neuron,neuron,synaptic_partners_jeopardized)) 



    def remove_synapse(self, edge_list: list, _print_message=True):                                
        '''
        Remove a synapse from the Connectome.
        
        Parameters
        ----------
        Edge_list: a list of tuples. Each tuple (u,v) comprises the pre and postsynaptic neurons of the synapse.
        
        _print_message: Specify True or False if you would like the software to notify you of the synapses status.
        '''
        self.modified=True
        
        # We must delete the synapse object to ensure that no current or signal is transmitted between neurons u and v.
        # We must also delete the callbacks on the PRESYNAPTIC neuron alone.
        
        for edge in edge_list:
            pre, post = edge
            if edge not in self.view_edges():
                if _print_message is True:
                    print('The synapse between {} and {} was already absent from the connectome'.format(pre, post))
            else:
                # Otherwise, we can remove the edge from the physical connectome.                        
                self._graph.remove_edges_from([edge])
                                        
            # We find the synapse instance corresponding with the edge, then we erase that instance altogether                           
            for synapse, synapse_instance in self.synapses.copy().items():

                if synapse_instance.u==pre and synapse_instance.v==post:
                    
                    # save the synapse instance and store it in the synapse_storage dictionary
                    saved_synapse_instance = synapse_instance
                    self.synapse_storage[edge] = saved_synapse_instance
            
                    # Next, we need to eliminate the fire_callbacks function related to the presynaptic neuron.
                    
                    # access the callback we want to remove from the presynaptic neuron (u)
                    func = self.synapse_callbacks_functions[(synapse_instance.u,synapse_instance.v)]
                    
                    # checking to see if presynaptic neuron was not already ablated. If it was not, then we can remove
                    # its callback function. 
                    if synapse_instance.u in self.view_neurons():
                        
                        # Remember, the presynaptic neuron (if ablated) already lost its callback function.
                        # In this case, we are removing the callback for the unablated neuron of the synapse.
                        self.get_neuron_instance(synapse_instance.u)._fire_callbacks.remove(func)
                    
                        # Now remove the callback from the dictionary synapse, self.synapse_callbacks_functions.
                        self.synapse_callbacks_functions.pop((synapse_instance.u,synapse_instance.v),None)
                    
                    else:
                        # if the presynaptic neuron was already ablated, its callback 
                        # RECORD in the self.synapse_callbacks_functions dictionary. still needs to be removed.
                        # Now remove the callback from the dictionary synapse, self.synapse_callbacks_functions.
                        self.synapse_callbacks_functions.pop((synapse_instance.u,synapse_instance.v),None)
                        None
                        
                    # Delete the record of the synapse from the synapses dictionary.
                    # This will not affect the saved_synapse_instance variable we saved earlier.
                    self.synapses.pop(synapse,None)   
        
        # Fortunately, deleting a synapse does not coincide with a neuron deletion. Though, the reverse is true.


    def undo_neuron_removal(self,neuron_list:list, recover_synapses: bool=True):
        '''
        Retrieve a neuron that had previously been removed from the connectome.
        
        Parameters
        ----------
        neuron_list: list of the name of the neuron(s) deleted.
        
        recover_synapses: True if the user would like to re-establish the synapses that were ablated when the neuron was ablated. If False, only the neurons will be reinstituted, and not their original synapses.
        '''
        self.modified=True
        
        # Recovering neurons
    
        for neuron in neuron_list:
                
            if neuron not in self.neuron_recovery_dictionary.keys():
                if neuron in self.neurons.keys():
                    print('{} currently exists in the connectome, therefore it will not be reintroduced'.format(neuron))
                else:
                    print('{} was never added into the connectome. Please revisit the add_neuron function if you would like to introduce the neuron {} into the connectome.'.format(neuron, neuron))
            else:
                neuron_instance = self.neuron_recovery_dictionary[neuron]
                
                # add the neuron back into the connectome. We call the add_neuron function and set the parameters 
                # to have the same values as the original ablated neuron.
                self.add_neuron(neuron, neuron_instance.model_name, neuron_instance.offset)
                
                # Now we can remove the neuron from the neuron_recovery_dictionary
                self.neuron_recovery_dictionary.pop(neuron, None)
                
                # Recovering Synapses
                if recover_synapses == True:
                    # Extracting the ablated synapse instances
                    synapse_instances_to_recover = self.connections[neuron]
                    # Re-adding each synapse into the connectome (brain)
                    for synapse_instance in synapse_instances_to_recover:
                    
                        self.add_synapse(synapse_instance.u, synapse_instance.v, synapse_instance._current, synapse_instance._edge_weight, synapse_instance.incorporate_weights)
                
                    # Now, let's remove these neuron, synapse_instances from the connections dictionary since we are finished
                    # with them.
                
                    self.connections.pop(neuron,None)
                
                elif recover_synapses == False:
                    pass   
 

    def undo_synapse_removal(self, edge_list: list):
        '''
        Provide the list of synaptic connections that you would like to reintroduce.
        Parameters
        ----------
        List of tuples, (pre,post). These are the edges that the user would like to reintegrate.
        '''
        self.modified=True
        
        # make sure that the edges given were actually deleted from the connectome before proceeding.
        
        for edge in edge_list:
            pre, post = edge
            if edge not in self.synapse_storage.keys():
                if self.get_synapse_instance(pre,post) is not None:
                    print('The synapse between {} and {} was never ablated. It still exists in the connectome.'.format(pre,post))
                else:
                    print('The connection between {} and {} was never established in the connectome. \nIf desired, add this synapse to the connectome using the add_synapse function.'.format(pre,post))
                    
            else:
                # Locate the synapse instance in the synapse_storage dictionary by using the edge key.
                synapse_instance = self.synapse_storage[edge]
                
                # This function will reinstate the synapse object instance and reintroduce the edge to the connectome.
                self.add_synapse(synapse_instance.u, synapse_instance.v, synapse_instance._current, synapse_instance._edge_weight, synapse_instance.incorporate_weights)
                
                # Finally, we can remove the edge from the synapse_storage dictionary.
                self.synapse_storage.pop(edge,None)
               
    
    def add_electrode(self, location: Hashable) -> Electrode:
        """
        Add an electrode targeted to the specified neuron.
        
        Parameters
        ----------
        location: user-given neuron name.
        """
        # The electrode must be aware of which neuron model the neurons adhere to
        # so that it can communicate to the right set of functions, like the step() function,
        # which differs between different neuron models.
        
        e = Electrode(self.get_neuron_instance(location))
        self.Electrodes.append((location,e))

    def step(self) -> None:
                                        
        """
        Perform one step of the brian simulation.
        """
        for _, neuron in self.view_neurons().items():
            neuron.step()
            
    # END of brain class

'========================================================================================================================'

class Experiment:
    """
    All experiments will be ran here.
    Here is where the ablations of neurons and edges will occur.
    The visuals will be executed here.
    """
    def __init__(self,brain): 
        '''
        We are granting the experiment class access to the brain class.
        
        For the upcoming experiments, we are going to duplicate the control brain to create the perturbed brain. From there, the
        
        any perturbations can be carried out in the perturbed brain.
        
        Parameters
        -----------
        brain: the instance of a Brain() class that contains the connectome 
        '''
        
        # controlled connectome 
        self.control_brain=brain
        
        
        # instantiate a new Brain that will function as the perturbed brain.
        
        self.perturbed_brain = Brain(**self.control_brain.kwargs,_print_message=False)
        
        # Create a perturbed brain that is identical to the control brain.
        
        # On the other hand, when a brain is created from scratch, all of the initial synapses and connections 
        # must be transferred over to the perturbed brain.
        
        # The following code will reinstate the neurons and synapses of a control brain that was prepared from 
        # scratch (i.e it does not have any initial kwargs instantiated) into the perturbed brain.
        
        if list(self.perturbed_brain._graph.nodes) == [] and list(self.perturbed_brain._graph.edges) == []:
            for neuron_name, neuron_instance in self.control_brain.neurons.items():
                model = self.control_brain.neuron_models[neuron_name]
                if model == 'Hodgkin Huxley':
                    self.perturbed_brain.add_neuron(name = neuron_name, neuron = 'HHNeuron', initialv_offset = neuron_instance.offset)
                elif model == 'Leaky Integrate and Fire':
                    self.perturbed_brain.add_neuron(name = neuron_name, neuron = 'LIAFNeuron', initialv_offset = neuron_instance.offset)
                    
            for synapse, synapse_instance in self.control_brain.synapses.items():
                self.perturbed_brain.add_synapse(synapse_instance.u, synapse_instance.v, synapse_instance._current, synapse_instance.incorporate_weights, synapse_instance._edge_weight)
                
        # Now, we also need to make provisions for brains that started from a ready-made connectome, but were still modified
        # using the add/remove synapse and neuron functions. We want to take care not to re-add the same neurons and synapses
        # all over again.
        
        elif list(self.perturbed_brain._graph.nodes) != [] and list(self.perturbed_brain._graph.edges) !=[] and self.control_brain.modified==True:
            
            # Create a new empty Brain.
            self.perturbed_brain = Brain(_print_message=False)
            # Add all neurons in the self.control_brain.instances() dictionary
            for neuron, neuron_instance in self.control_brain.neurons.items():
                self.perturbed_brain.add_neuron(name = neuron, neuron = neuron_instance.model_name, initialv_offset= neuron_instance.offset)
                
            # Add all of the synapses that comprise the self.control_brain.synapses() dictionary
            for synapse, synapse_instance in self.control_brain.synapses.items():
                self.perturbed_brain.add_synapse(synapse_instance.u, synapse_instance.v, synapse_instance._current, synapse_instance.incorporate_weights, synapse_instance._edge_weight)
            
        else:  
        # The perturbed brain that we first instantiated is sufficient since the user uploads a brain and 
        # does not alter it (i.e add/remove synapses or neurons).
            pass
 

    def place_electrodes(self, perturbed_neurons=None):
        '''
        Place electrodes on neurons. That way, you can measure their voltages.
        
        Parameters
        ----------
        perturbed_neurons : list of neurons to receive an electrode in the Perturbed Brain. Default: All neurons
        '''
        if perturbed_neurons==None:
            perturbed_neurons=list(self.perturbed_brain.view_nodes(all_data=False))
        
        for perturbed_neuron in perturbed_neurons:
            self.perturbed_brain.add_electrode(perturbed_neuron)
            
        # All neurons in the control brain will receive an electrode.
        control_neurons=list(self.control_brain.view_nodes(all_data=False))
        
        for control_neuron in control_neurons:
            self.control_brain.add_electrode(control_neuron)
   

    def run_simulator(self, stim_list: list, stim_amp: float, duration: float, recording_list: list=None, connectome='Control'):
        '''
        Complete a synchronous simulation of selected neurons by defining the following parameters.
        
        Parameters
        ----------
        stim_list: List of neurons that will be stimulated 
        
        stim_amp: the amount of current we will stimulate the stimulated neurons with (in microAmps)
        
        duration: duration of simulation (in ms)
        
        recording_list: list of neurons to be recorded. Default: all neurons
        
        connectome: Choose to simulate Control or Perturbed Connectome. Default Control.
        '''
        
        self.duration = duration
        # just defining main_brain. 'Brain' is just a placeholder.
        self.main_brain='Brain'
        
        if connectome=='Control':
            print('running control brain')
            self.main_brain = self.control_brain
        elif connectome=='Perturbed':
            print('running perturbed brain')
            self.main_brain=self.control_brain
            
            
        for entry in self.main_brain.Electrodes:
            neuron_name, electrode = entry
            
            # re-initialize the empty spike count lists
            electrode._target.spike_count=[]
            # we need to instantiate the V, m, n, h matrices for the HH model.
            if electrode._target.model_name == 'HH':
                # at the end of the simulation, reset the counter value to 1 in the HH model.
                electrode._target.i = 1   
        
            
        # read in the time step from an electrode. HH and LIAF should have identical dt in heterogeneous brain.
        n, e= self.main_brain.Electrodes[0]
        self.dt = e._target.dt
        
        for entry in self.main_brain.Electrodes:
            neuron_name, electrode = entry
            # we need to instantiate the V, m, n, h matrices for the HH model.
            if electrode._target.model_name == 'HH':
                electrode._target.V=np.zeros(int(self.duration/self.dt)+1) # voltage
                electrode._target.m=np.zeros(int(self.duration/self.dt)+1) # gating variable, inactivation
                electrode._target.n=np.zeros(int(self.duration/self.dt)+1) # gating variable, activation
                electrode._target.h=np.zeros(int(self.duration/self.dt)+1) # gating variable, inactivation 
                electrode._target.I_Leak=np.zeros(int(self.duration/self.dt)+1) # leak channel current
                electrode._target.I_Na=np.zeros(int(self.duration/self.dt)+1) # sodium channel current
                electrode._target.I_K=np.zeros(int(self.duration/self.dt)+1) # potassium channel current 
                electrode._target.conductance_K=np.zeros(int(self.duration/self.dt)+1) # conductance of potassium
                electrode._target.conductance_Na=np.zeros(int(self.duration/self.dt)+1) # conductance of sodium
                
                # initialize the values of these matrices
                electrode._target.V[0]= electrode._target.initial_membrane_potential
                electrode._target.m[0]= electrode._target.alpha_m(electrode._target.V[0])/(electrode._target.beta_m(electrode._target.V[0])+electrode._target.alpha_m(electrode._target.V[0]))
                electrode._target.n[0]= electrode._target.alpha_n(electrode._target.V[0])/(electrode._target.beta_n(electrode._target.V[0])+electrode._target.alpha_n(electrode._target.V[0]))
                electrode._target.h[0]= electrode._target.alpha_h(electrode._target.V[0])/(electrode._target.beta_h(electrode._target.V[0])+electrode._target.alpha_h(electrode._target.V[0]))
            else:
                None
            
        # If user gives a specific recording_list
        if recording_list!=None:  
            # record is a list of tuples containing the (neuron_name, electrode) for each neuron in recording_list
            self.record=[]
            for neuron in recording_list:
                # Make sure that the neuron exists.
                if neuron not in self.main_brain.view_nodes():
                    print('{} is not in the connectome. Please add it before a recording can be taken'.format(neuron))
                else:
                    # Add the neuron's electrode to the record list.
                    for entry in self.main_brain.Electrodes:
                        neuron_name, electrode = entry
                        if neuron==neuron_name:
                            self.record.append(entry) 
        
        # Default recording_list contains all neurons               
        elif recording_list==None:
            self.record=self.main_brain.Electrodes
                    
        # Stimulated_neurons is a list of tuples containing the (neuron_name, electrode) for each neuron in stim_list
        self.stimulated_neurons=[]
        for neuron in stim_list:
            # Make sure the neuron exists.
            if neuron not in self.main_brain.view_nodes():
                print('{} is not in the connectome. Please add {} to the connectome so that the stimulation can occur'.format(neuron)) 
            else:
                # Add the neuron's electrode to the stimulation list.
                for entry in self.main_brain.Electrodes:
                    neuron_name, electrode = entry
                    if neuron_name==neuron:
                        self.stimulated_neurons.append(entry)
        
        # Will hold all of the voltage recordings of the neurons.
        self.recordings={}
        
        # Join the lists record and stimulated_neurons. Call this list, complete_neurons
        self.complete_neurons=[]
        
        for entry in self.stimulated_neurons:
            self.complete_neurons.append(entry)
        for entry in self.record:
            # Make sure neurons are not double-counted. If a neuron is stimulated and recorded, we will treat it
            # as a stimulated neuron and record those values.
            if entry not in self.complete_neurons:
                self.complete_neurons.append(entry)
    
            # Create an empty list for each of the neurons in complete_neurons. The voltages of each neuron over 
            # time will be recorded.
            # Store them in the dictionary recordings.
        
        for entry in self.complete_neurons:
            neuron, electrode = entry
            self.recordings[neuron]=[]
        
        # First, let's obtain their initial membrane potentials.
        for entry in self.complete_neurons:
            neuron, electrode = entry
            self.recordings[neuron].append(electrode.measure())
            
        # Next, we proceed in the simulation and make steps.
        for step in tqdm(range(int(self.duration/self.dt)),ascii=True, desc='Simulation in Progress'):
            self.main_brain.step()
            for entry in self.complete_neurons:
                # If the neuron was marked to be stimulated, use the inject() function on it.
                if entry in self.stimulated_neurons:
                    neuron, electrode = entry
                    electrode.inject(stim_amp)
                    self.recordings[neuron].append(electrode.measure())
                # If the neuron is not marked for stimulation, just measure its membrane potential.
                elif entry not in self.stimulated_neurons:
                    neuron, electrode = entry
                    self.recordings[neuron].append(electrode.measure())
                
            # self.main_brain.step()
  
    def visualize_connectome(self, connectome, circular=False):
        '''
        Parameters
        ----------
        circular: True or False. Select whether the connectome's neurons will be placed along a circular border or not.
        
        connectome = Perturbed or Control
        '''
        graph='graph'
        
        if connectome=='Perturbed':
            graph=self.perturbed_brain._graph
            
        elif connectome=='Control':
            graph=self.control_brain._graph
            
        pos = nx.spring_layout(graph,k=1.5)        
        
        # k controls the distance between the nodes and varies between 0 and 1
        # iterations is the number of times simulated annealing is run
        # default k =0.1 and iterations=50
        # bipartite, circular,planar,rescale,spring,spiral,spectrum
        
        plt.figure(figsize=(12,12))
        
        if circular==True:
            nx.draw_circular(graph, font_size=12, width = 2.0)
            
        elif circular==False: 
            nx.draw(graph, pos,font_size=12, width = 2.0)
        
    def visualize_raster_plot(self,neuron_list: list = [], connectome: str = 'Control', title: str = 'Raster Plot for Selected Neurons'):
        '''
        Visualize the spike frequency of neurons.
        
        Parameters
        ----------
        neuron_list: A list of the neurons whose Raster Plot will be featured.
        connectome: State which connectome the raster plot is coming from. 'Perturbed' or 'Control'. Default: Control.
        '''
        brain = 'brain'
        if connectome == 'Control':
            brain = self.control_brain
        elif connectome == 'Perturbed':
            brain = self.perturbed_brain
            
        if neuron_list==[]:
            neuron_list = self.complete_neurons
        else:
            neuron_list = neuron_list
        
        total_spikes=0
        raster_plot, ax = plt.subplots(figsize=(7,7))
        num_neurons = len(neuron_list)
        spacing = float(1/num_neurons)
        intervals = []
        i=0
        intervals.append(i)
        for neuron in neuron_list:
            for entry in brain.Electrodes:
                n, e = entry
                if neuron == n:
                    ax.plot(e._target.spike_count,[float(i+(spacing/2))]*len(e._target.spike_count),'o', markersize=10, label=neuron)
                    total_spikes+=int(len(e._target.spike_count))
                else:
                    None
            i+=spacing
            intervals.append(i)
        ax.set_xlim([0,self.duration])
        ax.set_ylim([0,1.0])
        ax.set_xlabel('Time (ms)',fontsize=18)
        ax.set_ylabel('Spikes',fontsize=18)
        ax.set_title(title,fontsize=20)
        ax.set_yticks(intervals)
        ax.axes.yaxis.set_ticklabels([])
        ax.legend()
        ax.grid(axis='y')
        print("Total Spike Count of the selected neurons is: ", total_spikes) 
                    
        return 
    
    def interspike_intervals(self):
        '''
        Code will be developed in the following versions.
        '''
        return None
    
    def visualize_spike_trains(self, neuron_list: list=None):
        '''
        Present the spike trains from the simulation.
        
        Parameters
        ----------
        neuron_list: list of neurons to plot. Default = all neurons
        '''
        spike_trains = plt.figure(figsize=(7,7))
        self.times=np.arange(0,self.duration+self.dt,self.dt, dtype=float)
        

        if neuron_list == None:
            for neuron in self.complete_neurons:
                plt.plot(self.times,self.recordings[neuron])
            plt.legend(list(self.complete_neurons()))
            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage Potential (V)')
            ##plt.show()
            
        else:
            for neuron in neuron_list:
                if neuron not in self.recordings.keys():
                    print('Please place an electrode on {} then run the simulator on {}'.format(*neuron))
                else:
                    plt.plot(self.times,self.recordings[neuron])
                    
            plt.legend(neuron_list)
            plt.xlabel('Time (ms)', fontsize=18)
            plt.ylabel('Voltage Potential (mV)', fontsize=18)
            #plt.yticks(np.arange(0, 0.15, 0.01))
            plt.show()
    
        return 
    
    def visualize_connectivity_matrix(self):
        '''
        Code will be developed in the following versions.
        '''
        pass
    
    def structural_metrics(self, EVC: bool=True, BTC: bool=True, clustering: bool=True, graph_density: bool=True, shortest_path_length: bool=True):
        '''
        Parameters
        ----------
        EVC, BTC, clustering, graph_density and shortest_path_length are all graph metrics from networkx.
        If True, they will be computed.
        '''
        # rows are neuron names, columns are graph metrics
        
        self.connectome_metrics=pd.DataFrame() # compile all of the perturbed and control connectome data.
        
        # ----------------------------------------------------------------------------------------------------
        '''
        Eigenvector Centrality has 9 possible returns. 
            - [0] org_EVC is a dictionary of original EVC values for each node.
            - [1] per_EVC is a dictionary of perturbed EVC values for each node.
            - [2] org_avg_EVC is the average EVC value for the original graph.
            - [3] per_avg_EVC is the average EVC value for the perturbed graph.
            - [4] avg_EVC_difference is the absolute difference between org_avg_EVC and per_avg_EVC.
            - [5] org_EVC_df is a Pandas dataframe of original EVC values for each node in descending order.
            - [6] per_EVC_df is a Pandas dataframe of perturbed EVC values for each node in descending order.
            - [7] EVC_df_org_per is a Pandas dataframe combining org_EVC_df and per_EVC_df.
            - [8] summary_EVC_stats returns basic statistics on EVC_df_org_per
        '''
        
        if EVC==True:
            
            self.perturbed_connectome_EVC=pd.DataFrame()
        
            self.control_connectome_EVC=pd.DataFrame()
            
            self.org_EVC = nx.eigenvector_centrality(self.control_brain._graph)
            self.per_EVC = nx.eigenvector_centrality(self.perturbed_brain._graph)
    
            self.org_EVC_lst = [v for k,v in self.org_EVC.items()]
            self.org_avg_EVC = statistics.mean(self.org_EVC_lst) 
            
            self.per_EVC_lst = [v for k,v in self.per_EVC.items()]
            self.per_avg_EVC = statistics.mean(self.per_EVC_lst)
            
            self.avg_EVC_difference = abs(self.org_avg_EVC - self.per_avg_EVC)
    
            self.control_connectome_EVC=self.control_connectome_EVC.from_dict(self.org_EVC, orient='index',
                                    columns=['Control Eigenvector Centrality']).sort_values(by='Control Eigenvector Centrality',ascending=False)
        
            self.perturbed_connectome_EVC=self.perturbed_connectome_EVC.from_dict(self.per_EVC, orient='index',
                                    columns=['Perturbed Eigenvector Centrality']).sort_values(by='Perturbed Eigenvector Centrality',ascending=False)

            self.joint_EVC_df = self.control_connectome_EVC.join(self.perturbed_connectome_EVC)
            self.joint_EVC_df['Absolute Difference in EVC'] = abs(self.control_connectome_EVC['Control Eigenvector Centrality'] - self.perturbed_connectome_EVC['Perturbed Eigenvector Centrality'])
    
            self.spectral_gap = self.joint_EVC_df
            self.spectral_gap.drop(['Absolute Difference in EVC'], axis=1, inplace=True)
            self.spectral_gap = self.spectral_gap.diff()
    
    
            self.summary_EVC_stats = self.joint_EVC_df.describe()
        
            # compile the EVC data into a common dataframe with the other graph metrics.
        
            self.connectome_metrics['Control Eigenvector Centrality (EVC)']=self.control_connectome_EVC['Control Eigenvector Centrality']
            self.connectome_metrics['Perturbed Eigenvector Centrality (EVC)']=self.perturbed_connectome_EVC['Perturbed Eigenvector Centrality']
            
            self.connectome_metrics['Absolute Difference in EVC']=abs(self.control_connectome_EVC['Control Eigenvector Centrality'] - self.perturbed_connectome_EVC['Perturbed Eigenvector Centrality'])

        else:
            None

        # -----------------------------------------------------------------------------------------------------------
        
        '''
        BTC(original, perturbed) has 10 possible returns. 
            - [0] org_BTC is a dictionary of original BTC values for each node.
            - [1] per_BTC is a dictionary of perturbed BTC values for each node.
            - [2] org_avg_BTC is the average BTC value for the original graph.
            - [3] per_avg_BTC is the average BTC value for the perturbed graph.
            - [4] avg_BTC_difference is the absolute difference between org_avg_BTC and per_avg_BTC.
            - [5] org_BTC_df is a Pandas dataframe of original BTC values for each node in descending order.
            - [6] per_BTC_df is a Pandas dataframe of perturbed BTC values for each node in descending order.
            - [7] BTC_df_org_per is a Pandas dataframe combining org_BTC_df and per_BTC_df.
            - [8] network_hubs returns the network hubs of the given connectome.
            - [9] summary_BTC_stats returns basic statistics on BTC_df_org_per.
        '''
        if BTC==True:
            
            self.perturbed_connectome_BTC=pd.DataFrame()
        
            self.control_connectome_BTC=pd.DataFrame()
            
            self.org_BTC = nx.betweenness_centrality(self.control_brain._graph)
            self.per_BTC = nx.betweenness_centrality(self.perturbed_brain._graph)
    
            self.org_BTC_lst = [v for k,v in self.org_BTC.items()]
            self.org_avg_BTC = statistics.mean(self.org_BTC_lst)
            
            self.per_BTC_lst = [v for k,v in self.per_BTC.items()]
            self.per_avg_BTC = statistics.mean(self.per_BTC_lst)
            self.avg_BTC_difference = abs(self.org_avg_BTC - self.per_avg_BTC)
    
            self.control_connectome_BTC=self.control_connectome_BTC.from_dict(self.org_BTC, orient='index',
                                    columns=['Control Betweenness Centrality']).sort_values(by='Control Betweenness Centrality',ascending=False)
            self.perturbed_connectome_BTC=self.perturbed_connectome_BTC.from_dict(self.per_BTC, orient='index',
                                    columns=['Perturbed Betweenness Centrality']).sort_values(by='Perturbed Betweenness Centrality',ascending=False)
            
            self.joint_BTC_df = self.control_connectome_BTC.join(self.perturbed_connectome_BTC)
            self.joint_BTC_df['Absolute Difference in BTC'] = abs(self.control_connectome_BTC['Control Betweenness Centrality'] - self.perturbed_connectome_BTC['Perturbed Betweenness Centrality'])
    
            self.network_hubs = self.control_connectome_BTC.head(int(len(self.control_brain._graph.nodes)*.1))
    
            self.summary_BTC_stats = self.joint_BTC_df.describe()
        
            # compile the EVC data into a common dataframe with the other graph metrics. 
            
            self.connectome_metrics['Control Betweenness Centrality (BTC)']= self.control_connectome_BTC['Control Betweenness Centrality']
            self.connectome_metrics['Perturbed Betweenness Centrality (BTC)']=self.perturbed_connectome_BTC['Perturbed Betweenness Centrality']
            
            self.connectome_metrics['Absolute Difference in BTC']=self.joint_BTC_df['Absolute Difference in BTC']
    
        else:
            None

        # -----------------------------------------------------------------------------------------------------------
                    
        '''
        clustering(original, perturbed) has 8 possible returns. 
            - [0] org_clustering is a dictionary of original clustering coefficient values for each node.
            - [1] per_clustering is a dictionary of perturbed clustering coefficient values for each node.
            - [2] org_avg_clustering returns the average clustering coefficient for the original graph.
            - [3] per_avg_clustering returns the average clustering coefficient for the perturbed graph.
            - [4] avg_clustering_difference is the absolute difference between org_avg_clustering and per_avg_clustering.
            - [5] org_clustering_df is a Pandas dataframe of original clustering coefficient values for each node 
                    in descending order.
            - [6] per_clustering_df is a Pandas dataframe of perturbed clustering coefficient values for each node 
                in descending order.
            - [7] clustering_df_org_per is a Pandas dataframe combining org_clustering_df and per_clustering_df.
        '''
        if clustering==True:
            
            self.perturbed_connectome_CLUSTERING=pd.DataFrame()
        
            self.control_connectome_CLUSTERING=pd.DataFrame()
            
            self.org_clustering = nx.algorithms.cluster.clustering(self.control_brain._graph)
            self.per_clustering = nx.algorithms.cluster.clustering(self.perturbed_brain._graph)
    
            self.org_avg_clustering = nx.algorithms.cluster.average_clustering(self.control_brain._graph)
            self.per_avg_clustering = nx.algorithms.cluster.average_clustering(self.perturbed_brain._graph)
            self.avg_clustering_difference = abs(self.org_avg_clustering - self.per_avg_clustering)
    
            self.control_connectome_CLUSTERING=self.control_connectome_CLUSTERING.from_dict(self.org_clustering, orient='index', columns=['Control Clustering Coefficients']).sort_values(by='Control Clustering Coefficients',ascending=False)
        
            self.perturbed_connectome_CLUSTERING=self.perturbed_connectome_CLUSTERING.from_dict(self.per_clustering, orient='index', columns=['Perturbed Clustering Coefficients']).sort_values(by='Perturbed Clustering Coefficients',ascending=False)
            
            self.joint_clustering_df = self.control_connectome_CLUSTERING.join(self.perturbed_connectome_CLUSTERING)
            
            # compile the EVC data into a common dataframe with the other graph metrics.
            
            self.connectome_metrics['Control Clustering Coefficients']=self.control_connectome_CLUSTERING['Control Clustering Coefficients']
            self.connectome_metrics['Perturbed Clustering Coefficients']=self.perturbed_connectome_CLUSTERING['Perturbed Clustering Coefficients']
    
        else:
            None
        
        #------------------------------------------------------------------------------------------------------------
        '''
        graph_density(original, perturbed) has 3 possible returns. 
            - [0] org_density is the original graph density.
            - [1] per_density is the perturbed graph density.
            - [2] density_difference is the absolute difference between org_density and per_density. 
        '''
        if graph_density==True:
            
            self.org_density = .5 * (len(self.control_brain._graph.edges))/((len(self.control_brain._graph.nodes)) * (len(self.control_brain._graph.nodes)-1))
            
            self.per_density = .5 * (len(self.perturbed_brain._graph.edges))/((len(self.perturbed_brain._graph.nodes)) * (len(self.perturbed_brain._graph.nodes)-1))
    
            self.density_difference = abs(self.org_density - self.per_density)

        else:
            None
            
        #--------------------------------------------------------------------------------------------------------------
                    
        '''
        shortest_path_length(original, perturbed) has 3 possible returns. 
            - [0] org_shortest_path_length is the original shortest path length.
            - [1] per_shortest_path_length is the perturbed shortest path length.
            - [2] shortest_path_length_difference is the absolute difference between org_shortest_path_length 
            and per_shortest_path_length. 
        '''  
        if shortest_path_length==True:
            
            self.org_shortest_path_length = nx.algorithms.shortest_paths.generic.average_shortest_path_length(self.control_brain._graph)
            
            self.per_shortest_path_length = nx.algorithms.shortest_paths.generic.average_shortest_path_length(self.perturbed_brain._graph)
            
            self.shortest_path_length_difference = abs(self.org_shortest_path_length - self.per_shortest_path_length)
    
        else:
            None
            
        return self.connectome_metrics
    
    
    def retrieve_experimental_data(self, title:str):
        '''
        Run diagnostics on the experiment by calling the functional and graph metrics.
        Convert voltages/current from record list to a pandas data frame.
        Store as a csv or xsl file. Name the data with a descriptor e.g., Experiment 1: Ablation of ADAL
        User can provide this descriptor.
        
        Parameters
        ----------
        title: Name the experiment
        '''
        data=pd.DataFrame.from_dict(self.recordings)
        
        return data
    
        
    def display_brain_info(self, connectome: str = 'Control'):
        '''
        Make sure that the brain you have uploaded is what you anticipated
        
        Parameters
        ----------
        connectome: Control or Perturbed connectome.
        '''
        if connectome=='Control':
            print('This is your present brain:\n', nx.info(self.control_brain._graph))
            
            print("This is the brain's record of all the neurons:\n",self.control_brain.mapping)
            
            print('These are the neuron models associated with each neuron:\n',self.control_brain.neuron_models_dict())
            
        elif connectome=='Perturbed':
            print('This is your present brain:\n', nx.info(self.perturbed_brain._graph))
            
            print("This is the brain's record of all the neurons:\n",self.perturbed_brain.mapping)
            
            print('These are the neuron models associated with each neuron:\n',self.perturbed_brain.neuron_models_dict())          
              
          
    def adjust_connectivity_strength(self, synapse: tuple, new_weight: float):
        '''
        Rather than perform a complete ablation of an edge or node,
        reduce or increase the strength of the weights (strength of connectivity)
    
        Parameters
        ---------
        synapse: synapse (pre,post) whose weight will be adjusted
        
        new_weight: numerical value of the new weight
        '''
        self.perturbed_brain.update_edge_weight(self, synapse, new_weight)
            
            
    def view_ablated_connectome(self,circular=False):
        '''
        Visualize the perturbed connectome.
        Parameters
        ----------
        circular: True or False. Select whether connectome will feature neurons along a circular border or not.
        '''
        
        #VISUALIZING NODE ABLATION
        #Node color map that adds blue if node exists in control 
        #and perturbed connectome, and red if node exists 
        #in control but NOT perturbed
        
        node_color_map = []
        for nodes in list(self.control_brain.view_nodes(all_data=False)):
            if nodes in self.perturbed_brain.view_nodes(all_data=False):
                node_color_map.append('blue')
            else:
                node_color_map.append('red')

        # Edge color map that adds gray if edge exists in control/perturbed connectome
        # and red if edge exists in control but NOT perturbed
        
        edge_color_map = []
        for edges in list(self.control_brain.view_edges(all_data=False)):
            if edges in self.perturbed_brain.view_edges(all_data=False):
                edge_color_map.append('gray')
            else:
                edge_color_map.append('red')

        pos = nx.spring_layout(self.control_brain._graph,k=1.5)        
        
        # k controls the distance between the nodes and varies between 0 and 1
        # iterations is the number of times simulated annealing is run
        # default k =0.1 and iterations=50
        # bipartite, circular,planar,rescale,spring,spiral,spectrum
        
        plt.figure(figsize=(50,50))
        
        if circular==False: 
            nx.draw(self.control_brain._graph, pos,font_size=12, edge_color = edge_color_map,with_labels=True, node_color = node_color_map, width = 2.0)
            
        elif circular==True: 
            nx.draw_circular(self.control_brain._graph, font_size=12, edge_color = edge_color_map,with_labels=True, node_color = node_color_map, width = 2.0)
        








