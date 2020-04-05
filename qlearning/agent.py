import numpy as np 

class Agent:
    
    '''
    * This is the class for our reinforcement learning agent.

    * It will be containing the policy with wich it will interact with the environment. 
      It will be stored as a dictionary.

    * It requires following parameters:
        1. Learning Rate --> lr
        2. Discount Factor --> gamma
        3. no of actions --> n_actions
        4. no of states --> n_states
        5. eps_start --> initial value of epsilon for epsilon greedy process
        6. eps_end --> least value of epsilon for epsilon greedy process
        7. eps_decay --> decay rate value of epsilon for epsilon greedy process
    '''
    
    def __init__(self, lr,gamma,n_actions,n_states,eps_start,eps_end,eps_decay):
       
       '''
       we will store all the parameters as class members for easy acces.
       '''
       self.lr = lr
       self.n_actions = n_actions
       self.n_states = n_states
       self.gamma = gamma
       self.eps = eps_start
       self.eps_end = eps_end
       self.eps_decay = eps_decay

       self.Q = {} #* This is the policy which will be used by the agent to interact with the environment.

       self.init_Q() #* A function for initializing the policy.
    
    def init_Q(self):

        '''
        This function populates our state-action pairs with 0s.
        '''
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state,action)] = 0
    
    def decrement_epsilon(self):

        '''
        This function is used to decrease the epsilon value over time so that our agent first explores
        the environment and based on the experience exploits it. Simple epsilon-greedy approach.
        '''
        self.eps = self.eps*self.eps_decay if self.eps>self.eps_end else self.eps_end

    def choose_action(self,state):

        '''
        This function is for choosing the action according to the epsilon greedy approach
        We generate a random number and if it > epsilon then we choose action from the policy 
        otherwise take a random action
        '''
        
        if np.random.random()>self.eps:
            actions = np.array([self.Q[(state,a)] for a in range(self.n_actions)])
            action = np.argmax(actions)
        else:
            action = np.random.choice([a for a in range(self.n_actions)])
        return action

    def learn(self,state,action,reward,state_):
        
        '''
        This functions is for the learning step of the agent.
        We update the Q/policy table via the equation

         Q(s,a) = Q(s,a) + learning_rate*(reward + gamma*max(Q(s',a')) - Q(s,a))

         where 
            1. s --> current state
            2. a --> action
            3  s' --> next state
            
         After this we decrease or epsilon      
        '''
        
        actions = np.array([self.Q[(state_,a)] for a in range(self.n_actions)])
        action_max = np.argmax(actions)

        self.Q[(state,action)] = self.Q[(state,action)] + \
                                 self.lr*(
                                     reward+
                                      self.gamma*self.Q[(state_,action_max)] -
                                     self.Q[(state,action)]
                                 )
        self.decrement_epsilon()




    