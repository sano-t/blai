import sys

import yaml
from animalai_train.trainers.ppo.policy import PPOPolicy
from animalai.envs.brain import BrainParameters

import numpy as np

from keras.models import load_model

class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
        """

        # Load the configuration and model using ABSOLUTE PATHS
        self.configuration_file = '/aaio/AnimalAI-Olympics/examples/configs/trainer_config.yaml'
        self.model_path = '/aaio/AnimalAI-Olympics/examples/models/train_example/Learner'

        self.brain = BrainParameters(brain_name='Learner',
                                     camera_resolutions=[{'height': 84, 'width': 84, 'blackAndWhite': False}],
                                     num_stacked_vector_observations=1,
                                     vector_action_descriptions=['', ''],
                                     vector_action_space_size=[3, 3],
                                     vector_action_space_type=0,  # corresponds to discrete
                                     vector_observation_space_size=3
                                     )
        self.trainer_params = yaml.load(open(self.configuration_file))['Learner']
        self.trainer_params['keep_checkpoints'] = 0
        self.trainer_params['model_path'] = self.model_path
        self.trainer_params['use_recurrent'] = False

        self.policy = PPOPolicy(brain=self.brain,
                                seed=0,
                                trainer_params=self.trainer_params,
                                is_training=False,
                                load=True)

        self.memory = np.zeros( (15, 84, 84, 3) ) # keep 15 frames for prediction
        self.internal_model = load_model("model.h5")

        
    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        
        self.memory = np.zeros( (15, 84, 84, 3) ) # erase memory


    def is_blackout(self, obs):
        if (np.abs( np.mean( obs[0] ) )<1.e-6 ):
            print('blackout detected', file=sys.stderr)
            return True
        return False

    
    def prediction(self):
        """
        predict current obsavation by using memory
        """
        print('prediction', file=sys.stderr)
        pred = self.internal_model.predict(self.memory[np.newaxis,:,:,:,:])
        return pred[:,-1,:,:,:].reshape(1,84,84,3)
        
    
    def step(self, obs, reward, done, info):
        """
        :param obs: agent's observation of the current environment
        :param reward: amount of reward returned after previous action
        :param done: whether the episode has ended.
        :param info: contains auxiliary diagnostic information, including BrainInfo.
        :return: the action to take, a list or size 2
        """

        brain_info = info['brain_info']
        
        if(self.is_blackout(obs)):
            pred = self.prediction()
            brain_info.visual_observations[0] = pred


        # update memory
        #print(self.memory.shape)
        #print(brain_info.visual_observations[0].shape)

        self.memory = np.append(self.memory, brain_info.visual_observations[0], axis=0)
        self.memory = self.memory[1:16,:,:,:]

        
        action = self.policy.evaluate(brain_info=brain_info)['action']

        return action
