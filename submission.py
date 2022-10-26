import math
import pickle
import numpy as np
import copy
import os
import cv2

from abc import abstractmethod, ABC
from typing import Any
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.layers as layers


class CategoricalPd:
    def __init__(self, logits):
        self.logits = logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def logp(self, x):
        return -self.neglogp(x)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def sample(self, mask=1):
        u = tf.random.uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        self.logits = self.logits - (1-mask)*1e10
        return tf.argmax(self.logits - tf.math.log(-tf.math.log(u)), axis=-1)

class Model(tf.keras.Model, ABC):
    def __init__(self, observation_space: Any, action_space: Any, config: dict = None, model_id: str = '0',
                 *args, **kwargs) -> None:
        """
        This method MUST be called after (0.) in subclasses

        0. [IN '__init__' of SUBCLASSES] Define parameters, layers, tensors and other related variables
        1. If 'config' is not 'None', set specified configuration parameters (which appear after 'config')
        2. Build model

        :param model_id: The identifier of the model
        :param config: Configurations of hyper-parameters
        :param args: Positional configurations (ignored if specified in 'config')
        :param kwargs: Keyword configurations (ignored if specified in 'config')
        """
        super(Model, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_id = model_id
        self.config = config

        # 1. Set configurations
        if config is not None:
            self.load_config(config)

        # 2. Build up model
        self.build()

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """Build the computational graph"""
        pass

    @abstractmethod
    def set_weights(self, weights: Any, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_weights(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def call(self, states: Any, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def save(self, path: Path, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def load(self, path: Path, *args, **kwargs) -> None:
        pass

class TFModel(Model, ABC):
    def __init__(self, observation_space: Any, action_space: Any, config=None, model_id='0', session=None, scope=None,
                 *args, **kwargs):
        self.scope = scope

        # Initialize Tensorflow session

        super(TFModel, self).__init__(observation_space, action_space, config, model_id, *args, **kwargs)

        # Build assignment ops
        self._weight_ph = None
        self._to_assign = None
        self._nodes = None

    def set_weights(self, weights, *args, **kwargs) -> None:
        for i in range(len(self.trainable_variables)):
            self.trainable_variables[i].assign(weights[i])


    def get_weights(self, *args, **kwargs) -> Any:
        return self.trainable_variables

    def save(self, path: Path, *args, **kwargs) -> None:
        self.save_weights(path)

    def load(self, path: Path, *args, **kwargs) -> None:
        self.load_weights(path)

    def _build_assign(self):
        self._weight_ph, self._to_assign = dict(), dict()
        variables = self.trainable_variables
        variables = tf.trainable_variables(self.scope)
        for var in variables:
            self._weight_ph[var.name] = tf.placeholder(var.value().dtype, var.get_shape().as_list())
            self._to_assign[var.name] = var.assign(self._weight_ph[var.name])
        self._nodes = list(self._to_assign.values())

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def call(self, states: Any, *args, **kwargs) -> Any:
        pass

class ACModel(TFModel ,ABC):
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        super(ACModel, self).__init__(observation_space, action_space, config, model_id, scope=model_id,
                                        *args, **kwargs)
            
        self.observation_space = observation_space
        self.action_space = action_space

        self.logits_force = None
        self.logits_angle = None
        self.vf = None

        self.learn_action = None

    def get_action(self):
        self.pd_force = CategoricalPd(self.logits_force)
        self.pd_angle = CategoricalPd(self.logits_angle)
        action_force = self.pd_force.sample()
        action_angle = self.pd_angle.sample()

        neglogp_force = self.pd_force.neglogp(action_force)
        neglogp_angle = self.pd_angle.neglogp(action_angle)
        self.neglogp = neglogp_force + neglogp_angle
        
        action_force = tf.reshape(action_force, [-1, 1])
        action_angle = tf.reshape(action_angle, [-1, 1])
        self.action = tf.concat([action_force, action_angle], axis=1)

    def get_neglogp_entropy(self):
        force_ph, angle_ph = tf.split(self.learn_action, [1,1], axis=1)
        force_ph = tf.reshape(force_ph, [-1])
        angle_ph = tf.reshape(angle_ph, [-1])
        neglogp_force_ph = self.pd_force.neglogp(force_ph)
        neglogp_angle_ph = self.pd_angle.neglogp(angle_ph)
        self.neglogp_a = neglogp_angle_ph + neglogp_force_ph

        entropy_force = self.pd_force.entropy()
        entropy_angle = self.pd_angle.entropy()
        self.entropy = entropy_angle + entropy_force


    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def call(self, states, *args, **kwargs) -> None:
        pass

class ACCNNModel(ACModel, ABC):

    def build(self, *args, **kwargs) -> None:
        self.c1 = layers.Conv2D(
            filters=25,
            kernel_size=12, 
            strides=4, 
            padding='SAME', 
            activation='relu',
            trainable=True
        )
        self.p1 = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="valid")
        self.c2 = layers.Conv2D(
            filters = 25,
            kernel_size=12,
            strides=4, 
            padding='SAME', 
            activation='relu', 
            trainable=True
        )

        self.fc1 = layers.Dense(units=128, activation='relu', trainable=True)
        self.fl = layers.Flatten()

        self.pif_fc1 = layers.Dense(units=64, activation='relu')
        self.pif_fc2 = layers.Dense(units=64, activation='relu')
        self.pif_fc3 = layers.Dense(units=self.action_space[0], activation='relu')

        self.pia_fc1 = layers.Dense(units=64, activation='relu')
        self.pia_fc2 = layers.Dense(units=64, activation='relu')
        self.pia_fc3 = layers.Dense(units=self.action_space[0], activation='relu')

        self.v_fc1 = layers.Dense(units=64, activation='relu')
        self.v_fc2 = layers.Dense(units=64, activation='relu')
        self.v_fc3 = layers.Dense(units=1, activation='relu')

    def state_process(self):
        
        input_images = tf.cast(self.states, tf.float32)
        outstem = self.c1(input_images)
        outstem = self.p1(outstem)
        outstem = self.c2(outstem)
        outstem = self.fl(outstem)

        latent = self.fc1(outstem)

        pih1f = self.pif_fc1(latent)
        pih2f = self.pif_fc2(pih1f)
        self.logits_force = self.pif_fc3(pih2f)

        pih1a = self.pia_fc1(latent)
        pih2a = self.pia_fc2(pih1a)
        self.logits_angle = self.pia_fc3(pih2a)

        vh1 = self.v_fc1(latent)
        vh2 = self.v_fc2(vh1)
        self.vf = tf.squeeze(self.v_fc3(vh2), axis=1)
    
    def init_model(self):
        self(np.zeros([1, 40, 40, 16]))

    
    def call(self, states: Any, training = False, *args, **kwargs) -> Any:
        self.states = states
        self.state_process()
        self.get_action()
        if training:
            self.get_neglogp_entropy()
        return self.action.numpy(), self.vf.numpy(), self.neglogp.numpy()



def obs_pre_process(obs, state4, controled_player_idx):
    '''
    输入一个智能体的observation（40*40），输出一个智能体的四个state（共16个channel）的叠帧
    '''

    obs_ = np.zeros((40, 40, 4), dtype='int32')
    
    idx_to_enemy_color = [8, 10]
    enemy_color = idx_to_enemy_color[controled_player_idx]
    my_color = idx_to_enemy_color[1-controled_player_idx]
    for i in range(40):
        for j in range(40):
            if obs[i][j] == enemy_color:
                obs_[i][j][0] = 1
            elif obs[i][j] == 4:
                obs_[i][j][1] = 1
            elif obs[i][j] == 1:
                obs_[i][j][2] = 1
            elif obs[i][j] == my_color: 
                obs_[i][j][3] = 1
    
    state_ = np.expand_dims(obs_, 0)
    
    if len(state4) == 0:
        for i in range(3):
            state4.append(state_)
    # 叠4帧
    state4.append(state_)
    state = state4.pop(0)
    for i in range(3):
        state = np.concatenate((state, state4[i]), axis = 3)

    return state4, state

class Agent():

    def __init__(self) -> None:
        self.step = 0
        self.ball_center = (19, 32)
        self.angle = 0
        self.opp_color = 8
        self.collision_threshold = 7.5

        self.backword_threshold = 30
        self.energy = 1000

        self.state4 = []

        self.opp_length_list = [60]
        self.opp_pos_list = []
        self.opp_v = [0,0]
        self.opp_v_list = []

        self.opp2center_list = []

        self.border_error = [[], []]
        self.cross_error = [[], []]

        self.pos_list = []
        self.energy_list = []
        self.force_list = [0]
        self.delta_list = [0]
        
        self.pos = []
        self.v = [0, 0]

        self.back = 0

        self.v_trust_list = [True]

        self.ball_collision = False
        self.trustable = True

        model_slow_id = os.path.dirname(os.path.abspath(__file__)).split('\\')[-1] + '_slow'
        self.model_slow = ACCNNModel(observation_space=(40, 40, 16), action_space=(31, 31), model_id=model_slow_id)

        slow_file_path = os.path.dirname(os.path.abspath(__file__)) + "/slow.pkl"
        slow_file_path.replace('\\', '/')
        with open(slow_file_path, 'rb') as f:
            slow_weights = pickle.load(f)
        self.model_slow.init_model()

        self.model_slow.set_weights(slow_weights)

        self.model_choose = 0

        self.to_center_threshold = 18

        self.trust_predict_threshold = 5
        
        self.init_mode = True

        self.opp_v_trust = False
        
    def find_border(self):
        border_list = []
        border_sum = [0, 0]
        for i in range(40):
            for j in range(40):
                if self.obs[j][i] == 1:
                    border_list.append((i,j))
                    border_sum[0] += i
                    border_sum[1] += j
        if len(border_list) < 5:
            return []
        m = 0
        max_pos = (0,0)

        for i in range(len(border_list)-1):
            b0 = border_list[i]
            for j in range(i+1, len(border_list)):
                b1 = border_list[j]
                t = math.sqrt((b0[0]-b1[0])**2+(b0[1]-b1[1])**2)
                if m < t:
                    m = t
                    max_pos = (i,j)
        b0 = border_list[max_pos[0]]
        b1 = border_list[max_pos[1]]
        md = ((b0[0]+b1[0])/2, (b0[1]+b1[1])/2)

        if m > 20:
            p0 = border_sum[0]/len(border_list)
            p1 = border_sum[1]/len(border_list)
        else:
            p0 = md[0]
            p1 = md[1]
        
        theta = math.atan2(abs(b0[1]-b1[1]), abs(b1[0]-b0[0]))
        l = math.sqrt(self.L**2-((b0[0]-b1[0])**2+(b0[1]-b1[1])**2)/4)

        k0 = 1 if p0 > md[0] else (-1 if p0 < md[0] or p0 < self.ball_center[0] else 1)
        k1 = 1 if p1 > md[1] else (-1 if p1 < md[1] or p1 < self.ball_center[1] else 1)



        if (b0[0]-b1[0])*(b0[1]-b1[1]) < 0:
            if abs((b0[1]-b1[1])/(b0[0]-b1[0])) < 1:
                x = l * math.sin(theta) * k1 * -1
                y = l * math.cos(theta) * k1 * -1
            else:
                x = l * math.sin(theta) * k0 * -1
                y = l * math.cos(theta) * k0 * -1
        elif (b0[0]-b1[0])*(b0[1]-b1[1]) > 0:
            if abs((b0[1]-b1[1])/(b0[0]-b1[0])) < 1:
                x = l * math.sin(theta) * k1
                y = l * math.cos(theta) * k1 * -1
            else:
                x = l * math.sin(theta) * k0 * -1
                y = l * math.cos(theta) * k0
        else:
            x = l * math.sin(theta) * k0 * -1
            y = l * math.cos(theta) * k1 * -1

        center = (md[0]+x, md[1]+y)
        return self.obs2pos(self.ball_center, center, (300, 350))

    def cross_point(self, line1, line2): 
        point_is_exist=False
        x=0
        y=0
        x1 = line1[0]  
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]

        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]

        if (x2 - x1) == 0:
            k1 = None
        else:
            k1 = (y2 - y1) * 1.0 / (x2 - x1)  
            b1 = y1 * 1.0 - x1 * k1 * 1.0  

        if (x4 - x3) == 0:  
            k2 = None
            b2 = 0
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)  
            b2 = y3 * 1.0 - x3 * k2 * 1.0

        if k1 is None:
            if not k2 is None and abs(k2) < 0.2:
                x = x1
                y = k2 * x1 + b2
                point_is_exist=True
        elif k2 is None:
            if abs(k1) < 0.2:
                x=x3
                y=k1*x3+b1
                point_is_exist=True
        elif (k1*k2 > -2 and k1*k2 < -0.5) or (k1 * k2 == 0 and max(abs(k1), abs(k2)) > 12):
            x = (b2 - b1) * 1.0 / (k1 - k2)
            y = k1 * x * 1.0 + b1 * 1.0
            point_is_exist=True
        return point_is_exist,[x, y]

    def find_center(self):
        obs = np.zeros((40, 40), 'uint8')
        for i in range(40):
            for j in range(40):
                if self.obs[i][j] == 4:
                    obs[i,j] = 255

        lines_long = cv2.HoughLinesP(image=obs, rho=1, theta=np.pi/180, threshold=10, minLineLength=10, maxLineGap=8)
        circle = cv2.HoughCircles(image=obs, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, circles=None, param1=100, param2=12, minRadius=7, maxRadius=20)

        center_lines = self.lines2points(lines_long)

        if type(circle) == np.ndarray:
            center_circle = np.sum(circle[0, :,:2], axis=0)/len(circle[0])
        pos_c = self.obs2pos(self.ball_center, center_circle, (300, 350)) if type(circle) == np.ndarray else []
        pos_l = self.obs2pos(self.ball_center, center_lines, (300, 350)) if len(center_lines) else []
        return pos_c, pos_l

    def lines2points(self, lines):
        if type(lines) != np.ndarray:
            return []
        lines =lines[:, 0, :]
        pos_list = []
        length_list = []
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i]
            Li = (x1-x2)**2 + (y1-y2)**2
            for x3, y3, x4, y4 in lines[i:]:
                point_is_exist, [x, y]=self.cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
                if point_is_exist:
                    Lj = (x1-x2)**2 + (y1-y2)**2
                    pos_list.append([x, y])
                    length_list.append((Li+Lj)/2)
        if pos_list:
            combined_length_list = [length_list[0]]
            combined_pos_list = [[pos_list[0]]]
            for p, l in zip(pos_list[1:], length_list[1:]):
                for i in range(len(combined_pos_list)):
                    b = True
                    for pos in combined_pos_list[i]:
                        if math.sqrt((pos[0]- p[0])**2+(pos[1]-p[1])**2) > 2:
                            b = False
                            break
                    if b:
                        combined_pos_list[i].append(p)
                        combined_length_list[i] += l
                        break
                if not b:
                    combined_pos_list.append([p])
                    combined_length_list.append(l)
            final_pos_list = []
            final_length_list = []
            for combined_pos, combined_length in zip(combined_pos_list, combined_length_list):
                final_pos_list.append(np.sum(np.array(combined_pos), axis=0)/len(combined_pos)) 
                final_length_list.append(combined_length/len(combined_pos))
                            
            return final_pos_list[final_length_list.index(max(final_length_list))]
        return []

    def find_opponent(self):
        opp = [[], []]
        n = 0
        opp_l = 40
        for i in range(40):
            for j in range(40):
                if self.obs[j, i] == self.opp_color:
                    opp[0].append(i)
                    opp[1].append(j)
                    n += 1
                    
                    t = math.sqrt((i-self.ball_center[0])**2+(j-self.ball_center[1])**2)
                    if t < opp_l :
                        opp_l = t
        
        if n:
            opp_center_s = []
            for i in range(2):
                if max(opp[i]) == 39:
                    opp_center_s.append(min(opp[i]) + 3.5)
                elif min(opp[i]) == 0:
                    opp_center_s.append(max(opp[i]) - 3.5)
                else:
                    opp_center_s.append(sum(opp[i])/n)


            self.opp_length_list.append(opp_l)

        else:
            opp_center_s = []
            self.opp_length_list.append(8)
        return opp_center_s,  opp_l
        
    def to_target_obs(self, target):
        an = math.atan2(target[0]-self.ball_center[0], self.ball_center[1]-target[1]) * 180 / math.pi + 90
        if an < 0:
            an += 360


        if an < 180:
            self.back = 0
            return self.toward_run(an)
        else:
            if self.v2center():
                return self.turn_over(an)
            else:
                return self.backward_run(an)
    
    def v2center(self):
        l = math.sqrt((self.pos[0]-300)**2+(self.pos[1]-350)**2) 
        v_l = math.sqrt(self.v[0]**2+self.v[1]**2)
        return ((300-self.pos[0])*self.v[0]+(350*self.pos[1])*self.v[1])/l/v_l > 0.8

    def toward_run(self, angle):
        if angle >= 60 and angle <= 120:
            agent_angle = angle - 90
        elif angle < 60:
            agent_angle = -30
        else:
            agent_angle = 30
        return [[200], [agent_angle]]
    
    def backward_run(self, angle):
        if angle >= 240 and angle <= 300:
            agent_angle = angle - 270
        elif angle > 300:
            agent_angle = 30
        else:
            agent_angle = -30
        return [[-100], [agent_angle]]

    def turn_over(self, angle):
        return [[0], [30 if angle < 270 else -30]]

    def obs2pos(self, target_obs, obs, pos):
        theta = math.atan2((obs[0]-target_obs[0]),(target_obs[1]-obs[1]))
        l = math.sqrt((obs[0]-target_obs[0])**2+(obs[1]-target_obs[1])**2)
        x = l*math.sin(theta+self.angle/180*np.pi)
        y = l*math.cos(theta+self.angle/180*np.pi)
        if self.idx == 0:
            final = [pos[0]+x*5,pos[1]-y*5]
        else:
            final = [pos[0]-x*5,pos[1]+y*5]
        return final

    def get_energy_cost(self):
        if self.trustable:
            if len(self.pos_list) > 1 and self.pos_list[-2][0] > 0:
                v = min(math.sqrt(self.v[0]**2+self.v[1]**2)*10, 120)
                energy = abs(self.force_list[-1] * v / 500) - 20
                self.energy_list.append(energy)
            else:
                self.energy_list.append(-1)
                energy = abs(self.force_list[-1] * 0.2) - 20
                
        else:
            self.energy_list.append(-1)
            energy = abs(self.force_list[-1] * 0.2) - 200
    
        self.energy = max(min(1000, self.energy-energy), 0)

    def strong_rule(self):
        my_pos = [self.pos[0] + self.v[0], self.pos[1] + self.v[1]]
        if math.sqrt((self.pos[0]-self.opp_pos[0])**2 + (self.pos[1]-self.opp_pos[1])**2) < 40:
            opp_pos = self.opp_pos
        else:
            opp_pos = [self.opp_pos[0] + self.opp_v[0], self.opp_pos[1] + self.opp_v[1]]

        opp2center = math.sqrt((300-opp_pos[0])**2+(350-opp_pos[1])**2)
        me2center = math.sqrt((300-my_pos[0])**2+(350-my_pos[1])**2)
        opp2me = math.sqrt((my_pos[0]-opp_pos[0])**2+(my_pos[1]-opp_pos[1])**2)
        if me2center > opp2center:
            target = (300, 350)
        elif me2center == 0 or (opp2center**2+me2center**2-opp2me**2)/(2*opp2center*me2center) > 0.85:
            if opp2center-me2center > 20:
                target = [opp_pos[0] + (300-opp_pos[0]) * 30 / opp2center, opp_pos[1] + (350-opp_pos[1]) * 30 / opp2center]
            else:
                target = (300, 350)
        else:
            target = [opp_pos[0] + (300-opp_pos[0]) * me2center / opp2center, opp_pos[1] + (350-opp_pos[1]) * me2center / opp2center]

        return self.to_target_pos(target, my_pos)

    def pos2obs(self, pos, agent_pos):
        theta = (self.angle/180 + (0.5-self.idx))*np.pi
        angle = math.atan2((agent_pos[1]-pos[1]),(agent_pos[0]-pos[0]))
        l = math.sqrt((pos[0]-agent_pos[0])**2+(pos[1]-agent_pos[1])**2)/5
        x = self.ball_center[0]+l*math.sin(np.pi+angle-theta)
        y = self.ball_center[1]-l*math.cos(np.pi+angle-theta)
        return [x,y]

    def motion_pos_predict(self):
        self.pos[0] += self.v[0]
        self.pos[1] += self.v[1]

        self.v[1] += self.force_list[-1]/100*math.cos(self.angle/180*math.pi)*((0.5-self.idx)*2)
        self.v[0] += self.force_list[-1]/100*math.sin(self.angle/180*math.pi)*((self.idx-0.5)*2)
        v = math.sqrt(self.v[0]**2+self.v[1]**2)/10
        if v > 1:
            self.v[0]/=v
            self.v[1]/=v
    
    def init_rule(self):
        if self.step < 5:
            return [[200], [0]]
        elif self.step > 9:
            angle = 30 * (-1 if (self.step - 7) % 4 < 2 else 1)
            return [[0], [angle]]
        else:
            return [[0], [0]]
            
    def to_target_pos(self, target, my_pos):
        theta = math.atan2((target[1]-my_pos[1])*(0.5-self.idx), (target[0]-my_pos[0])*(0.5-self.idx))/math.pi*180 - 90
        an = (theta - self.angle + 450) % 360
        if an < 180:
            return self.toward_run(an)
        else:
            if self.v2center():
                return self.turn_over(an)
            else:
                return self.backward_run(an)
        
    def save_energy(self, agent_action):
        angle = ((self.angle + agent_action[1][0]+180*(0.5-self.idx))%360)/180*math.pi
        force = agent_action[0][0]/100
        vf = [self.v[0]+math.cos(angle)*force, self.v[1]+math.sin(angle)*force]
        r = max(math.sqrt(vf[0]**2+vf[1]**2)/10, 1)
        vf = [vf[0]/r, vf[1]/r]

        force = min(math.sqrt((vf[0] - self.v[0])**2+(vf[1] - self.v[1])**2)*120, 200)
        return [[force], agent_action[1]]

    def controll(self, observation):
        self.idx = observation['controlled_player_index']

        self.opp_color = 8 + self.idx*2
        
        self.obs = observation['obs']['agent_obs']

        self.state4, state = obs_pre_process(observation['obs']['agent_obs'], self.state4, self.idx)

        if len(self.pos) == 0:
            self.pos = [300, 200 + 300*self.idx]
            self.opp_pos = [300, 500 - 300*self.idx]

        
        opp, opp_l = self.find_opponent()
        if not self.ball_collision and self.opp_length_list[-2] < self.collision_threshold:
            self.ball_collision = True
        self.motion_pos_predict()

        pos_b = self.find_border()
        pos_c, pos_l = self.find_center()

        self.exchange_v = False

        # pos part
        pos_p = copy.deepcopy(self.pos)

        if self.ball_collision:
            error_b = math.sqrt((pos_b[0]-self.pos[0])**2+(pos_b[1]-self.pos[1])**2) if pos_b else 100
            error_c = math.sqrt((pos_c[0]-self.pos[0])**2+(pos_c[1]-self.pos[1])**2) if pos_c else 100
            error_l = math.sqrt((pos_l[0]-self.pos[0])**2+(pos_l[1]-self.pos[1])**2) if pos_l else 100
            
            Error_b_b = max(abs(pos_b[0]-self.pos[0]), abs(pos_b[1]-self.pos[1])) if pos_b else 100
            Error_b_c = max(abs(pos_c[0]-self.pos[0]), abs(pos_c[1]-self.pos[1])) if pos_c else 100
            Error_b_l = max(abs(pos_l[0]-self.pos[0]), abs(pos_l[1]-self.pos[1])) if pos_l else 100
            if self.opp_length_list[-2] < self.collision_threshold:
                if pos_l:
                    self.pos = pos_l
                    self.trustable = True
                elif pos_b:
                    self.pos = pos_b
                    self.trustable = True
                elif pos_c:
                    self.pos = pos_c
                    self.trustable = True
                else:
                    self.trustable = False

            elif min(Error_b_b, Error_b_c, Error_b_l) < self.trust_predict_threshold and self.trustable:
                if Error_b_l > self.trust_predict_threshold and Error_b_l != 100 and not (error_l > 20 and min(error_c, error_l) < 10):
                    self.pos = pos_l
                elif Error_b_l > self.trust_predict_threshold and max(Error_b_c, Error_b_b) > self.trust_predict_threshold and max(Error_b_c, Error_b_b) != 100:
                    self.pos = pos_c if error_c < error_b else pos_b
            else:
                if pos_l and not (error_l > 20 and min(error_c, error_l) < 10):
                    self.pos = pos_l
                    self.trustable = True
                elif pos_b:
                    self.pos = pos_b
                    self.trustable = True
                elif pos_c and pos_b and math.sqrt((pos_b[0]-pos_c[0])**2+(pos_b[1]-pos_c[0])**2) < 20:
                    self.pos = pos_c if error_c < error_b else pos_b
                    self.trustable = True

            if len(self.pos_list) and abs(pos_p[0]-self.pos[0]) + abs(pos_p[1]-self.pos[1]) != 0:
                if  math.sqrt((pos_p[0]-self.pos[0])**2 + (pos_p[1]-self.pos[1])**2) > 5 and self.opp_length_list[-2] < self.collision_threshold and self.opp_v_trust and math.sqrt(self.opp_v[0]**2+self.opp_v[1]**2) < 13:
                    self.exchange_v = True
                    self.v = copy.deepcopy(self.opp_v)
                else:
                    self.v[0] = self.pos[0] - self.pos_list[-1][0]
                    self.v[1] = self.pos[1] - self.pos_list[-1][1]

        self.pos_list.append(copy.deepcopy(self.pos))

        if self.opp_length_list[-2] < self.collision_threshold:
            self.v_trust_list.append(False)
        else:
            self.v_trust_list.append(math.sqrt((pos_p[0]-self.pos[0])**2+(pos_p[1]-self.pos[1])**2)<2)

        # opp part
        if len(opp) > 0:
            self.opp_pos = self.obs2pos(opp, self.ball_center, self.pos)
            self.opp2center_list.append(math.sqrt((self.opp_pos[0]-300)**2+(self.opp_pos[1]-350)**2))
        else:
            self.opp_pos = []
            self.opp2center_list.append(-1)

        self.opp_pos_list.append(self.opp_pos)
        
        if len(self.opp_pos_list) > 1 and len(self.opp_pos_list[-1]) and len(self.opp_pos_list[-2]):
            self.opp_v[0] = self.opp_pos_list[-1][0] - self.opp_pos_list[-2][0]
            self.opp_v[1] = self.opp_pos_list[-1][1] - self.opp_pos_list[-2][1]
            if self.exchange_v:
                self.opp_v_trust = False
        else:
            self.opp_v_trust = False
            self.opp_v = [0, 0]

        self.opp_v_list.append(self.opp_v)

        self.get_energy_cost()

        if len(self.opp_pos) > 0 or (self.pos[1]-350)*(self.idx - 0.5) < 0:
            self.init_mode = False

        if self.init_mode:
            agent_action = self.init_rule()
        elif len(self.opp_pos):
            agent_action = self.strong_rule()
        elif len(pos_b) > 0 or math.sqrt((self.pos[0]-300)**2+(self.pos[1]-350)**2) > 130:
            agent_action = self.to_target_obs(self.pos2obs((300, 350), self.pos))
            l = math.sqrt((self.pos[0]-300)**2+(self.pos[1]-350)**2) 
            v_l = math.sqrt(self.v[0]**2+self.v[1]**2)
            if l == 0:
                agent_action[0][0] = 0
            elif l < 110 and v_l > 9 and ((300-self.pos[0])*self.v[0]+(350*self.pos[1])*self.v[1])/l/v_l > 0.8:
                agent_action[0][0] = min(90, agent_action[0][0])
        else:
            self.back = 0
            action, _, _ = self.model_slow(state)
            agent_action = [[action[0][0] * 10 - 100], [action[0][1] * 2 - 30]]

        if self.energy < 200 and math.sqrt((self.pos[0]-300)**2+(self.pos[1]-350)**2) < 150:
            agent_action[0][0] = min(90, agent_action[0][0])

        if self.init_mode or (self.v_trust_list[-1] and self.v_trust_list[-2] and math.sqrt((self.pos[0]-300)**2+(self.pos[1]-350)**2) < 150):
            agent_action = self.save_energy(agent_action)

        self.step += 1
        self.angle = (self.angle + agent_action[1][0]) % 360
        self.force_list.append(agent_action[0][0])
        return agent_action
    
agent = Agent()

def my_controller(observation, action_space, is_act_continuous=False):
    return agent.controll(observation)
    