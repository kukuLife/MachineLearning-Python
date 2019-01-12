import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient ():


	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate=0.01,
			reward_decay=0.95,
			output_graph=False
	): 
	
		self.ep_obs, self.ep_as, self.ep_vs = [], [], []
		self.gamma = reward_decay
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self._build_net()
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		


	def _build_net(self):
		with tf.name_scope('inputs'):
			self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations") 
			self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actionss_num")
			self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")


		layer = tf.layers.dense(
			inputs = self.tf_obs,	
			units = 10,
			activation=tf.nn.tanh,
			kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc1'
		)

		all_layers = tf.layers.dense(
			inputs=layer,
			units=self.n_actions,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc2'
		)

		self.all_act_prob = tf.nn.softmax(all_layers, name='act_prob')

		with tf.name_scope('loss'):

			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_layers, labels=self.tf_acts)
			loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

		with tf.name_scope('train'):

			self.train_option = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_vs.append(r)
		print("------ep_vs--------")
		print(self.ep_vs)
		print("------ep_vs--------")

	def choose_action(self, observation):
		prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action

	def descern_diposit_place(self):
		discounted_ep_rs = np.zeros_like(self.ep_vs)
		running_add = 0
		for t in reversed (range(0, len(self.ep_vs))):
			running_add = running_add * self.gamma + self.ep_vs[t]
			discounted_ep_rs[t] = running_add

		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)

		return discounted_ep_rs

	def learn(self):
		discounted_diposit_ep_rs_norm = self.descern_diposit_place()
		print("----------self.ep_obs---------")
		print(self.ep_obs)
		print("----------self.ep_obs---------")
		self.sess.run(
			self.train_option, feed_dict={
				self.tf_obs: np.vstack(self.ep_obs),
				self.tf_acts: np.array(self.ep_as),
				self.tf_vt: discounted_diposit_ep_rs_norm
			})
		return discounted_diposit_ep_rs_norm

