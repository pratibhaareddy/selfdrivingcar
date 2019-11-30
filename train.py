import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import model
import load_data


session = tf.InteractiveSession()
t = tf.subtract(model.y_, model.y)

loss = tf.reduce_mean(tf.square(t)) + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.001
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
session.run(tf.initialize_all_variables())

tf.summary.scalar("loss", loss) #create summary
all_summary =  tf.summary.merge_all() #merge summaries

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

summation = tf.summary.FileWriter('./records', graph=tf.get_default_graph())

# Training the dataset for 30 iterations
for epoch in range(30):
  for i in range(int(load_data.no_of_images/100)):
    split_x, split_y = load_data.LoadTrainData(100)
    train_step.run(feed_dict={model.x: split_x, model.y_: split_y, model.keep_prob: 0.8})
    if i % 10 == 0:
      split_x, split_y = load_data.LoadTestData(100)
      loss_value = loss.eval(feed_dict={model.x:split_x, model.y_: split_y, model.keep_prob: 1.0})
    
    z = epoch * load_data.no_of_images/100 + i
    summation.add_summary(all_summary.eval(feed_dict={x:split_x, y_: split_y, model.keep_prob: 1.0}), z)

    if i % batch_size == 0:
      if not os.path.exists('./saved'):
        os.makedirs('./saved')
      checkpoint_path = os.path.join('./saved', "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model will be saved" % filename)

print("try running otherwise")
