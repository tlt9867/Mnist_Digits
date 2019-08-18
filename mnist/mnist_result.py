# -*- coding: utf-8 -*-
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from PIL import Image, ImageFilter
import tensorflow as tf
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import cv2
import logging
import time
from datetime import timedelta

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
KEYSPACE = "mnistkeyspace"
app.send_file_max_age_default = timedelta(seconds=1)

def imageprepare(image_path): 
    im = Image.open(image_path).convert('L') #读取的图片所在路径，注意是28*28像素
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels
# 
    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if nheight == 0:  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
 
#    # newImage.save("sample.png")
# 
    tv = list(newImage.getdata())  # get pixel values
 
#    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
#def imageprepare(file_name):
#    myimage = Image.open(file_name)
#    myimage = myimage.resize((28, 28), Image.ANTIALIAS).convert('L')  #变换成28*28像素，并转换成灰度图
#    tv = list(myimage.getdata())  # 获取像素值
#    tva = [(255-x)*1.0/255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑
#    return tva


def Prediction(image_path):
    x = tf.placeholder(tf.float32, [None, 784])
    
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev = 0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1,shape = shape)
        return tf.Variable(initial)
    
    def conv2d(x,W):
        return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x,[-1,28,28,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/model.ckpt")
        prediction=tf.argmax(y_conv,1)
        predint=prediction.eval(feed_dict={x: [imageprepare(image_path)],keep_prob: 1.0}, session=sess)
        print(predint[0])
    return predint[0]

log = logging.getLogger()

log.setLevel('INFO')

handler = logging.StreamHandler()

handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

log.addHandler(handler)

def createKeySpace(image_path, prediction, time):

   cluster = Cluster(contact_points=['127.0.0.1'],port=9042)

   session = cluster.connect()


   
   try:
       log.info("Creating keyspace...")   
       session.execute("""

           CREATE KEYSPACE %s

           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }

           """ % KEYSPACE)


       log.info("setting keyspace...")

       session.set_keyspace(KEYSPACE)

       log.info("creating table...")

       session.execute("""

           CREATE TABLE Mnist (

               image text,

               prediction int,

               time float,

               PRIMARY KEY (image)

           )

           """)
           
          
   except Exception as e:
       
       if str(e) == "Keyspace 'mykeyspace' already exists":
            session.set_keyspace(KEYSPACE)
            session.execute("""insert into Mnist (image, prediction, time) values ('digits/digitis2.png',4,10)""")
            rows = session.execute('select * from mykeyspace.Mnist')
            for row in rows:#遍历查询的结果
                print(row[0],row[1],row[2])
       else:
           log.error("Unable to create keyspace")
           log.error(e)
           print(e)

@app.route('/result',methods=['Get','Post'])
def get_tasks():
    
    
    html = "<h3>Hello {name}!</h3>" 
    return html.format(name =Prediction())


   
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
 
        basepath = os.path.dirname(__file__)  
 
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  
        f.save(upload_path)
 
        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.png'), img)
        prediction = Prediction(os.path.join(basepath,'static/images','test.png'))
        createKeySpace(upload_path,int(prediction),time.time())
        return render_template('upload_ok.html',pred = prediction, val1=time.time())
 
    return render_template('upload.html')



if __name__  =='__main__':
        app.run(host='0.0.0.0', port=80)
    
