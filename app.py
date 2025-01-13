from flask import Flask, render_template, request, redirect, url_for, flash,session
import cv2
from forms import RegistrationForm  
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask_mysqldb import MySQL
from datetime import datetime
import mysql.connector

import tensorflow as tf
import sys
sys.path.append('StressPackage')
sys.path.append('HeartRatePackage')

from collections import Counter
from StressPackage.main import *
from HeartRatePackage.GetHeartRate import *

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'projuser'  
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'users'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.secret_key = 'your_secret_key'

mysql = MySQL(app)


@app.route('/user_register_actions', methods=['POST'])
def user_register_actions():
    if request.method == 'POST':
        name = request.form['name']
        loginid = request.form['loginid']
        password = request.form['password']
        mobile = request.form['mobile']
        email = request.form['email']
        roadno = request.form['roadno']
        city = request.form['city']
        state = request.form['state']
        pincode = request.form['pincode']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM registration_details WHERE loginid = %s", [loginid])
        existing_user = cur.fetchone()
        
        if existing_user:
            flash('Login ID already exists. Please choose another.', 'error')
            return redirect(url_for('register_page'))
         
        cur.execute("INSERT INTO registration_details (name, loginid, password, mobile, email, roadno, city, state, pincode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (name, loginid, password, mobile, email, roadno, city, state, pincode))
        mysql.connection.commit()
        cur.close()
        
        flash('User registered successfully!', 'success')
        return redirect(url_for('login_page'))

    
@app.route('/user_login_check', methods=['POST'])
def user_login_check():
    if request.method == 'POST':
        loginid = request.form['loginid']
        password = request.form['password']

        cur = mysql.connection.cursor()

        cur.execute("SELECT * FROM registration_details WHERE loginid = %s AND password = %s", (loginid, password))
        
        user = cur.fetchone()  

        if user:
            flash('Login successful!', 'success')
            session['user']={'name':user[0],'loginid':user[1], 'password':user[2], 'mobile':user[3], 'email':user[4], 'roadno':user[5], 'city':user[6], 'state':user[7], 'pincode':user[8]}
            return redirect(url_for('index'))  
        else:
            flash('Invalid login credentials. Please try again.', 'error')
            return redirect(url_for('login_page')) 

@app.route('/dashboard', endpoint='dashboard_page')
def dashboard():
    user=session.get('user')
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM image where loginId = %s",(user.loginId))
    image_data = cur.fetchall()
    cur.execute("SELECT * FROM livestream where loginId = %s",(user.loginId))
    livestream_data = cur.fetchall()
    cur.close()
    return render_template('dashboard.html', imagedata=image_data, videodata=livestream_data)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login', endpoint='login_page')
def login():
    return render_template('Login.html')

@app.route('/register',methods=['GET','POST'], endpoint='register_page')
def register():
    form = RegistrationForm(request.form)

    return render_template('Registerations.html',form=form)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/stress_detection_image', methods=['GET', 'POST'])
def stress_detection_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file:
            filename = file.filename
            img_path = os.path.join('uploads', filename)
            file.save(img_path)

            img = cv2.imread(img_path)
            er=DetectFace()
            frame, stress_result ,resized_frame= er.predict_stress(img)
            print(img_path,"prediction is ",stress_result)  

            predarr = []
            health_tip="NA"
            if (stress_result=="Stressed"):
                health_tip="Do Meditation"
            
            flash(f'Stress level detected: {stress_result}', 'success')
            if health_tip != "NA":
                flash(f"{health_tip} to decrease stress")
            cv2.imshow("Stress Detection Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cur = mysql.connection.cursor()
            current_datetime = datetime.now()
            current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
            image_insert_query = "INSERT INTO image (filename, stress, health_tip, cdate) VALUES (%s, %s, %s, %s)"
            image_data = (filename,stress_result, health_tip, current_datetime_str) 
            cur.execute(image_insert_query, image_data)
            mysql.connection.commit()

            cur.close()
            return redirect(url_for('index'))

@app.route('/stress_heartrate_detection_video')
def stress_heartrate_detection_video():
        print("Streaming Started")
        er = DetectFace()
        cap = cv2.VideoCapture(0)
        stress_arr=[]
        video_frames=[]
        i=0
        while True:
            ret, frame = cap.read()
            try:
            
             frame, stress,resized_frame = er.predict_stress(frame)
             stress_arr.append(stress)
             print(stress)
             video_frames.append(resized_frame)
            except Exception as e:
             print("Hold the camera properly")
            
                
            cv2.imshow("Press q to exit",frame)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        Cntr=Counter(stress_arr)
        Cntr_filtered = {key: value for key, value in Cntr.items() if key != "No Face"}
        
        if Cntr_filtered:
            stressedornot=max(Cntr_filtered, key=Cntr_filtered.get)
            health_tip="NA"
            if stressedornot=="Stressed":
                health_tip="Do Meditation"
            heartrateobj=HeartRateCalci(np.array(video_frames))
            heartrate=heartrateobj.HeartRateMethod()
            flash(f"Stress:{stressedornot},Heart Rate:{heartrate}" )
            if health_tip != "NA":
                flash(f"{health_tip} to decrease stress")
            cur = mysql.connection.cursor()
            current_datetime = datetime.now()
            current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

            video_data = (stressedornot,heartrate, health_tip, current_datetime_str)  
            video_insert_query = "INSERT INTO livestream (stress, heart_rate, health_tip, cdate) VALUES (%s, %s, %s, %s)"
            cur.execute(video_insert_query, video_data)

            mysql.connection.commit()

            cur.close()
        else:
            flash(f"Stress,Heart Rate cannot be found as there is no face found" )

        return redirect(url_for('index'))
            

if __name__ == '__main__':
    app.run(debug=True)
