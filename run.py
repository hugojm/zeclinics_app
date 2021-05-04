#!/usr/bin/python3
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
from PIL import Image, ImageTk
import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from display import plate
from pathlib import Path
from flask import json
from process import process_video
import heartpy as hp
import matplotlib
import pickle
import time
import random
import threading


UPLOAD_FOLDER = 'static/images'
UPLOAD_FOLDER_CARDIO = 'static/videos'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'xml'}
ALLOWED_EXTENSIONS_CARDIO = {'lif'}

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_CARDIO'] = UPLOAD_FOLDER_CARDIO
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0





@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        time.sleep(1)
        return redirect('upload.html')
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        time.sleep(1)
        return redirect('upload.html')
    return render_template('index.html')



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file_cardio(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_CARDIO

@app.route('/upload.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        files = request.files.getlist("file[]")
        plate_name = Path(files[1].filename).parent.parts[0]
        for file in files:
            if file and allowed_file(file.filename):
                path_name = Path(file.filename)
                if (Path(os.path.join(app.config['UPLOAD_FOLDER'],path_name.parent)).exists() == False):
                    os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'],path_name.parent))
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], path_name))
        plate(plate_name,app.config['UPLOAD_FOLDER'])
        dirname = os.path.join(app.config['UPLOAD_FOLDER'], plate_name)
        dirs = [ name for name in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, name)) ]
        dirs2 = [dirname+ "/" + sub for sub in dirs]
        dirs2.sort()
        with open('static/dict/booleans.pckl', 'rb') as handle:
            booleans = pickle.load(handle)
        return render_template('terato.html', plates = dirs2, done=True, dict = booleans)
    return render_template('upload.html')

@app.route('/cardio.html', methods=['GET', 'POST'])
def cardio():
    if request.method == "POST":
        files = request.files.getlist("file[]")
        for file in files:
            if file and allowed_file_cardio(file.filename):
                path_name = Path(file.filename)
                path_name = os.path.join('./' , path_name)
                matplotlib.use('agg')
                # masks_a, masks_v , a , v, _ = process_video(path_name,update_it=4, skip=1, debug=True, gen_video=False, video_name=os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out.webm'),mode="ac", p_out_shape="original")
                # hp.plotter(a[1],a[2],show=False, title='Atrium signal').savefig(os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out1.png'))
                # hp.plotter(v[1],v[2],show=False, title='Ventricle signal').savefig(os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out2.png'))
        with open('static/dict/metrics.pckl', 'rb') as handle:
            metrics = pickle.load(handle)
        return render_template('cardio.html', print=True, dict = metrics)
    return render_template('cardio.html', dict={})

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
