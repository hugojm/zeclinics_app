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



UPLOAD_FOLDER = 'static/images'
UPLOAD_FOLDER_CARDIO = 'static/videos'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'xml'}
ALLOWED_EXTENSIONS_CARDIO = {'lif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_CARDIO'] = UPLOAD_FOLDER_CARDIO
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0





@app.route('/')
def index():
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

@app.route('/terato.html', methods=['GET', 'POST'])
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
        return render_template('terato.html', plates = dirs2, done=True)
    return render_template('terato.html', done=False)

@app.route('/cardio.html', methods=['GET', 'POST'])
def cardio():
    if request.method == "POST":
        files = request.files.getlist("file[]")
        for file in files:
            if file and allowed_file_cardio(file.filename):
                path_name = Path(file.filename)
                path_name = os.path.join('./' , path_name)
                matplotlib.use('agg')
                masks , a , v, _ =process_video(path_name,update_it=1,skip=40, debug=True, gen_video=True, video_name=os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out.mp4'))
                hp.plotter(a[5],a[6],show=False, title='Atrium signal').savefig(os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out1.png'))
                hp.plotter(v[5],v[6],show=False, title='Ventricle signal').savefig(os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out2.png'))
        return render_template('cardio.html', print=True)
    return render_template('cardio.html')

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
