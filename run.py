#!/usr/bin/python3
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
from PIL import Image, ImageTk
import os
import torch
from read_roi import read_roi_file
import cv2
import imageio
import scipy.misc
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import terato.display as Tox
from pathlib import Path
from flask import json
from cardio.process import process_video
import heartpy as hp
import matplotlib
import json
import pickle
import time
import random
import threading
from flaskwebgui import FlaskUI  # import FlaskUI
import xml.etree.ElementTree as ET
from tqdm import tqdm

UPLOAD_FOLDER = 'static/images'
UPLOAD_FOLDER_CARDIO = 'static/videos'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'xml'}
ALLOWED_EXTENSIONS_CARDIO = {'lif'}

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_CARDIO'] = UPLOAD_FOLDER_CARDIO
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

ui = FlaskUI(app, width=3000, height=3000, maximized=True)

device = 'cpu'


@app.route('/', methods=['GET', 'POST'])
def index():
    print('Youre in /')
    if request.method == "POST":
        print('In / requested', request.form.to_dict().keys())
        time.sleep(1)
        if list(request.form.to_dict().keys())[0] == 'upload':
            return redirect('upload.html')
        return redirect('upload_cardio.html')
    return render_template('index.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    print("Youre in home")
    if request.method == "POST":
        print('In home requested', request.form.to_dict().keys())
        time.sleep(1)
        if list(request.form.to_dict().keys())[0] == 'upload':
            return redirect('upload.html')
        return redirect('upload_cardio.html')
    return render_template('index.html')


@app.route('/upload_cardio.html', methods=['GET', 'POST'])
def upload_cardio():
    print('You in cardio')
    if request.method == "POST":
        files = request.files.getlist("file[]")
    processed = os.listdir(app.config['UPLOAD_FOLDER_CARDIO'])
    return render_template('upload_cardio.html', process=processed)


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

def dict_from_xml(plate_path, plate_name):
    dic_images = {}
    dic_feno = {}
    xml_path = plate_path + "/" + plate_name + ".xml"
    tree = ET.parse(xml_path)
    plate = tree.getroot()
    for well in tqdm(plate):
        well_name = 'Well_' + well.attrib['name']
        dic_feno[well_name] = {}
        dic_images[well_name] = [well.attrib['dorsal_image'],well.attrib['lateral_image']]
        for feno in well:
            dic_feno[well_name][feno.tag] = feno.attrib['probability']
    return dic_images, dic_feno

#roi path
#mask_name (p ej. eye_up_dorsal, fishoutline_dorsal,...) - >es el nombre del roi sin el .roi
def create_mask(roi_paths, mask_name):

    #colors in BGR
    colors = {
        "eye_up_dorsal": [0,177,204,116],
        "eye_down_dorsal": [0,177,204,116],

        "ov_lateral": [0,255,204,204],
        "yolk_lateral": [0,138,148,241],
        "fishoutline_dorsal": [0,111,220,247],
        "fishoutline_lateral": [0,233,193,133],
        "heart_lateral": [0,85,97,205]
    }

    #read roi and get mask
    img = np.zeros((190,1024,1), np.uint8)
    roi = read_roi_file(roi_paths)[mask_name]


    img = Tox.obtain_mask(img,roi)
    cv2.imwrite(mask_name +'.png', img)

    img = cv2.imread(mask_name +'.png')
    img = (255-img)

    # convert to graky
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

    # negate mask
    mask = 255 - mask

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)

    #set color
    indices = np.where(result==0)
    color = colors[mask_name]
    result[indices[0], indices[1], :] = color
    #set transparency
    result[...,3] = 127

    # save resulting masked image
    cv2.imwrite(mask_name+'.png', result)

@app.route('/upload.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        files = request.files.getlist("file[]")
        for file in files:
            if file and allowed_file(file.filename):
                path_name = Path(file.filename)
                if (not Path(os.path.join(app.config['UPLOAD_FOLDER'], path_name.parent)).exists()):
                    os.makedirs(os.path.join(
                        app.config['UPLOAD_FOLDER'], path_name.parent))
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], path_name))
        plate_name = Path(files[0].filename).parent.parent
        if plate_name == Path('.'):
            plate_name = Path(files[0].filename).parent
        dirname = os.path.join(app.config['UPLOAD_FOLDER'], plate_name)
        Tox.generate_and_save_predictions(str(dirname), batch_size=4,
                                          model_path_seg='static/weight/weights.pt',
                                          model_path_bools='static/weight/weights_bool.pt',
                                          mask_names=[
                                              "outline_lat", "heart_lat", "yolk_lat", "ov_lat", "eyes_dor", "outline_dor"],
                                          feno_names=['bodycurvature', 'yolkedema', 'necrosis', 'tailbending', 'notochorddefects',
                                                      'craniofacialedema', 'finabsence', 'scoliosis', 'snoutjawdefects'],
                                          device=device)
        dirs = [name for name in os.listdir(
            dirname) if os.path.isdir(os.path.join(dirname, name))]
        dirs2 = [dirname + "/" + sub for sub in dirs]
        dirs2.sort()
        images, phenotypes =dict_from_xml(dirname, plate_name)
        return render_template('terato2.html', plates=dirs2, done=True, data=phenotypes, images=images)
    processed = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('upload.html', process=processed)


@app.route('/terato', methods=['GET', 'POST'])
def terato():
    if request.method == 'POST':
        plate_name = request.form['submit_button']
        dirname = os.path.join(app.config['UPLOAD_FOLDER'], plate_name)
        dirs = [name for name in os.listdir(
            dirname) if os.path.isdir(os.path.join(dirname, name))]
        dirs2 = [dirname + "/" + sub for sub in dirs]
        dirs2.sort()
        images, phenotypes =dict_from_xml(dirname, plate_name)
        return render_template('terato2.html', plates=dirs2, plate_name=plate_name, data=phenotypes, images=images)
    else:
        print("fail")


@app.route('/cardio.html', methods=['GET', 'POST'])
def cardio():
    if request.method == "POST":
        files = request.files.getlist("file[]")
        for file in files:
            if file and allowed_file_cardio(file.filename):
                path_name = Path(file.filename)
                path_name = os.path.join('./', path_name)
                matplotlib.use('agg')
                # masks_a, masks_v , a , v, _ = process_video(path_name,update_it=4, skip=1, debug=True, gen_video=False, video_name=os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out.webm'),mode="ac", p_out_shape="original")
                # hp.plotter(a[1],a[2],show=False, title='Atrium signal').savefig(os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out1.png'))
                # hp.plotter(v[1],v[2],show=False, title='Ventricle signal').savefig(os.path.join(app.config['UPLOAD_FOLDER_CARDIO'],'out2.png'))
        with open('static/dict/metrics.pckl', 'rb') as handle:
            metrics = pickle.load(handle)
        return render_template('cardio.html', print=True, dict=metrics)
    else:
        with open('static/dict/metrics.pckl', 'rb') as handle:
            metrics = pickle.load(handle)
        return render_template('cardio.html', print=True, dict=metrics)
    return render_template('cardio.html', dict={})


@app.route('/getmask/', methods=['GET', 'POST'])
def getmask():
    if request.method == "POST":
        data = json.loads(request.data)
        masks = ["eye_up_dorsal","eye_down_dorsal","ov_lateral","yolk_lateral","fishoutline_dorsal","fishoutline_lateral","heart_lateral"]
        for mask in masks:
            create_mask(data+'/'+mask+'.roi',mask)
    return


@app.route('/deletetemp/', methods=['GET', 'POST'])
def deletetemp():
    if request.method == "POST":
        os.remove()
    return 'hola'


@app.route('/deleteplate/', methods=['GET', 'POST'])
def deleteplate():
    if request.method == "POST":
        data = json.loads(request.data)
        try:
            os.rmdir(UPLOAD_FOLDER + '/' + data)
            print('hola2')
        except:
            os.remove(UPLOAD_FOLDER + '/' + data)
            print('hola1')
    return 'hola'


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


if __name__ == "__main__":
    # app.run() for debug
    ui.run()
