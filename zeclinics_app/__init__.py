#!/usr/bin/python3
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
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
from cardio.process import process_video, ecg, save_csv
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
import glob
from terato.exploratory_analysis import pca, Mca, doseperresponse
import pandas as pd
import shutil

f = open('processing.txt', 'w')
f.close()

root_package = Path(os.path.abspath(__file__)).parent
print(root_package)
static_path = root_package / 'static'
templates_path = root_package / 'templates'

UPLOAD_FOLDER = str(static_path/'images')
UPLOAD_FOLDER_CARDIO = str(static_path/'videos')

if not os.path.isdir(str(static_path / 'temp')):
    os.makedirs(str(static_path / 'temp'))
if not os.path.isdir(str(static_path / 'temp/terato')):
    os.makedirs(str(static_path / 'temp/terato'))
if not os.path.isdir(str(static_path / 'temp/plots')):
    os.makedirs(str(static_path / 'temp/plots'))


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'xml'}
ALLOWED_EXTENSIONS_CARDIO = {'lif'}

app = Flask(__name__, static_folder=str(static_path), template_folder = str(templates_path))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_CARDIO'] = UPLOAD_FOLDER_CARDIO
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

ui = FlaskUI(app, width=3000, height=3000, maximized=True,start_server = "flask")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        fps = int(request.form['fps'])
        base_it = int(request.form['base_it'])
        update_it = int(request.form['update_it'])
        skip_it = int(request.form['skip_it'])
        for file in files:
            if file and allowed_file_cardio(file.filename):
                path_name = Path(file.filename)
                path_name = os.path.join('./', path_name)
                out = path_name.split('/');
                lif = out[len(out)- 1];
                lif = lif.split('.')[0]
                lif_path = os.path.join(app.config['UPLOAD_FOLDER_CARDIO'], lif)
                if not Path(lif_path).exists():
                    os.mkdir(lif_path)
                _,_ , a , v, metrics,_ = process_video(path_name,base_it=base_it, update_it=update_it, skip=skip_it, debug=True, gen_video=True, video_name=str(static_path / 'videos')+'/'+lif+'/video.webm', p_out_shape="original", fps = fps)
                path = os.path.join('./', lif_path)
                ecg(a,v, os.path.join(path,'ecg.html'), save=True)
                _ = save_csv(metrics, path+'/')
                return render_template('cardio.html', dict=metrics, lif = lif)
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
    i = 0
    listdir = os.listdir(plate_path)
    well_name_example = listdir[i]
    while well_name_example[:4] != 'Well':
        i += 1
        well_name_example = listdir[i]

    dic_images = {}
    dic_feno = {}
    xml_path = str(plate_path) + "/" + str(plate_name) + ".xml"
    tree = ET.parse(xml_path)
    plate = tree.getroot()
    for well in tqdm(plate):
        if 'well_folder' in well.attrib:
            well_name = well.attrib['well_folder']
        else:
            well_name = well_name_example[:-3] + well.attrib['name']
        if int(well.attrib['show2user']):
            dic_feno[well_name] = {}
            dic_images[well_name] = [well.attrib['dorsal_image'],well.attrib['lateral_image']]
            for feno in well:
                if 'probability' in feno.attrib:
                    dic_feno[well_name][feno.tag] = feno.attrib['probability']
                elif 'value' in feno.attrib:
                    dic_feno[well_name][feno.tag] = feno.attrib['value']
                else:
                    dic_feno[well_name][feno.tag] = -1
    return dic_images, dic_feno

#roi path
#mask_name (p ej. eye_up_dorsal, fishoutline_dorsal,...) - >es el nombre del roi sin el .roi
def create_mask(roi_paths, mask_name, well):
    #colors in BGR
    colors = {
        "eye_up_dorsal": [177,204,116,0],
        "eye_down_dorsal": [177,204,116,0],

        "ov_lateral": [255,204,204,0],
        "yolk_lateral": [138,148,241,0],
        "fishoutline_dorsal": [111,220,247,0],
        "fishoutline_lateral": [233,193,133,0],
        "heart_lateral": [85,97,205,0]
    }
    #read roi and get mask
    img = np.zeros((190,1024,1), np.uint8)
    roi = read_roi_file(roi_paths)[mask_name]

    pts = zip(roi['x'],roi['y'])
    pts2 = np.array(list(pts), 'int32')
    #Important to put the brackets []!!!!
    cv2.fillPoly( img , [pts2], (255))

#    img = Tox.obtain_mask(img,roi)
    cv2.imwrite(str(static_path / 'temp/terato')+'/'+well + mask_name +'.png', img)

    img = cv2.imread(str(static_path / 'temp/terato')+'/'+well + mask_name +'.png')
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
    result[:, :, 3] = mask

    # save resulting masked image
    cv2.imwrite(str(static_path / 'temp/terato')+'/'+well + mask_name+'.png', result)

def generate_plots(plate_path, plate):
    print(plate_path,plate)
    df = pd.read_csv(os.path.join(plate_path, 'stats.csv'))
    pca(df)
    if not Path(str(static_path / 'temp/plots')+'/'+plate+'/').exists():
        os.mkdir(str(static_path / 'temp/plots')+'/'+plate+'/')
    Mca(df,str(static_path / 'temp/plots')+'/'+plate+'/mca.png')
    doseperresponse(df, str(static_path / 'temp/plots')+'/'+plate+'/')
    os.rename('biplot_2d.png', str(static_path / 'temp/plots')+'/'+plate+'/biplot_2d.png')




@app.route('/upload.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        files = request.files.getlist("file[]")
        plate_name = Path(files[0].filename).parent.parent
        if plate_name == Path('.'):
            plate_name = Path(files[0].filename).parent
        dirname = os.path.join(app.config['UPLOAD_FOLDER'], plate_name)
        if str(plate_name) not in os.listdir(app.config['UPLOAD_FOLDER']):
            with open('processing.txt', 'w') as f:
                f.write(str(plate_name) + '\n')
            for file in files:
                if file and allowed_file(file.filename):
                    path_name = Path(file.filename)
                    if (not Path(os.path.join(app.config['UPLOAD_FOLDER'], path_name.parent)).exists()):
                        os.makedirs(os.path.join(
                            app.config['UPLOAD_FOLDER'], path_name.parent))
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], path_name))
            Tox.generate_and_save_predictions(str(dirname), batch_size=4,
                                              model_path_seg=str(static_path / 'weight/weights.pt'),
                                              model_path_bools=str(static_path / 'weight/weights_bool.pt'),
                                              mask_names=[
                                                  "outline_lat", "heart_lat", "yolk_lat", "ov_lat", "eyes_dor", "outline_dor"],
                                              feno_names=['bodycurvature', 'yolkedema', 'necrosis', 'tailbending', 'notochorddefects',
                                                          'craniofacialedema', 'finabsence', 'scoliosis', 'snoutjawdefects'],
                                              device=device)
            with open('processing.txt', 'r') as f:
                output = []
                for line in f:
                    if str(plate_name) != line.strip():
                        output.append(line)
                f.close()
                f = open('processing.txt', 'w')
                f.writelines(output)
                f.close()

        with open('processing.txt') as f:
            processing = True
            while processing:
                for line in f:
                    if line.strip() == str(plate_name):
                        break
                processing = False
        dirs = [name for name in os.listdir(
            dirname) if os.path.isdir(os.path.join(dirname, name))]
        dirs2 = [dirname + "/" + sub for sub in dirs]
        dirs2.sort()
        images, phenotypes =dict_from_xml(dirname, plate_name)
        return render_template('terato2.html', plates=dirs2, done=True, data=phenotypes, images=images)
    if not Path(app.config['UPLOAD_FOLDER']).exists():
        os.mkdir(app.config['UPLOAD_FOLDER'])
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
        generate_plots(dirname, plate_name)
        return render_template('terato2.html', plates=dirs2, plate_name=plate_name, data=phenotypes, images=images)
    else:
        print("fail")


def csv_to_dict(csv):
    df = pd.read_csv(csv, header=None)
    metrics = {}
    for i in range(len(df.iloc[0,:])):
        metrics[df.iloc[0,i]] = df.iloc[1,i]
    return metrics

@app.route('/download', methods=['GET', 'POST'])
def download():
    plate = request.args.get('plate', None)
    shutil.make_archive(app.config['UPLOAD_FOLDER'] +'/'+ plate, 'zip', os.path.join(app.config['UPLOAD_FOLDER'], plate))
    zip_file = send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename=plate+'.zip', as_attachment=True)
    os.remove(app.config['UPLOAD_FOLDER'] +'/'+ plate + '.zip')
    return zip_file

@app.route('/download_cardio', methods=['GET', 'POST'])
def download_cardio():
    lif = request.args.get('lif', None)
    shutil.make_archive(app.config['UPLOAD_FOLDER_CARDIO'] +'/'+lif, 'zip', os.path.join(app.config['UPLOAD_FOLDER_CARDIO'], lif))
    zip_file = send_from_directory(directory=app.config['UPLOAD_FOLDER_CARDIO'], filename=lif+'.zip', as_attachment=True)
    os.remove(app.config['UPLOAD_FOLDER_CARDIO'] +'/'+ lif + '.zip')
    return zip_file

@app.route('/cardio', methods=['GET', 'POST'])
def cardio():
    if request.method == "POST":
        lif_name = request.form['submit_button']
        lif_path = os.path.join(app.config['UPLOAD_FOLDER_CARDIO'], lif_name)
        metrics = csv_to_dict(os.path.join(lif_path, lif_name+'.csv'))
        return render_template('cardio.html', dict=metrics, lif = lif_name)
    return render_template('cardio.html', dict={})

@app.route('/graphics')
def graphics():
    plate = request.args.get('plate', None)
    for file in os.listdir(str(static_path / 'temp/plots') +'/'+ plate):
        print(file)
    return render_template('graphics.html', plate=plate)

@app.route('/getmask/', methods=['GET', 'POST'])
def getmask():
    if request.method == "POST":
        data = json.loads(request.data)
        out = data.split('/');
        well = out[-1];
        masks = ["eye_up_dorsal","eye_down_dorsal","ov_lateral","yolk_lateral","fishoutline_dorsal","fishoutline_lateral","heart_lateral"]
        for mask in masks:
            '''
            try:
                create_mask(data+'/'+mask+'.roi',mask, well)
            except:
                print('fail')
            '''
            create_mask(data+'/'+mask+'.roi',mask, well)
    return 'Created mask'


@app.route('/deletetemp/', methods=['GET', 'POST'])
def deletetemp():
    if request.method == "POST":
        print('prueba')
        files = glob.glob(str(static_path / 'temp/terato/*'))
        for f in files:
            print(f)
            os.remove(f)
    return 'hola'


@app.route('/deleteplate/', methods=['GET', 'POST'])
def deleteplate():
    if request.method == "POST":
        data = json.loads(request.data)
        try:
            shutil.rmtree(UPLOAD_FOLDER + '/' + data)
            print('hola2')
        except:
            os.remove(UPLOAD_FOLDER + '/' + data)
            print('hola1')
    return 'hola'

@app.route('/deletelif/', methods=['GET', 'POST'])
def deletelif():
    if request.method == "POST":
        data = json.loads(request.data)
        try:
            shutil.rmtree(UPLOAD_FOLDER_CARDIO + '/' + data)
            print('hola2')
        except:
            os.remove(UPLOAD_FOLDER_CARDIO + '/' + data)
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

def start():
    app.run()

start()
