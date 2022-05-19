# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:34:00 2021

@author: sindura saraswathi

The AI tool to explore, enrich and analyse your data
"""
"""import libraries"""
import os 
import io
import re
from flask import Flask, render_template, request, redirect, url_for
import flask_excel
from pandas import read_csv, read_excel, DataFrame, concat
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from base64 import b64encode
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)
#set app config to return correct template instead of wrong cached template
app.config['TEMPLATES_AUTO_RELOAD'] = True
snapshot_dict = {}


def label_encoder(df):
    dict_mapper = {}
    for idx, _value in enumerate(df.unique()):
        dict_mapper[_value] = idx
    df = df.replace(dict_mapper)
    return df


def splitter(df, n , encode_part):
    split_1 = df.apply(lambda x:x[0:n])
    split_2 = df.apply(lambda x: x[n:])
    split_1 = split_1.infer_objects()
    split_2 = split_2.infer_objects()
    if encode_part == 1:
        split_1 = label_encoder(split_1)
    elif encode_part == 2:
        split_2 = label_encoder(split_2)
    return split_1, split_2


def date_extractor(df):
    year = df.dt.year
    month = df.dt.month
    day = df.dt.day
    return year, month, day


def scale_df(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)
    scaled_df = DataFrame(scaled_data, columns=df.columns)
    scaled_df = scaled_df.fillna('-000')
    return scaled_df


@app.route('/')
def home_page():
    return render_template('home_page.html', files=snapshot_dict.keys())


@app.route('/upload')
def upload_files():
    return render_template('upload_files.html', files=snapshot_dict.keys())


@app.route('/upload/success/', methods = ["POST"])
def upload_success():
    global f
    newfile = request.files.getlist("file")
    for i in newfile:
        if i.filename.endswith('.csv'):
            snapshot_dict[i.filename] = read_csv(i)  
        if i.filename.endswith('.xlsx'):
            snapshot_dict[i.filename] = read_excel(i)
        f = snapshot_dict[i.filename]
    return render_template('upload_success.html', files=snapshot_dict.keys())


@app.route('/database/')
def database():
    data = snapshot_dict.keys()
    return render_template('list_files.html', database=data,
                           files=snapshot_dict.keys())


@app.route('/database/<file>/view/')
def file_content(file):
    """view the uploaded file"""
    global f
    f = snapshot_dict[file]
    f = f.infer_objects()
    display_content = concat([f.head(10),f.tail(10)], ignore_index=True).values
    return render_template('file_content.html',
                           length=len(f),
                           column_names=f.columns,
                           datatype=f.dtypes,
                           rows=display_content,
                           files=snapshot_dict.keys())


@app.route('/filecontent/view/', methods=["POST"])
def file_view_range():
    """View the uploaded file content based on the range entered"""
    if request.method == 'POST':
        global f
        from_range = int(request.form['from_range'])
        to_range = int(request.form['to_range'])
        return render_template('file_view_range.html',
                               length=len(f),
                               column_names=f.columns,
                               datatype=f.dtypes,
                               rows=f[(from_range-1):to_range].values,
                               files=snapshot_dict.keys())
    
    
@app.route('/database/profiling/', methods=["POST"])
def profiling():
    """pandas profiling for data"""
    if request.method == "POST":
        output_html_path = r'C:/Users/sindu/Work/MS_folder/Spring_2022/AI_Tool/templates/'
        if 'output.html' in os.listdir(output_html_path):
            os.remove(output_html_path + 'output.html')
        global f
        col_profile = request.form.getlist('col_profile')
        if col_profile == []:
            col_profile = f.columns
        prof = ProfileReport(f[col_profile])
        prof.to_file(output_html_path + 'output.html')
        return render_template('output.html')
    
    
@app.route('/database/dtypechange/', methods=["POST"])
def datatype_change():
    """preprocess stage to change data type"""
    if request.method == "POST":
        global f
        global datatype_df
        global dtypes_to_change
        global preprocess_df
        global col_dict
        datatype_df = f
        dtypes_to_change = request.form.getlist('datatype')
        dtype_dict = {f.columns[i]:dtypes_to_change[i] for i in range(len(f.columns))}
        datatype_df = f.astype(dtype_dict)
        col_dict = {}
        for i in f.columns:
            col_dict[i] = datatype_df[i]
        preprocess_df = datatype_df 
        display_content = concat([datatype_df.head(10), datatype_df.tail(10)],
                                 ignore_index=True).values
        return render_template('preprocessing.html',
                               column_names=f.columns,
                               datatype=datatype_df.dtypes,
                               rows=display_content,
                               files=snapshot_dict.keys())


@app.route('/preprocess/', methods=["POST", "GET"])
def preprocess():
    """preprocess the uploaded data"""
    if request.method == "POST":
        global f
        global col_dict
        global preprocess_df
        prep_list = request.form.getlist('prep')
        split = request.form.getlist('split')
        for j in range(len(f.columns)):
            if prep_list[j] == 'Drop_column':
                preprocess_df[f.columns[j]] = col_dict[f.columns[j]]
                preprocess_df = preprocess_df.drop(columns=[f.columns[j]])
            if prep_list[j] == 'Restore':
                preprocess_df[f.columns[j]] = col_dict[f.columns[j]]
            if prep_list[j] == 'Encoding':
                df = col_dict[f.columns[j]]
                preprocess_df[f.columns[j]] = label_encoder(df)
            if prep_list[j] == 'Splitting':
                df = col_dict[f.columns[j]]
                n = int(split[j])
                split_1, split_2 = splitter(df, n)
                preprocess_df = preprocess_df.drop(columns=[f.columns[j]])
                preprocess_df[f.columns[j]+'_1'] = split_1
                preprocess_df[f.columns[j]+'_2'] = split_2 
            if prep_list[j] == 'Split-encode-1':
                df = col_dict[f.columns[j]]
                n = int(split[j])
                split_1, split_2 = splitter(df, n, 1)
                preprocess_df = preprocess_df.drop(columns=[f.columns[j]])
                preprocess_df[f.columns[j]+'_1'] = split_1
                preprocess_df[f.columns[j]+'_2'] = split_2 
            if prep_list[j] == 'Split-encode-2':
                df = col_dict[f.columns[j]]
                n = int(split[j])
                split_1, split_2 = splitter(df, n, 2)
                preprocess_df = preprocess_df.drop(columns=[f.columns[j]])
                preprocess_df[f.columns[j]+'_1'] = split_1
                preprocess_df[f.columns[j]+'_2'] = split_2 
            if prep_list[j] == 'split-date':
                df = col_dict[f.columns[j]]
                year, month, day = date_extractor(df)
                preprocess_df = preprocess_df.drop(columns=[f.columns[j]])
                preprocess_df[f.columns[j]+'_yr'] = year
                preprocess_df[f.columns[j]+'_mnth'] = month
                preprocess_df[f.columns[j]+'_day'] = day
        return render_template('modelling.html',
                               files=snapshot_dict.keys())
    return render_template('modelling.html', files=snapshot_dict.keys())


@app.route('/snapshot/', methods=["POST"])
def snapshot_save():
    if request.method == "POST":
        global preprocess_df
        file_name = request.form['file_name']
        snapshot_dict[file_name] = preprocess_df 
        return redirect(url_for('preprocess'))
    
            
@app.route('/snapshot/<data>/view')
def snapshot_view(data):
    global f
    f = snapshot_dict[data]
    display_content = concat([f.head(10), f.tail(10)], ignore_index=True).values
    return render_template('file_content.html',
                           column_names=f.columns,
                           datatype=f.dtypes,
                           rows=display_content,
                           files=snapshot_dict.keys())


@app.route('/modelling/', methods=["POST"])
def modelling():
    """data modelling with respect to given clustering options"""
    if request.method == "POST":
        global f
        global preprocess_df
        global validate_df
        validate_df = DataFrame()
        validate_df = validate_df.append(f)
        len_df = len(validate_df)
        algorithm = request.form.getlist('algo')
        dataset_kmeans = request.form['dataset_kmeans']
        dataset_dbscan = request.form['dataset_dbscan']
        dataset_agc = request.form['dataset_agc']
        if dataset_kmeans == 'preprocess_df':
            data1_df = preprocess_df.values 
        if dataset_kmeans == 'scaled_df':
            data1_df = scale_df(preprocess_df)
            data1_df = data1_df.values
        # if dataset == 'pca_df':
            # data_df = pca_df
        if dataset_dbscan == 'scaled_df':
            data2_df = scale_df(preprocess_df)  
            data2_df = data2_df.values
        if dataset_agc == 'preprocess_df':
            data3_df = preprocess_df.values 
        if dataset_agc == 'scaled_df':
            data3_df = scale_df(preprocess_df)
            data3_df = data3_df.values
        if 'kmeans' in algorithm:
            clusters = int(request.form['cluster'])
            kmeans_model = KMeans(n_clusters=clusters,
                                  algorithm='elkan',
                                  random_state=1)
            prediction_kmeans = kmeans_model.fit_predict(data1_df)
            validate_df['KMeans_Cluster_labels'] = prediction_kmeans
        if 'dbscan' in algorithm:
            minsamples = int(request.form['min-samples'])
            epsilon = int(request.form['epsilon'])
            db_model = DBSCAN(min_samples=minsamples, eps=epsilon)
            prediction_dbscan = db_model.fit_predict(data2_df)
            validate_df['DBSCAN_cluster_labels'] = prediction_dbscan
        if 'agglomerative' in algorithm:
            agc_clusters = int(request.form['agc-cluster'])
            agc_model = AgglomerativeClustering(n_clusters=agc_clusters)
            prediction_agc = agc_model.fit_predict(data3_df)
            validate_df['AGC_Cluster_labels'] = prediction_agc
        return render_template('validation_view.html',
                               length=len_df,
                               column_names=validate_df.columns,
                               rows=validate_df.head(20).values,
                               files=snapshot_dict.keys())
    
    
@app.route('/elbow/kmeans/', methods=["POST"])
def elbow_method_kmeans():
    """elbow curve for the preprocessed data"""
    if request.method == "POST":
        global preporcess_df
        from_cluster = int(request.form['fromcluster'])
        to_cluster = int(request.form['tocluster'])
        dataset = request.form['data']
        if dataset == 'preprocess_df':
            data_df = preprocess_df.values
        if dataset == 'scaled_df':
            data_df = scale_df('preprocess_df')
            data_df = data_df.values
            
        sse_list = []
        cluster_size = range(from_cluster, to_cluster)
        
        for i in cluster_size:
            cluster = KMeans(n_clusters=i, random_state=1)
            cluster.fit(data_df)
            sse_list.append(cluster.inertia_)
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(cluster_size, sse_list, marker='*')
        plt.xticks(cluster_size)
        plt.grid(True)
        plt.title('The elbow method to find optimum numer of cluster')
        plt.xlabel('Number of Clusters')
        
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        pic = output.getvalue()
        encoded = b64encode(pic).decode("utf-8")
        return render_template("plot_graph.html", image=encoded)


@app.route('/kdistancegraph/dbscan/', methods=["POST"])
def k_distance_graph():
    """dbscan for clustering the data"""
    if request.method == "POST":
        global preprocess_df
        dataset = request.form['kdata']
        if dataset == "scaled_df":
            data_df = scale_df(preprocess_df)
            data_df = data_df.values
        neigh = NearestNeighbors()
        nbrs = neigh.fit(data_df)
        distance, indices = nbrs.kneighbors(data_df)
        distance = np.sort(distance, axis=0)
        distance = distance[:,1]
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(distance)
        plt.title('K-distance Graph', fontsize=20)
        plt.xlabel('data Points sorted by distance', fontsize=14)
        plt.ylabel('Epsilon', fontsize=14)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        pic = output.getvalue()
        encoded = b64encode(pic).decode("utf-8")
        return render_template("plot_graph.html", image=encoded)
   
    
@app.route('/validation/', methods=["POST"])
def validation_view_range():
    if request.method == "POST":
        global validate_df
        from_range = int(request.form['from_range'])
        to_range = int(request.form['to_range'])
        return render_template('validation_view.html',
                               length=len(validate_df),
                               column_names=validate_df.columns,
                               rows=validate_df[(from_range-1):(to_range)].values,
                               files=snapshot_dict.keys())
    
    
@app.route('/download/')
def downloadFile():
    global validate_df 
    val_dict = {}
    for i in validate_df.columns:
        val_dict[i] = list(validate_df[i])
    return flask_excel.make_response_from_dict(val_dict,
                                               'xlsx',
                                               file_name="result")
        
    
if '__main__' == __name__:
    flask_excel.init_excel(app)
    app.run()