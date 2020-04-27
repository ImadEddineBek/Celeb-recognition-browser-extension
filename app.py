import copy
import json
import os
import sys
from multiprocessing.pool import ThreadPool
import subprocess
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from loguru import logger
from pandas.io.s3 import s3fs
from waitress import serve

from embeddings import Embedder
from index import Node
import math
import random
import os

import pandas as pd
import requests
import s3fs
from PIL import Image
import requests
from io import BytesIO

from index_builder import get_index, get_name, get_brute_force

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))


#
# with open("version") as version_file:
#     VERSION = version_file.read().strip()
#
# THRESHOLD = 0.01
#
# SUBSAMPLE_SIZE = 100
# BATCH_SIZE = 10
#
# HTTP_UI_ADDRESS = os.getenv("HTTP_UI_ADDRESS", "https://hydro-serving.dev.hydrosphere.io")
# S3_ENDPOINT = os.getenv("S3_ENDPOINT")
#
# tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
#                      'min_max': ('min_max', 'same'),
#                      'hull': ('delaunay', 'same')}
# FEATURE_LAKE_BUCKET = "feature-lake"
# BATCH_SIZE = 10


# import random, time, sys
# from tqdm import tqdm_notebook
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.spatial import distance
# import pickle
#
# K_random = 512
# maxsize = 500
# start = time.time()
# R = np.random.rand(maxsize, K_random)
# R = [(row, "stub value {}".format(i)) for i, row in enumerate(R)]
# print(R[:3])
# finish = time.time()
# print("{} rows generated in {:.2f} s".format(len(R), finish - start))
#
# kdtree = Node(K=K_random).build_kd_tree(R)
# kdtree.get_nn(np.random.rand(K_random), 5)[0][1]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)

CORS(app)


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am Celebrity Face Detector"


@app.route("/who", methods=["GET"])
def get_metrics():
    possible_args = {"type", "src"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400
    try:
        type = request.args.get('type')
        src = request.args.get('src')
    except:
        return jsonify({"message": f"Was unable to cast model_version to int"}), 400

    logger.info("Detecting names of {} in image: {}".format(type, src))

    response = requests.get(src)
    img = Image.open(BytesIO(response.content))
    embder = Embedder()
    embeddings = embder.embed_one(img)

    if embeddings is not None:
        P, E = embeddings.size()
        name = ""
        for i in range(P):
            if type == 'friends':
                r = requests.get(f'http://0.0.0.0:5001/who_brute?embedding={list(embeddings[i].numpy())}')
                name += r.text + "    "
            else:
                r = requests.get(f'http://0.0.0.0:5002/who_brute?embedding={list(embeddings[i].numpy())}')
                name += r.text + "    "
            # name += get_brute_force(embeddings[i], 'celebs.index') + "    "
            # json_dump = json.dumps({"person": name}, cls=NumpyEncoder)
        if len(name) < 2:
            return "Could not Find in database"
        return name
    else:
        return "no face detected"

    # cluster = Cluster(HTTP_UI_ADDRESS)
    # model = Model.find(cluster, model_name, model_version)
    #
    # d2 = get_subsample(model, size=SUBSAMPLE_SIZE, s3fs=fs)
    # d1 = get_training_data(model, fs)
    #
    # input_fields_names = [field.name for field in model.contract.predict.inputs]
    #
    # d1 = d1[input_fields_names]
    # d2 = d2[input_fields_names]
    #
    # types1, types2 = get_types(d1), get_types(d2)
    # d1 = d1.values
    # d2 = d2.values
    # (c1, cf1), (c2, cf2) = get_disc(d1, types1), get_disc(d2, types2)
    # (d1, f1), (d2, f2) = get_cont(d1, types1), get_cont(d2, types2)
    # stats1, stats2 = get_all(d1), get_all(d2)
    # histograms = get_histograms(d1, d2, f1)
    #
    # full_report = {}
    # pool = ThreadPool(processes=1)
    # async_results = {}
    # for test in tests:
    #     async_results[test] = pool.apply_async(one_test, (d1, d2, test))
    # for test in tests:
    #     full_report[test] = async_results[test].get()
    # per_statistic_change_probability(full_report)
    #
    # final_report = {'per_feature_report': fix(f1, stats1, histograms, stats2,
    #                                           per_statistic_change_probability(full_report),
    #                                           per_feature_change_probability(full_report)),
    #                 'overall_probability_drift': overall_probability_drift(full_report)}
    #
    # warnings = {'final_decision': final_decision(full_report),
    #             'report': interpret(final_report['per_feature_report'])}
    #
    # final_report['warnings'] = warnings
    #
    # if cf1 and len(cf1) > 0:
    #     for training_feature, deployement_feature, feature_name in zip(c1, c2, cf1):
    #         final_report['per_feature_report'][feature_name] = process_one_feature(training_feature,
    #                                                                                deployement_feature)
    #
    # json_dump = json.dumps(final_report, cls=NumpyEncoder)
    # return json.loads(json_dump)


@app.route("/config", methods=['GET', 'PUT'])
def get_params():
    global THRESHOLD
    if request.method == 'GET':
        return jsonify({'THRESHOLD': THRESHOLD})

    elif request.method == "PUT":
        possible_args = {"THRESHOLD"}
        if set(request.args.keys()) != possible_args:
            return jsonify(
                {"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

        logger.info('THRESHOLD changed from {} to {}'.format(THRESHOLD, request.args['THRESHOLD']))
        THRESHOLD = float(request.args['THRESHOLD'])
        return Response(status=200)
    else:
        return Response(status=405)


def launch_instances():
    # print(sys.executable)
    DETACHED_PROCESS = 0x00000008
    my_env = os.environ.copy()
    # my_env["INDEX_TYPE"] = 'friends.index'
    # my_env["PORT"] = '5001'
    # logger.info('here')
    # logger.info(subprocess.check_output([sys.executable, "index_builder.py"]))
    # os.spawnl(os.P_NOWAIT, 'INDEX_TYPE=friends.index PORT=5001' + sys.executable + ' index_builder.py', [''])
    # os.spawnl(os.P_NOWAIT, 'INDEX_TYPE=celebs.index PORT=5002' + sys.executable + ' index_builder.py', [''])
    logger.info('Instances are up')
    pid = subprocess.Popen(
        [sys.executable, "index_builder.py", '--port', str(5001), '--index_type', 'friends.index']).pid
    logger.info("friends: pid {}, port {}".format(pid, 5001))

    pid = subprocess.Popen(
        [sys.executable, "index_builder.py", '--port', str(5002), '--index_type', 'celebs.index']).pid
    logger.info("celebrities: pid {}, port {}".format(pid, 5002))

    # my_env = os.environ.copy()
    # my_env["INDEX_TYPE"] = 'celebs.index'
    # my_env["PORT"] = '5002'
    #
    # pid = subprocess.call([sys.executable, "index_builder.py"], env=my_env,
    #                       ).pid
    # logger.info("pid ", pid)
    # os.spawnl(os.P_NOWAIT, 'INDEX_TYPE=friends.index PORT=5001' + sys.executable + 'index_builder')
    # os.spawnl(os.P_NOWAIT, 'INDEX_TYPE=celebs.index PORT=5002' + sys.executable + 'index_builder')


if __name__ == "__main__":
    launch_instances()

    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
