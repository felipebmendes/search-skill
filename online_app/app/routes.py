import logging
import numpy as np
import os
from flask import Blueprint, current_app, flash, jsonify, request, redirect, render_template, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.urls import url_parse
from .carol_login import carol_login, CarolUser, carol_logout
from .forms import LoginForm
from pycarol import Carol, Storage, Query
from pycarol.filter import Filter, TYPE_FILTER, TERMS_FILTER
from sentence_transformers import SentenceTransformer
from webargs import fields
from webargs.flaskparser import parser

# Logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

login = Carol()
storage = Storage(login)

server_bp = Blueprint('main', __name__)

def get_file_from_storage(filename):
    return storage.load(filename, format='pickle', cache=False)

def get_questions_by_ids(question_ids):
    dm_name = 'lgpdarticles'
    query_filter = Filter.Builder().must(TYPE_FILTER(value=f'{dm_name}Golden')) \
        .must(TERMS_FILTER(key='mdmGoldenFieldAndValues.id.raw', value=question_ids)).build().to_json()
    page_size=1000
    return Query(login, page_size=page_size).query(query_filter).go().results

def get_similar_questions(model, sentence_embeddings, query, k):
    query_vec = model.encode([query])
    score = np.sum(query_vec[0] * sentence_embeddings, axis=1) / np.linalg.norm(sentence_embeddings, axis=1)
    topk_scores = np.sort(score)[::-1]
    topk_idx = np.argsort(score)[::-1]
    return list(topk_idx[:k]), list(topk_scores[:k])
    
# Get files from Carol storage
logger.debug('Loading sentence embeddings.')
sentence_embeddings = get_file_from_storage('sentence_embeddings')
logger.debug('Loading index to question id mapping.')
index_to_question_id_mapping = get_file_from_storage('index_to_question_id_mapping')
logger.debug('Loading question id to index mapping.')
question_id_to_index_mapping = get_file_from_storage('question_id_to_index_mapping')

# Load Sentence model
logger.debug('Loading pre-trained model.')
model = SentenceTransformer('/app/model/')

logger.debug('Done')

@server_bp.route('/', methods=['GET'])
def ping():
    return jsonify('App is running. Send a request to /query')

@server_bp.route('/query', methods=['POST'])
def query():
    query_arg = {
        "query": fields.Str(required=True, 
            description='Query cannot be blank.')
    }
    args = parser.parse(query_arg, request)
    query = args['query']
    k = 5
    topk_idx, topk_scores = get_similar_questions(model, sentence_embeddings, query, k)
    question_ids = [index_to_question_id_mapping[idx] for idx in topk_idx]
    results = get_questions_by_ids(question_ids)
    responses = []
    for i, question_id in enumerate(question_ids):
        response = ''
        result = [result for result in results if result['id'] == question_id][0]
        response = response + 'Pergunta similar: ' + result['question'] + '\n'
        response = response + 'Resposta: ' + result['solution'] + '\n'
        response = response + 'Score: ' + str(topk_scores[topk_idx.index(question_id_to_index_mapping[question_id])])
        if i != 4:
            response = response +  '\n\n'
        output = {
                'type': 'text',
                'content':  response
            }
        responses.append(output)

    return {'session_id': 1, 'response': responses}

#TODO: disambiguate

@server_bp.errorhandler(422)
@server_bp.errorhandler(400)
def handle_error(err):
    headers = err.data.get("headers", None)
    messages = err.data.get("messages", ["Invalid request."])
    messages = messages.get('json', messages)
    if headers:
        return jsonify(messages), err.code, headers
    else:
        return jsonify(messages), err.code
