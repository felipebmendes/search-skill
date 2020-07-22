import itertools
import logging
import numpy as np
import os
import re
from flask import Blueprint, current_app, flash, jsonify, request, redirect, render_template, url_for
from pycarol import Carol, Storage, Query
from pycarol.apps import Apps
from pycarol.filter import Filter, TYPE_FILTER, TERMS_FILTER
from sentence_transformers import SentenceTransformer
from webargs import fields, ValidationError
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
_settings = Apps(login).get_settings()

server_bp = Blueprint('main', __name__)

def update_embeddings():
    global sentence_embeddings
    global index_to_question_id_mapping
    global question_id_to_index_mapping
    global index_to_filter_mapping
    global filter_to_index_mapping
    
    # Get files from Carol storage
    logger.debug('Loading sentence embeddings.')
    sentence_embeddings_tmp = get_file_from_storage('sentence_embeddings')
    logger.debug('Loading index to question id mapping.')
    index_to_question_id_mapping_tmp = get_file_from_storage('index_to_question_id_mapping')
    logger.debug('Loading question id to index mapping.')
    question_id_to_index_mapping_tmp = get_file_from_storage('question_id_to_index_mapping')
    logger.debug('Loading index to filter mapping.')
    index_to_filter_mapping_tmp = get_file_from_storage('index_to_filter_mapping')
    logger.debug('Loading filter to index mapping.')
    filter_to_index_mapping_tmp = get_file_from_storage('filter_to_index_mapping')
    logger.debug('Done')

    # Update values after all of them are loaded from Carol storage
    sentence_embeddings = sentence_embeddings_tmp
    index_to_question_id_mapping = index_to_question_id_mapping_tmp
    question_id_to_index_mapping = question_id_to_index_mapping_tmp
    index_to_filter_mapping = index_to_filter_mapping_tmp
    filter_to_index_mapping = filter_to_index_mapping_tmp

def get_file_from_storage(filename):
    return storage.load(filename, format='pickle', cache=False)

def get_questions_by_ids(question_ids):
    dm_name = _settings['data_model_name']
    dm_field = _settings['data_model_field']
    query_filter = Filter.Builder().must(TYPE_FILTER(value=f'{dm_name}Golden')) \
        .must(TERMS_FILTER(key=f'mdmGoldenFieldAndValues.{dm_field}.raw', value=question_ids)).build().to_json()
    page_size=1000
    return Query(login, page_size=page_size).query(query_filter).go().results

def get_similar_questions(model, sentence_embeddings, query, threshold, filter, type_filter, k):
    query_vec = model.encode([query])
    logger.debug(query)
    logger.debug(threshold)
    logger.debug(filter)
    logger.debug(type_filter)
    logger.debug(k)
    score = np.sum(query_vec[0] * sentence_embeddings, axis=1) / np.linalg.norm(sentence_embeddings, axis=1)
    topk_scores = np.sort(score)[::-1]
    topk_idx = np.argsort(score)[::-1]
    topk_mapping = {idx:topk_scores[i] for i,idx in enumerate(topk_idx)}
    if filter:
        if type_filter == 'include':
            topk_idx = [idx for idx in topk_idx if idx in filter]
        else:
            topk_idx = [idx for idx in topk_idx if idx not in filter]
        topk_scores = [topk_mapping[idx] for idx in topk_idx]
    topk_scores = topk_scores[:k]
    topk_scores = [score for score in topk_scores if score >= threshold/100]
    topk_idx = topk_idx[:len(topk_scores)]
    return list(topk_idx), list(topk_scores)

sentence_embeddings = None
index_to_question_id_mapping = None
question_id_to_index_mapping = None
index_to_filter_mapping = None
filter_to_index_mapping = None

# Get files from Carol storage
update_embeddings()    

# Load Sentence model
logger.debug('Loading pre-trained model.')
model = SentenceTransformer('/app/model/')

logger.debug('Done')

@server_bp.route('/', methods=['GET'])
def ping():
    return jsonify('App is running. Send a request to /query')


@server_bp.route('/update_embeddings', methods=['GET'])
def update_embeddings_route():
    update_embeddings()
    return jsonify('Embeddings are updated.')


@server_bp.route('/query', methods=['POST'])
def query():
    query_arg = {
        "query": fields.Str(required=True, 
            description='Query to be searched in the documents.'),
        "threshold": fields.Int(required=False, missing=55, description='Documents with scores below this threshold \
            are not considered similar to the query. Default: 55.'),
        "k": fields.Int(required=False, missing=5, description='Number of similar documents to be return. \
            Default: 5.'),
        "filter": fields.List(fields.Str(), required=False, missing=None, validate=validate_filter, description='Filter value to be used \
            after ranking the documents.'),
        "type_filter": fields.Str(required=False, missing=None, validate=lambda x: x in ['include', 'exclude'],
            description='Type of filter to be used. Required if filter is provided. If "include", similar documents that \
                match the specified filter are returned. If "exclude", similar documents that do not match the specified \
                filter are returned.'),
    }
    args = parser.parse(query_arg, request, validate=validate_type_filter)
    query = args['query']
    threshold = args['threshold']
    k = args['k']
    filter = args['filter']
    type_filter = args['type_filter']
    responses = []
    if filter:
        filter_idx = [filter_to_index_mapping[x] for x in filter]
        filter_idx = list(itertools.chain.from_iterable(filter_idx))
    else:
        filter_idx = None
    topk_idx, topk_scores = get_similar_questions(model, sentence_embeddings, query, threshold, filter_idx, type_filter, k)
    logger.debug(f'topk_idx: {topk_idx}')
    if not topk_idx:
        return {'session_id': 1, 'response': responses}
    question_ids = [index_to_question_id_mapping[idx] for idx in topk_idx]
    logger.debug(f'questions: {question_ids}')
    results = get_questions_by_ids(question_ids)
    logger.debug(f'results: {results}')
    if not results:
        return {'session_id': 1, 'response': responses}
    responses = []
    for i, question_id in enumerate(question_ids):
        result = [result for result in results if result['id'] == question_id][0]
        response = {
            'matched_question': result['question'],
            'answer': result['solution'],
            'score': str(topk_scores[topk_idx.index(question_id_to_index_mapping[question_id])]),
            'matched_question_id': question_id
        }
        output = {
                'type': 'text',
                'content':  response
            }
        responses.append(output)

    return {'session_id': 1, 'response': responses}


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

def validate_filter(val):
    if filter and not filter_to_index_mapping:
        raise ValidationError("This api does not accept filter. Please update your model.")
    keys = [*filter_to_index_mapping]
    if filter and any(x not in keys for x in val):
        if len(keys) < 11:
            raise ValidationError(f"Invalid value. Options are: {', '.join(keys)}")
        raise ValidationError("Invalid value.")

def validate_type_filter(args):
    if args['filter'] and not args['type_filter']:
        raise ValidationError("A type filter: 'exclude' or 'include' should be provided when a filter is used.")
