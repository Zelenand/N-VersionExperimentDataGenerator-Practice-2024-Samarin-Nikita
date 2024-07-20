"""Database access module for vote algorithms data

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
import inspect
import json

from dataaccess import data_base_connector

DB_NAME = 'experiment_edu.db'


def save_vote_algorithm(vote_algorithm):
    """Saves the vote_algorithm object in the database."""
    cur_conn = data_base_connector.DBConnector(DB_NAME)
    if not cur_conn.table_exists('algorithm'):
        create_query = 'create table algorithm (id integer primary key autoincrement not null, '
        create_query += 'name varchar(127) not null, src_code text null, module_name varchar(255) not null, '
        create_query += 'func_name varchar(127) not null, module_pkg varchar(255) null);'
        cur_conn.execute_query(create_query, [], True, False)

    if vote_algorithm.id is None:
        insert_query = f"insert into algorithm (name, src_code, module_name, func_name, module_pkg) values ("
        insert_query += f"'{str(vote_algorithm.name)}', '{json.dumps(inspect.getsource(vote_algorithm.vote_module))}', "
        insert_query += f"'{vote_algorithm.module_name}', '{vote_algorithm.vote_func_name}', 'VoteAlgorithms');"
        # Возвращается список кортежей, берём первый элемент первого кортежа, т.к. id всего 1 возвращется!
        vote_algorithm.id = cur_conn.execute_query(insert_query, [], True)[0][0]
    else:
        update_query = f"update algorithm set name = '{vote_algorithm.name}', "
        update_query += f"src_code = '{json.dumps(inspect.getsource(vote_algorithm.vote_algorithm))}', "
        update_query += f"module_path = '{vote_algorithm.module_name}' where id = {vote_algorithm.id};"
        cur_conn.execute_query(update_query, [], True, False)


def save_vote_results(vote_algorithm):
    """Saves the vote result in the database."""
    cur_conn = data_base_connector.DBConnector(DB_NAME)
    if not cur_conn.table_exists('vote_result'):
        create_query = 'create table vote_result (id integer primary key autoincrement not null, '
        create_query += 'algorithm_id integer not null, experiment_data_id integer not null, '
        create_query += 'vote_answer real null, unique(algorithm_id, experiment_data_id, vote_answer) '
        create_query += 'on conflict replace, foreign key ("algorithm_id") references algorithm(id)'
        create_query += 'foreign key ("experiment_data_id") references experiment_data(id));'
        cur_conn.execute_query(create_query, [], True, False)

    res_insert_lst = list()
    for res in vote_algorithm.vote_result:
        for iteration_res in res['data']:
            res_insert_lst.append((vote_algorithm.id, iteration_res.id, res['res']))
    if len(res_insert_lst) > 0:
        insert_query = 'insert into vote_result (algorithm_id, experiment_data_id, vote_answer) values(?,?,?);'
        cur_conn.execute_query(insert_query, res_insert_lst, True, False)
        return True


def load_vote_result_id(algorithm_id, iteration_res_id, res):
    """Load the vote_result from the database."""
    cur_conn = data_base_connector.DBConnector(DB_NAME)
    select_query = f"select id from vote_result where algorithm_id = {algorithm_id} and "
    select_query += f"experiment_data_id = {iteration_res_id} and vote_answer = {res};"

    return cur_conn.execute_query(select_query)


def load_algorithms() -> list:
    """Load all vote algorithms from database"""
    cur_conn = data_base_connector.DBConnector(DB_NAME)
    if cur_conn.table_exists('algorithm'):
        select_query = 'select id, name, func_name, module_name, src_code, module_pkg from algorithm order by id;'
        return cur_conn.execute_query(select_query)
    else:
        raise LookupError(
            f'There is no "algorithm" table in {DB_NAME} data base. Save some algorithm before load it.')
