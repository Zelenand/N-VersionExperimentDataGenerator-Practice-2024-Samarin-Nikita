"""Database access module for versions and modules data

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""
import json

from dataaccess.data_base_connector import DBConnector
from userinput.user_input import input_num

DB_NAME = 'experiment_edu.db'


def nversion_save(nversion, module_id=None):
    """Save version data to database"""
    cur_conn = DBConnector(DB_NAME)
    if not cur_conn.table_exists('version'):
        create_query = 'create table version ("id" integer primary key autoincrement not null, '
        create_query += '"name" varchar(255) not null, "const_diversities_coordinates" varchar(512) null, '
        create_query += '"dynamic_diversities_coordinates" varchar(512) null, "reliability" real not null, '
        create_query += '"module" integer null, foreign key ("module") references module(id));'
        cur_conn.execute_query(create_query, [], True, False)

    if nversion.id is None:
        additional_query_str = ', NULL'
        if module_id is not None:
            additional_query_str = f', {module_id}'

        insert_query = f"insert into version (name, const_diversities_coordinates, dynamic_diversities_coordinates,"
        insert_query += f" reliability, module) values ('{nversion.name}', "
        insert_query += f"'{json.dumps({nversion.formal_json_db_lst_name: nversion.const_diversities})}', "
        insert_query += f"'{json.dumps({nversion.formal_json_db_lst_name: nversion.dynamic_diversities})}', "
        insert_query += f"{nversion.reliability}" + additional_query_str + ");"
        print(insert_query)
        # Т.к. у нас возвращается список кортежей, берём первый элемент первого кортежа, т.к. id сего 1 возвращется!
        return int(cur_conn.execute_query(insert_query, [], True)[0][0])
    else:
        update_query = f"update version set name = '{nversion.name}', "
        update_query += f"const_diversities_coordinates = "
        update_query += f"'{json.dumps({nversion.formal_json_db_lst_name: nversion.const_diversities})}', "
        update_query += f"dynamic_diversities_coordinates = "
        update_query += f"'{json.dumps({nversion.formal_json_db_lst_name: nversion.dynamic_diversities})}', "
        update_query += f"reliability = {nversion.reliability}, module = {module_id} where id = {nversion.id};"
        cur_conn.execute_query(update_query, [], True, False)


def nversion_load(nversion):
    """Load version data from database"""
    cur_conn = DBConnector('experiment_edu.db')
    if cur_conn.table_exists('version'):
        if nversion.id is None:
            select_query = 'select name, const_diversities_coordinates, dynamic_diversities_coordinates, '
            select_query += 'reliability, module from version order by id;'
            q_set = cur_conn.execute_query(select_query)
            get_num_str = 'Choice version by id to load it.\n'
            for mdl in q_set:
                get_num_str += f'\n{str(mdl)}'
            get_num_str += '\n'
            chosen_id = input_num(get_num_str)
        else:
            chosen_id = nversion.id

        select_query = f'select id, name, const_diversities_coordinates, dynamic_diversities_coordinates, '
        select_query += f'reliability, module from version where id = {chosen_id}'
        select_res = cur_conn.execute_query(select_query)

        return select_res
    else:
        raise LookupError(
            f'There is no "VERSION" table in {DB_NAME} data base. Save module data before load it.'
        )


def nversion_load_versions_2_module(module_id: int):
    """Load versions data of certain module from database"""
    if module_id is not None and type(module_id) is int:
        cur_conn = DBConnector(DB_NAME)
        if cur_conn.table_exists('version'):
            select_query = f'select id, name, const_diversities_coordinates, dynamic_diversities_coordinates, '
            select_query += f'reliability, module from version where module = {module_id}'
            select_res = cur_conn.execute_query(select_query)
            return select_res
        else:
            raise LookupError(
                f'There is no "VERSION" table in {DB_NAME} data base. Save module data before load it.'
            )
    else:
        raise AttributeError(f'Invalid module_id parameter. Int is expected. {str(module_id)} was got.')


def nmodule_save(nmodule):
    """Save nmodule data to database"""
    cur_conn = DBConnector(DB_NAME)
    if not cur_conn.table_exists('module'):
        create_query = 'create table module ("id" integer primary key autoincrement not null, '
        create_query += '"name" varchar(255) not null, "round_to" integer not null, '
        create_query += '"dynamic_diversities_intervals" varchar(1024) null, '
        create_query += '"const_diversities_count" integer not null, "dynamic_diversities_count" integer null, '
        create_query += '"min_out_val" real not null, "max_out_val" real null);'
        cur_conn.execute_query(create_query, [], True, False)

    if nmodule.id is None:
        insert_query = f"insert into module (name, round_to, dynamic_diversities_intervals, const_diversities_count"
        insert_query += f", dynamic_diversities_count, min_out_val, max_out_val) values ('{nmodule.name}', "
        insert_query += f"{nmodule.round_to}, '{json.dumps(nmodule.dynamic_diversities_intervals_dict)}', "
        insert_query += f"{nmodule.const_diversities_count}, {nmodule.dynamic_diversities_count}, {nmodule.min_out_val}, "
        insert_query += f'{nmodule.max_out_val});'
        # Т.к. у нас возвращается список кортежей, берём первый элемент первого кортежа, т.к. id сего 1 возвращется!
        return cur_conn.execute_query(insert_query, [], True)[0][0]
    else:
        update_query = f"update module set name = '{nmodule.name}', round_to = {nmodule.round_to}, "
        update_query += f"dynamic_diversities_intervals = '{json.dumps(nmodule.dynamic_diversities_intervals_dict)}',"
        update_query += f"const_diversities_count = '{nmodule.const_diversities_count}', "
        update_query += f"dynamic_diversities_count = {nmodule.dynamic_diversities_count}, "
        update_query += f"min_out_val = {nmodule.min_out_val}, max_out_val = {nmodule.max_out_val} where id = {nmodule.id};"
        cur_conn.execute_query(update_query, [], True, False)


def nmodule_save_experiment_data(nmodule):
    """Save experiment data of nmodule from database"""
    if len(nmodule.global_results_lst_to_write) > 0:
        cur_conn = DBConnector(DB_NAME)
        if not cur_conn.table_exists('experiment_data'):
            create_query = 'create table experiment_data ("id" integer primary key autoincrement not null, '
            create_query += 'version_id integer not null, version_name varchar(255), version_reliability real not '
            create_query += 'null, version_common_coordinates varchar(1024) not null, version_answer real not null,'
            create_query += 'correct_answer real not null, module_id integer not null, module_name varchar(255), '
            create_query += 'module_connectivity_matrix varchar(4095), module_iteration_num int not null, '
            create_query += 'experiment_name varchar(31) not null, '
            create_query += 'unique(version_id, version_reliability, version_common_coordinates, version_answer, '
            create_query += 'correct_answer, module_id, module_connectivity_matrix, module_iteration_num) '
            create_query += 'on conflict replace, foreign key ("version_id") references version(id), '
            create_query += 'foreign key ("module_id") references module(id));'
            cur_conn.execute_query(create_query, [], True, False)

        # Обращаемся к первому результату первой итерации для провеки, присвоен ли ему id, чтобы понять, надо ли
        # сохранять результаты, или они уже сохранены, т.к. можно только перегенерировать их, но не изменить
        if nmodule.global_results_lst[0][0].id is None:
            insert_query = 'insert into experiment_data (version_id, version_name, version_reliability, '
            insert_query += 'version_common_coordinates, version_answer, correct_answer, module_id, module_name, '
            insert_query += 'module_connectivity_matrix, module_iteration_num, experiment_name) '
            insert_query += ' values(?,?,?,?,?,?,?,?,?,?,?);'
            cur_conn.execute_query(insert_query, nmodule.global_results_lst_to_write, True, False)
    else:
        raise LookupError('There is no experiment data to save into data base')


def nmodule_load_experiment_data(nmodule, experiment_name: str = None):
    """Load experiment data of nmodule from database"""
    cur_conn = DBConnector(DB_NAME)
    if cur_conn.table_exists('experiment_data'):
        can_we_go_further = False
        if experiment_name is None and nmodule.experiment_name is not None:
            experiment_name = nmodule.experiment_name
            can_we_go_further = True

        if experiment_name is not None:
            can_we_go_further = True

        if can_we_go_further:
            select_query = "select id, version_id, version_name, version_reliability, version_common_coordinates, "
            select_query += "version_answer, correct_answer, module_id, module_name, module_connectivity_matrix, "
            select_query += "module_iteration_num, experiment_name from experiment_data where experiment_name = "
            select_query += f"'{experiment_name}';"

            select_res = cur_conn.execute_query(select_query)
            return select_res
        else:
            raise ValueError(
                f'Unexpected value {experiment_name} of experiment_name parameter!'
            )
    else:
        raise LookupError(
            f'There is no "EXPERIMENT_DATA" table in {nmodule.db_name} data base. Save experiment data before load it'
        )


def get_experiments_names():
    """Get experiment names from database."""
    cur_conn = DBConnector(DB_NAME)
    if cur_conn.table_exists('experiment_data'):
        experiment_select_res = cur_conn.execute_query("select distinct experiment_name from experiment_data;")
        return [exp_name[0] for exp_name in experiment_select_res]


def nmodule_load_module(nmodule):
    """Load module data from database"""
    cur_conn = DBConnector(DB_NAME)
    if cur_conn.table_exists('module'):
        if nmodule.id is None:
            q_set = cur_conn.execute_query('select distinct id, name, round_to from module order by id;')
            get_num_str = 'Choice module by id to load it.\n'
            for mdl in q_set:
                get_num_str += f'\n{str(mdl)}'
            get_num_str += '\n'
            chosen_id = input_num(get_num_str)
        else:
            chosen_id = nmodule.id

        select_query = f'select id, name, round_to, dynamic_diversities_intervals, const_diversities_count, '
        select_query += f'dynamic_diversities_count, min_out_val, max_out_val from module where id = {chosen_id};'
        select_res = cur_conn.execute_query(select_query)

        return select_res
