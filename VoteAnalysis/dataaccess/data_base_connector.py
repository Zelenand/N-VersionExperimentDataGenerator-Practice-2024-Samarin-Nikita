"""Data base query execution module

Program for simulation several N-versions work of one module to test vote algorithms.
Experiment is carried out in Denis V. Gruzenkin PhD thesis writing.
"""

import sqlite3

__author__ = "Denis V. Gruzenkin"
__copyright__ = "Copyright 2021, Denis V. Gruzenkin"
__credits__ = ["Denis V. Gruzenkin"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Denis V. Gruzenkin"
__email__ = "gruzenkin.denis@good-look.su"
__status__ = "Production"


class DBConnectorMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DBConnector(metaclass=DBConnectorMeta):
    def __init__(self, db_name: str):
        self._db_name = db_name
        self._connection = sqlite3.connect(self._db_name)

    @property
    def db_name(self) -> str:
        return self._db_name

    @db_name.setter
    def db_name(self, new_name: str):
        self._db_name = new_name

    def connect_2_db(self):
        self._connection = sqlite3.connect(self._db_name)

    def execute_query(self, query_str, parameters=None, commit=False, return_query_set=True):
        """
        It executes query string with or without parameters and commit and return query set if it is necessary
        :param query_str: SQL-query string for execution
        :param parameters: list ot tuples query parameters, by default is empty
        :param commit: Is needed commit query flag, by default is False
        :param return_query_set: Is needed return query set flag, by default is True
        :return: SQL query set or None
        """
        if parameters is None:
            parameters = []
        many_parameters = False
        if len(parameters) > 0:
            # Если в качестве параметров передан список или кортеж, содержащий составной тип,
            prev_param_len = len(parameters[0])
            for param in parameters:
                if type(param) not in (tuple, list) or prev_param_len != len(param):
                    break
                prev_param_len = len(param)
            else:  # то считаем, что в запрос передано множество параметров (скорее всего на вставку)
                many_parameters = True

        cursor = self._connection.cursor()

        if many_parameters:
            cursor.executemany(query_str, parameters)
        else:
            cursor.execute(query_str)

        if commit:
            self._connection.commit()

        result = None
        if return_query_set:
            # Если выполняется запрос на вставку, и необходимо вернуть результат, то возвращаем id добавленной строки
            query_lst = query_str.lower().split(' ')
            if query_lst[0] == 'insert' and self.table_exists(query_lst[2]):
                # Форма запроса на вставку: insert into table_name
                #                              0     1      2
                # Соответственно имя таблицы будет под индексом 2
                cursor.execute(f'SELECT id FROM {query_lst[2]} WHERE rowid=last_insert_rowid();')
            result = cursor.fetchall()

        cursor.close()

        return result

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        print(check_query)
        result = False
        if len(self.execute_query(check_query)) > 0:
            result = True
        return result

    def __str__(self):
        return f'{self._db_name} – {str(self._connection)}'
