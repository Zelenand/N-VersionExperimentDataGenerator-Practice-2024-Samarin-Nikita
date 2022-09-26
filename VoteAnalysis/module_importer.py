from importlib.util import find_spec, module_from_spec


def check_import(module, package):
    """Проверка возможности импорта.
    :param module: имя импортируемого модуля
    :param package: путь до модуля
    :return: спецификацию модуля или None если модуль не найден
    """
    module_spec = find_spec(f'.{module}', package=package)
    return module_spec


def import_module_from_spec(module_spec):
    """Импорт модуля.
    :param module_spec: спецификация модуля
    :return: импортированный модуль
    """
    res_module = None
    if module_spec is not None:
        module = module_from_spec(module_spec)
        try:
            module_spec.loader.exec_module(module)
            res_module = module
        except ModuleNotFoundError as e:
            error_str = f'Module not found {e.msg}'
            print(error_str)
            raise ModuleNotFoundError(error_str)
        except Exception as e:
            error_str = f'Something is wrong - unexpected exception {e}'
            print(error_str)
            raise e
    else:
        print('Module specification does not exists')

    return res_module


class ModuleNotLoadedError(Exception):
    pass


class FunctionNotFoundInModuleError(Exception):
    pass
