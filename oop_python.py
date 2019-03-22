def is_object_iterable(object_arg):
    try:
        iter(object_arg)
    except TypeError:
        return False
    return True
import pandas as pd

class Filter(object):

    def __init__(self, filters=None):
        self._filters = list()
        if filters is not None:
            if is_object_iterable(filters):
                self._filters += [i for i in filters if callable(i)]
            elif callable(filters):
                self._filters.append(filters)


class ContentFilter:

    def __init__(self, functions=None):
        self.__filter = Filter(functions)

    def filter_metod(self, content):
        for filter in self.__filter._filters:
            content = filter(content)
            # to do for None
        return content


class BooleanFilter:
    def __init__(self, functions=None):
        self.__filter = Filter(functions)

    def filter_metod(self, lst_of_content):
        if is_object_iterable(lst_of_content):
            for filter1 in self.__filter._filters:
                lst_of_content = list(filter(filter1, lst_of_content))
        return lst_of_content