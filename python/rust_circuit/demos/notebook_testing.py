import contextlib
from typing import ClassVar


class NotebookDoneTestingException(Exception):
    """
    A demo/notebook can raise this exception when __name__ != "__main__" to be tested only up to some point.
    """

    pass


class NotebookInTesting(contextlib.ContextDecorator):
    currently_in_notebook_test: ClassVar[bool] = False

    def __enter__(self):
        if self.__class__.currently_in_notebook_test:
            raise RuntimeError("NotebookInTesting context manager not nestable")
        self.__class__.currently_in_notebook_test = True
        return self

    def __exit__(self, exc_type: Exception, exc_val, exc_tb) -> bool:
        # Cannot nest this context manager correctly, because on exiting we always
        # set this variable to False.
        # To nest it properly, we'd have to remember what the value was on __enter__: too much work.
        self.__class__.currently_in_notebook_test = False
        if exc_type == NotebookDoneTestingException:
            return True  # Ignore exception
        return False

    @classmethod
    def exit_if_in_testing(cls):
        if cls.currently_in_notebook_test:
            raise NotebookDoneTestingException
