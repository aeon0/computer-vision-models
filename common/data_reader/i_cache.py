from abc import ABCMeta, abstractmethod


class ICache(metaclass=ABCMeta):
    @abstractmethod
    def get(self, keys: list):
        """
        Get the data for a certain key from the cache
        :param keys: array of keys with which the cache saved the data
        :return: data if it is found, None otherwise
        """

    @abstractmethod
    def set(self, key: str, value: str):
        """
        Set data for a certain key in the cache
        :param key: key with which the data is saved
        :param value: data that should be saved to the key
        :return: True if successful, False otherwise
        """

    def exist(self, keys: list):
        """
        :param keys: array of keys
        :return: True if keys exist, false otherwise
        """

    def clear(self):
        """
        Clear cache
        :return: None
        """
