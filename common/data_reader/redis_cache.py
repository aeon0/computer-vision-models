from typing import NamedTuple
import random
import string
import pickle
import redis
from common.data_reader import ICache


class RedisConfig(NamedTuple):
    host: str = "localhost"
    port: int = 6379
    maxmemory: str = "4gb"
    expire: int = 7200  # in seconds


class RedisCache(ICache):
    """
    Using redis to in-memory cache training/validation data fetched from another source.
    """
    def __init__(self,
                 config: RedisConfig):
        """
        :param config: RedisConfig instance for configuration
        """
        self.config = config
        self.lock_key = None
        self.r = None
        self.lock_key = ''.join(random.choice(string.ascii_lowercase) for i in range(12))

        # find empty database
        db = 0
        while True:
            self.r = redis.Redis(host=self.config.host, port=self.config.port, db=db)
            if len(self.r.keys()) == 0:
                # add a lock item so the next redis cache will not use that db
                self.r.set(self.lock_key, "lock", self.config.expire)
                self.r.config_set("maxmemory", self.config.maxmemory)
                self.r.config_set("maxmemory-policy", "noeviction")
                print("USING DB: " + str(db))
                break
            elif db > 100:
                break
            db += 1
        if self.r is None:
            raise ValueError("Could not find any empty redis database")

    def exist(self, keys: []):
        """
        Check if a list of keys exist within the cache
        :param keys: array of keys as string
        """
        for key in keys:
            if not self.r.exists(str(key)):
                return False
        return True

    def get(self, keys: list):
        """
        Generator for a list of keys
        :param keys: array of keys as string
        """
        for key in keys:
            byte_data = self.r.get(str(key))
            data = pickle.loads(byte_data)
            if data is not None:
                self.r.expire(str(key), self.config.expire)
                yield data

    def set(self, key, value):
        """
        Set data to a specific key
        :param key: key as string
        :param value: value
        :return: True if it was successful, False otherwise (e.g. if maxmemory is reached)
        """
        try:
            data = pickle.dumps(value)
            self.r.set(str(key), data, self.config.expire)
            return True
        except redis.exceptions.ResponseError:
            return False

    def clear(self):
        """
        Clear the cache
        :return: None
        """
        self.r.flushdb()
